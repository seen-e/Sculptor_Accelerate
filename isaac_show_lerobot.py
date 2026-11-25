#!/usr/bin/env python3
"""Replay one LeRobot episode's joint states in Isaac Sim (kinematic) next to the dataset's own
video frames, to eyeball image <-> joint-state alignment. Row n drives the robot and video
frame n is shown beside it -- exactly the pairing a policy is trained on.

  PY=/home/seene/miniconda3/envs/isaaclab/bin/python
  RT=<retimed dataset dir>   ORIG=openpi_sculptor_dataset/lerobot_20260918_fixed640_guard_12eps
  $PY isaac_show_lerobot.py --dataset $RT --ep 0                      # retimed only, live window
  $PY isaac_show_lerobot.py --dataset $RT --ref $ORIG --ep 0 --sync source --start 30 --end 60
  $PY isaac_show_lerobot.py --dataset $RT --ref $ORIG --ep 0 --start 30 --end 40 \
       --headless --record plots_isaac/ep0_sync.mp4                  # side-by-side mp4

Clock (--sync):
  own    every track plays on its own timeline (retimed finishes earlier -> shows the speedup)
  source tracks are placed by source time (frame_map.parquet of a retimed dataset), so the
         original and retimed robots should overlap pose-for-pose, and so should their images
--start / --end are on the playback clock. Only the [start, end] video frames are decoded
(about 200 KB per frame per camera), so keep windows to a minute or so.
Base: base_vx / base_vy / base_wz are integrated into a base pose (x, y, theta) and drive the
URDF base_x / base_y / base_theta joints, so the EE moves as in the world (--no-base: arm only).
The camera follows the bases (--no-follow to keep it fixed). --no-cams: robots only, no images.
Suction / effort are ignored.
"""
import argparse
import glob
import os
import shutil
import subprocess
import tempfile
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import json

HERE = os.path.dirname(os.path.abspath(__file__))
ROBOT_USD = ("/home/seene/Tele_Isaac/sculptor_isaacsim_teleop-1010/robots/sculptor_10_12_hm_pitch/urdf/"
             "SE01-A0-000 Sculptor EVT_URDF_1010(simplified).SLDASM_cam30/"
             "SE01-A0-000 Sculptor EVT_URDF_1010(simplified).SLDASM_cam30.usd")
ARM = [f"elfin_joint{i}" for i in range(1, 7)]
SPACING_Y = 2.6

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True, help="LeRobot dataset dir (e.g. the retimed one)")
ap.add_argument("--ref", help="second LeRobot dataset shown alongside (e.g. the original)")
ap.add_argument("--ep", type=int, required=True, help="episode_index")
ap.add_argument("--cams", nargs="*", default=["cam0_focus", "cam9"], help="video keys or suffixes")
ap.add_argument("--no-cams", action="store_true", help="robots only: no video frames decoded or shown")
ap.add_argument("--no-base", action="store_true", help="keep the base fixed (arm joints only, old behaviour)")
ap.add_argument("--no-follow", action="store_true", help="do not move the camera with the robot base")
ap.add_argument("--sync", choices=["own", "source"], default="own")
ap.add_argument("--start", type=float, default=0.0)
ap.add_argument("--end", type=float, default=None, help="default: start + 60 s, clipped to the episode")
ap.add_argument("--speed", type=float, default=1.0)
ap.add_argument("--loop", action="store_true")
ap.add_argument("--headless", action="store_true")
ap.add_argument("--record", help="write a side-by-side mp4 (viewport | video frames) and exit")
ap.add_argument("--record-fps", type=float, default=10.0, help="playback-clock samples per second in --record")
ap.add_argument("--eye", type=float, nargs=3, default=None)
ap.add_argument("--target", type=float, nargs=3, default=None)
ap.add_argument("--light", type=float, default=1.0)
args = ap.parse_args()
if args.no_cams:
    args.cams = []
BASE = ["base_x_joint", "base_y_joint", "base_theta_joint"]


def load_track(ds, label):
    ds = os.path.abspath(ds.rstrip("/"))
    info = json.load(open(os.path.join(ds, "meta", "info.json")))
    fps = info["fps"]
    eps = pa.concat_tables([pq.read_table(f) for f in sorted(glob.glob(os.path.join(ds, "meta", "episodes", "*", "*.parquet")))]).to_pandas()
    e = eps[eps.episode_index == args.ep]
    assert len(e) == 1, f"episode {args.ep} not in {ds}"
    e = e.iloc[0]
    tab = pa.concat_tables([pq.read_table(f, columns=["observation.state", "episode_index", "frame_index"],
                                          filters=[("episode_index", "=", args.ep)])
                            for f in sorted(glob.glob(os.path.join(ds, "data", "*", "*.parquet")))]).to_pandas()
    tab = tab.sort_values("frame_index")
    st = np.stack(tab["observation.state"].to_numpy()).astype(np.float64)
    q = st[:, :6]
    own_t = np.arange(len(q)) / fps
    # base pose from base_vx, base_vy (body frame, m/s) and base_wz (rad/s), integrated over this
    # dataset's own rows (retimed velocities are already scaled by the speedup -> same distance)
    dt = 1.0 / fps
    vx, vy, wz = st[:, 7], st[:, 8], st[:, 9]
    th = np.r_[0.0, np.cumsum(wz[:-1] * dt)]
    bx = np.r_[0.0, np.cumsum(((vx * np.cos(th) - vy * np.sin(th)) * dt)[:-1])]
    by = np.r_[0.0, np.cumsum(((vx * np.sin(th) + vy * np.cos(th)) * dt)[:-1])]
    base = np.stack([bx, by, th], 1)
    fm_path = os.path.join(ds, "frame_map.parquet")
    if os.path.exists(fm_path):
        fm = pq.read_table(fm_path, filters=[("episode_index", "=", args.ep)]).to_pandas().sort_values("frame_index")
        src_t, orig_f = fm.source_time_s.to_numpy(), fm.original_frame_index.to_numpy()
    else:
        src_t, orig_f = own_t, np.arange(len(q))
    vkeys = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    videos = {}
    for c in args.cams:
        vk = next((k for k in vkeys if k == c or k.endswith("." + c)), None)
        if vk is None:
            raise SystemExit(f"camera {c} not in {ds}: {vkeys}")
        rel = info["video_path"].format(video_key=vk, chunk_index=int(e[f"videos/{vk}/chunk_index"]),
                                        file_index=int(e[f"videos/{vk}/file_index"]))
        videos[c] = (os.path.join(ds, rel), info["features"][vk]["shape"][:2])
    play_t = src_t if args.sync == "source" else own_t
    return dict(label=label, name=os.path.basename(ds), q=q, base=base, play_t=play_t, src_t=src_t,
                orig_f=orig_f, videos=videos, fps=fps)


def row_at(tr, t):
    """Zero-order hold: the dataset row shown at playback time t."""
    return int(np.clip(np.searchsorted(tr["play_t"], t, side="right") - 1, 0, len(tr["play_t"]) - 1))


def decode_rows(path, hw, r0, r1):
    """RGBA uint8 frames r0..r1 (inclusive) of a video."""
    h, w = hw
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", f"select='between(n\\,{r0}\\,{r1})'",
                          "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgba", "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 4)


tracks = [load_track(args.dataset, "dataset")]
if args.ref:
    tracks.append(load_track(args.ref, "ref"))
t_max = max(tr["play_t"][-1] for tr in tracks)
args.end = min(args.end if args.end is not None else args.start + 60.0, t_max)
for tr in tracks:
    r0, r1 = row_at(tr, args.start), row_at(tr, args.end)
    tr["r0"] = r0
    tr["imgs"] = {c: decode_rows(p, hw, r0, r1) for c, (p, hw) in tr["videos"].items()}
    print(f"[load] {tr['label']} {tr['name']} ep{args.ep}: {len(tr['q'])} rows, {tr['play_t'][-1]:.1f}s "
          f"({args.sync} clock), rows {r0}-{r1} decoded for {list(tr['imgs'])}", flush=True)

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": args.headless, "width": 1600, "height": 900})

import omni.ui as ui  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import SingleArticulation  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics  # noqa: E402

world = World(stage_units_in_meters=1.0, physics_dt=1 / 60, rendering_dt=1 / 60)
world.get_physics_context().set_gravity(0.0)
world.scene.add_default_ground_plane()
stage = omni.usd.get_context().get_stage()
dome = UsdLux.DomeLight.Define(stage, "/World/show_dome_light")
dome.CreateIntensityAttr(1500.0 * args.light)
sun = UsdLux.DistantLight.Define(stage, "/World/show_sun_light")
sun.CreateIntensityAttr(3000.0 * args.light)
sun.CreateAngleAttr(2.0)
UsdGeom.XformCommonAPI(sun).SetRotate(Gf.Vec3f(-50.0, 20.0, -30.0))

robots = []
for k, tr in enumerate(tracks):
    y = (k - (len(tracks) - 1) / 2) * SPACING_Y
    slot = UsdGeom.Xform.Define(stage, f"/World/slot_{k}")
    UsdGeom.XformCommonAPI(slot).SetTranslate(Gf.Vec3d(0.0, y, 0.0))
    path = f"/World/slot_{k}/robot"
    add_reference_to_stage(ROBOT_USD, path)
    prims = [p for p in stage.Traverse() if str(p.GetPath()).startswith(path + "/")]
    for p in prims:  # kinematic display only
        if p.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Set(False)
    root = next(p for p in prims if p.HasAPI(UsdPhysics.ArticulationRootAPI))
    root_tf = UsdGeom.Xformable(root).ComputeLocalToWorldTransform(0).RemoveScaleShear()
    fix = UsdPhysics.FixedJoint.Define(stage, Sdf.Path(f"/World/slot_{k}_world_fix"))
    fix.CreateBody1Rel().SetTargets([root.GetPath()])
    fix.CreateLocalPos0Attr().Set(Gf.Vec3f(root_tf.ExtractTranslation()))
    fix.CreateLocalRot0Attr().Set(Gf.Quatf(root_tf.ExtractRotationQuat()))
    robots.append(world.scene.add(SingleArticulation(prim_path=str(root.GetPath()), name=f"robot_{k}")))

world.reset()
idx, bidx = [], []
for art in robots:
    names = list(art.dof_names)
    idx.append(np.array([names.index(n) for n in ARM]))
    bidx.append(np.array([names.index(n) for n in BASE]) if all(n in names for n in BASE) else None)
if not args.no_base and any(b is None for b in bidx):
    print(f"[warn] base joints {BASE} not found in the articulation; base held fixed", flush=True)
    art.set_joint_positions(np.zeros(art.num_dof))
    art.set_joint_velocities(np.zeros(art.num_dof))

eye = np.array(args.eye or ([5.4, -3.2, 3.0] if len(robots) > 1 else [3.4, -2.8, 2.2]))
target = np.array(args.target or ([0.5, 0.0, 0.5] if len(robots) > 1 else [0.6, 0.0, 0.6]))
set_camera_view(eye=eye.tolist(), target=target.tolist())

# HUD: per track a text line + its video frames (the frame of the same dataset row)
win = ui.Window(f"ep{args.ep} joints vs images", width=max(560, 300 + 270 * len(args.cams)),
                height=120 + (300 if args.cams else 60) * len(tracks))
time_lbl, labels, providers = None, [], []
with win.frame:
    with ui.VStack(spacing=4):
        time_lbl = ui.Label("", style={"font_size": 20}, height=24)
        if len(tracks) > 1:
            ui.Label("robot left = dataset, right = ref", style={"font_size": 16}, height=20)
        for tr in tracks:
            labels.append(ui.Label("", style={"font_size": 16}, height=40, word_wrap=True))
            row = {}
            if not args.cams:
                providers.append(row)
                continue
            with ui.HStack(height=260, spacing=6):
                for c in args.cams:
                    with ui.VStack(width=256):
                        ui.Label(c, height=16)
                        prov = ui.ByteImageProvider()
                        ui.ImageWithProvider(prov, width=256, height=256)
                        row[c] = prov
            providers.append(row)


def set_pose(t):
    time_lbl.text = f"ep{args.ep}  t = {t:7.2f} s  ({args.start:.1f}-{args.end:.1f}, {args.sync} clock)"
    info_rows = []
    centre = []
    for k, (tr, art) in enumerate(zip(tracks, robots)):
        r = row_at(tr, t)
        full = np.zeros(art.num_dof)
        full[idx[k]] = tr["q"][r]
        if not args.no_base and bidx[k] is not None:
            full[bidx[k]] = tr["base"][r]
            centre.append(tr["base"][r, :2])
        art.set_joint_positions(full)
        art.set_joint_velocities(np.zeros(art.num_dof))
        art.apply_action(ArticulationAction(joint_positions=full))
        bx, by, bth = tr["base"][r]
        txt = (f"{tr['label']}: {tr['name']}\n row {r}  src t {tr['src_t'][r]:.3f}s  "
               f"image = orig frame {tr['orig_f'][r]} ({tr['orig_f'][r] / tr['fps']:.3f}s)  "
               f"base ({bx:+.2f}, {by:+.2f}) m {np.rad2deg(bth):+.0f} deg")
        labels[k].text = txt
        for c, prov in providers[k].items():
            im = tr["imgs"][c][min(max(r - tr["r0"], 0), len(tr["imgs"][c]) - 1)]
            prov.set_data_array(np.ascontiguousarray(im), [im.shape[1], im.shape[0]])
        info_rows.append((tr, r))
    if centre and not args.no_follow:
        c = np.mean(centre, 0)
        off = np.array([c[0], c[1], 0.0])
        set_camera_view(eye=(eye + off).tolist(), target=(target + off).tolist())
    return info_rows


for _ in range(30):
    world.step(render=True)

if args.record:
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
    from PIL import Image, ImageDraw

    tmp = tempfile.mkdtemp(prefix="isaac_rec_")
    times = np.arange(args.start, args.end + 1e-9, 1.0 / args.record_fps)
    composed = []
    for i, t in enumerate(times):
        for _ in range(3):
            rows = set_pose(t)
            world.step(render=True)
        png = os.path.join(tmp, f"vp_{i:05d}.png")
        capture_viewport_to_file(get_active_viewport(), png)
        for _ in range(4):
            app.update()
        composed.append((png, t, rows))
        if (i + 1) % 50 == 0:
            print(f"[record] {i + 1}/{len(times)}", flush=True)
    for _ in range(10):
        app.update()
    # compose: viewport (scaled to 720 px high) | per track a column of camera frames
    out_path = os.path.abspath(args.record)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    H = 256 * len(args.cams) + 60 if args.cams else 720
    frames = []
    for png, t, rows in composed:
        if not os.path.exists(png):
            continue
        vp = Image.open(png).convert("RGB")
        vp = vp.resize((int(vp.width * H / vp.height), H))
        canvas = Image.new("RGB", (vp.width + 260 * len(rows), H), (20, 20, 20))
        canvas.paste(vp, (0, 0))
        d = ImageDraw.Draw(canvas)
        d.text((10, 10), f"ep{args.ep}  t={t:.2f}s ({args.sync})", fill=(255, 255, 0))
        for k, (tr, r) in enumerate(rows):
            x = vp.width + 260 * k
            d.text((x + 4, 4), f"{tr['label']} row {r} src {tr['src_t'][r]:.2f}s", fill=(255, 255, 255))
            d.text((x + 4, 20), f"img = orig f{tr['orig_f'][r]}", fill=(200, 200, 200))
            bx, by, bth = tr["base"][r]
            d.text((x + 4, 36 if not args.cams else 40 + 256 * len(args.cams) + 4),
                   f"base ({bx:+.2f},{by:+.2f}) m", fill=(200, 200, 200))
            for j, c in enumerate(args.cams):
                im = tr["imgs"][c][min(max(r - tr["r0"], 0), len(tr["imgs"][c]) - 1)]
                canvas.paste(Image.fromarray(im[..., :3]), (x + 2, 40 + 256 * j))
        frames.append(canvas)
    if frames:
        w, h = frames[0].size
        w, h = w - w % 2, h - h % 2
        enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                "-s", f"{w}x{h}", "-r", str(args.record_fps), "-i", "-", "-c:v", "libx264",
                                "-pix_fmt", "yuv420p", "-crf", "20", out_path], stdin=subprocess.PIPE)
        for f in frames:
            enc.stdin.write(np.asarray(f.crop((0, 0, w, h))).tobytes())
        enc.stdin.close()
        enc.wait()
        print(f"[record] {len(frames)} frames -> {out_path}", flush=True)
    else:
        print("[record] no viewport captures were written", flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
    app.close()
    raise SystemExit(0)

wall0 = time.monotonic()
while app.is_running():
    t = args.start + (time.monotonic() - wall0) * args.speed
    if t > args.end:
        if args.loop:
            wall0, t = time.monotonic(), args.start
        else:
            t = args.end
    set_pose(t)
    world.step(render=True)
    if args.headless and t >= args.end:
        break
app.close()
