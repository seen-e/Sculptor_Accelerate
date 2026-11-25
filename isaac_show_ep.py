#!/usr/bin/env python3
"""Replay EVT episode joint states (before / after retiming) in Isaac Sim, kinematic only.

  PY=/home/seene/miniconda3/envs/isaaclab/bin/python
  $PY isaac_show_ep.py --ep 35                       # left: 01 原始 state, right: 02 修改后候选
  $PY isaac_show_ep.py --ep 358 --only after --speed 0.5
  $PY isaac_show_ep.py --ep 99 --sync source --start 20 --end 60
  $PY isaac_show_ep.py --ep 35 --after-dir modified_v1   # right: retime_ep.py output
  options: --only before|after  --loop  --headless  --snapshots T ...

Clock (--sync):
  own    each robot plays on its own time axis (after finishes earlier -> shows the speedup)
  source both follow the original path time; "after" is placed via its source_time_s,
         so the two should overlap pose-for-pose (checks the path is preserved)
Only joints 0-5 (elfin_joint1..6) are used. Base joints held at 0, suction / effort ignored.
"""
import argparse
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "EVT五条轨迹_state修改前后_20261106")
ROBOT_USD = ("/home/seene/Tele_Isaac/sculptor_isaacsim_teleop-1010/robots/sculptor_10_12_hm_pitch/urdf/"
             "SE01-A0-000 Sculptor EVT_URDF_1010(simplified).SLDASM_cam30/"
             "SE01-A0-000 Sculptor EVT_URDF_1010(simplified).SLDASM_cam30.usd")
ARM = [f"elfin_joint{i}" for i in range(1, 7)]
J5_SINGULAR = 0.15
SPACING_Y = 2.6

ap = argparse.ArgumentParser()
ap.add_argument("--ep", type=int, required=True, help="35 / 99 / 187 / 358 / 418")
ap.add_argument("--only", choices=["before", "after"])
ap.add_argument("--sync", choices=["own", "source"], default="own")
ap.add_argument("--after-dir", help="show retime_ep.py output (e.g. modified_v1) instead of 02 as 'after'")
ap.add_argument("--start", type=float, default=0.0)
ap.add_argument("--end", type=float, default=None, help="default: end of the longest track")
ap.add_argument("--light", type=float, default=1.0, help="lighting intensity multiplier")
ap.add_argument("--speed", type=float, default=1.0)
ap.add_argument("--loop", action="store_true")
ap.add_argument("--headless", action="store_true")
ap.add_argument("--snapshots", nargs="*", type=float, default=[],
                help="playback times to save viewport PNGs at (into plots_isaac/), then exit")
args = ap.parse_args()


def load_tracks():
    """Each track: (label, playback t, joint q[N,6], source t for HUD)."""
    ep = f"EP{args.ep:03d}"
    tracks = []
    if args.only != "after":
        b = np.load(os.path.join(PKG, "01_修改前_原始state30", ep, "state30.npz"))
        names = list(b["state_names"])
        q = b["state"][:, [names.index(n) for n in ARM]].astype(np.float64)
        tracks.append(("before", b["time_s"], q, b["time_s"]))
    if args.only != "before" and args.after_dir:  # retime_ep.py output
        a = np.load(os.path.join(HERE, args.after_dir, ep, "state30.npz"))
        t = a["source_time_s"] if args.sync == "source" else a["time_s"]
        tracks.append((args.after_dir, t, a["state"][:, :6].astype(np.float64), a["source_time_s"]))
    elif args.only != "before":
        a = np.load(os.path.join(PKG, "02_修改后_离线候选state", ep, "state30_partial.npz"))
        t = a["source_time_s"] if args.sync == "source" else a["candidate_time_s"]
        tracks.append(("after(02)", t, a["joint_position_rad"], a["source_time_s"]))
    return tracks


def sample(t, ts, vals):
    """Zero-order hold, same as isaac_show.py."""
    i = np.clip(np.searchsorted(ts, t, side="right") - 1, 0, len(ts) - 1)
    return vals[i]


tracks = load_tracks()
for label, ts, qs, _ in tracks:
    print(f"[load] EP{args.ep:03d} {label}: {len(ts)} samples, {ts[-1]:.1f} s, "
          f"q range {np.round(qs.min(0), 2)} .. {np.round(qs.max(0), 2)}", flush=True)
if args.end is None:
    args.end = float(max(ts[-1] for _, ts, _, _ in tracks))

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
for k, (label, _, _, _) in enumerate(tracks):
    y = (k - (len(tracks) - 1) / 2) * SPACING_Y
    slot = UsdGeom.Xform.Define(stage, f"/World/slot_{k}")  # fresh xform carries the offset
    UsdGeom.XformCommonAPI(slot).SetTranslate(Gf.Vec3d(0.0, y, 0.0))
    path = f"/World/slot_{k}/robot"
    add_reference_to_stage(ROBOT_USD, path)
    prims = [p for p in stage.Traverse() if str(p.GetPath()).startswith(path + "/")]
    # kinematic display only: joints are teleported every frame, so collisions just blow up
    for p in prims:
        if p.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Set(False)
    # pin the articulation root to the world at its current pose
    root = next(p for p in prims if p.HasAPI(UsdPhysics.ArticulationRootAPI))
    root_tf = UsdGeom.Xformable(root).ComputeLocalToWorldTransform(0).RemoveScaleShear()
    fix = UsdPhysics.FixedJoint.Define(stage, Sdf.Path(f"/World/slot_{k}_world_fix"))
    fix.CreateBody1Rel().SetTargets([root.GetPath()])
    fix.CreateLocalPos0Attr().Set(Gf.Vec3f(root_tf.ExtractTranslation()))
    fix.CreateLocalRot0Attr().Set(Gf.Quatf(root_tf.ExtractRotationQuat()))
    art = world.scene.add(SingleArticulation(prim_path=str(root.GetPath()), name=f"robot_{k}"))
    robots.append((label, art, y))

world.reset()
idx = []
for _, art, _ in robots:
    names = list(art.dof_names)
    idx.append(np.array([names.index(n) for n in ARM]))
    zero = np.zeros(art.num_dof)
    art.set_joint_positions(zero)
    art.set_joint_velocities(zero)

if len(robots) > 1:
    set_camera_view(eye=[4.2, -1.8, 2.3], target=[0.5, 0.0, 0.5])
else:
    set_camera_view(eye=[2.8, -2.2, 1.8], target=[0.6, 0.0, 0.6])

# HUD
win = ui.Window(f"EP{args.ep:03d} state playback", width=620, height=130 + 70 * len(robots))
labels = []
with win.frame:
    with ui.VStack(spacing=4):
        time_lbl = ui.Label("", style={"font_size": 22})
        if len(robots) > 1:
            ui.Label("left: before (original)    right: after (retimed)", style={"font_size": 18})
        for _ in robots:
            labels.append(ui.Label("", style={"font_size": 20}))


def set_pose(t):
    time_lbl.text = (f"EP{args.ep:03d}  t = {t:7.2f} s   ({args.start:.1f} - {args.end:.1f}, "
                     f"x{args.speed:g}, sync={args.sync})")
    for k, ((label, ts, qs, src), (_, art, _)) in enumerate(zip(tracks, robots)):
        q = sample(t, ts, qs)
        full = np.zeros(art.num_dof)
        full[idx[k]] = q
        art.set_joint_positions(full)
        art.set_joint_velocities(np.zeros(art.num_dof))
        art.apply_action(ArticulationAction(joint_positions=full))  # keep drives from pulling back
        done = "  (done)" if t > ts[-1] else ""
        warn = "  ! wrist singular |J5|<%.2f" % J5_SINGULAR if abs(q[4]) < J5_SINGULAR else ""
        labels[k].text = (f"{label}  src t={sample(t, ts, src):6.2f}s  "
                          f"J4={q[3]:+.2f} J5={q[4]:+.2f} J6={q[5]:+.2f}{warn}{done}")
        labels[k].style = {"font_size": 20, "color": 0xFF3B3BD0 if warn else 0xFFDDDDDD}


for _ in range(30):  # let meshes and materials load before playback
    world.step(render=True)

if args.snapshots:
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

    snap_dir = os.path.join(HERE, "plots_isaac")
    os.makedirs(snap_dir, exist_ok=True)

wall0 = time.monotonic()
snap_i = 0
while app.is_running():
    if args.snapshots:  # deterministic stepping through the requested times
        if snap_i >= len(args.snapshots):
            break
        t = args.snapshots[snap_i]
    else:
        t = args.start + (time.monotonic() - wall0) * args.speed
    if t > args.end:
        if not args.loop:
            t = args.end
        else:
            wall0 = time.monotonic()
            t = args.start
    set_pose(t)
    world.step(render=True)
    if args.snapshots:
        for _ in range(10):  # settle, then capture
            set_pose(t)
            world.step(render=True)
        out = os.path.join(snap_dir, f"EP{args.ep:03d}_{args.only or 'both'}_{args.sync}_t{t:07.2f}.png")
        capture_viewport_to_file(get_active_viewport(), out)
        for _ in range(5):
            app.update()
        for k, (label, art, _) in enumerate(robots):
            got = art.get_joint_positions()[idx[k]]
            want = sample(t, tracks[k][1], tracks[k][2])
            print(f"[snap] t={t:.2f} {label} max|q_sim-q_track|={np.max(np.abs(got - want)):.4f} -> {out}", flush=True)
        snap_i += 1
        continue
    if args.headless and t >= args.end:
        for k, (label, art, _) in enumerate(robots):
            got = art.get_joint_positions()[idx[k]]
            want = sample(t, tracks[k][1], tracks[k][2])
            print(f"[check] {label} max |q_sim - q_track| = {np.max(np.abs(got - want)):.4f} rad")
        break

app.close()
