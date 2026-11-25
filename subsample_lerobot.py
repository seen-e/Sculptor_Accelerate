#!/usr/bin/env python3
"""Speed up a LeRobot v3.0 dataset (openpi_sculptor_dataset) by dropping frames. Every output
row and every output image is one original frame (no interpolation), so state / action /
images stay exactly aligned; output fps stays 30 Hz, so dropped frames = faster motion.

  PY=/home/seene/miniconda3/envs/isaaclab/bin/python
  $PY subsample_lerobot.py --src openpi_sculptor_dataset/lerobot_20260918_fixed640_guard_12eps
  $PY subsample_lerobot.py --src openpi_sculptor_dataset/lerobot_*          # all datasets
  $PY subsample_lerobot.py --src ... --mode uniform                          # every 3rd frame, no limits
  $PY subsample_lerobot.py --src ... --mode realtime                         # 30 -> 10 Hz, real speed

Modes
  variable (default): each output frame advances 1..--stride original frames.
    * protected windows (step 1 = original speed): place = suction_do_ch3 1->0->1 pulse,
      pick = fluid_pressure drops below --pick-pressure; widened by --guard-pre/--guard-post
      (same detection as retime_ep.py).
    * elsewhere the step rate r <= min(stride, margin*V/|v|, sqrt(margin*A/|a|)), v / a from a
      smoothing spline of the original joint path. Steps are integers held over runs (no 2,3,2,3
      jitter): a step grows by 1 only after --dwell output frames, and shrinks by <= 1 per output
      frame (backward pass so it is back to 1 when a window starts).
    * hard check on every step: |q[f+k] - q[f]| * fps <= margin * V, otherwise k is reduced.
  uniform : every --stride-th frame, fps unchanged (stride x faster everywhere, ignores limits).
  realtime: every --stride-th frame, fps / stride (plain downsampling, real speed).

Velocity-like fields are scaled by the local rate r_n = (f_{n+1} - f_{n-1}) / 2 (rate 1 in
realtime mode): state joint velocities + base_vx/vy/wz, action base_vx/vy/wz.
Action (--action next, default): original action[i] ~= joint state[i + 1], so row n gets the
whole original action row at f_{n+1} - 1 (~= state of the next output frame).
--action same keeps action[f_n].

Outputs (per dataset): a full LeRobot v3.0 dataset, frame_map.parquet (episode_index,
frame_index, original_frame_index, rate, protected), subsample_report.json.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.interpolate import make_smoothing_spline

POS = slice(0, 6)
SUCTION, PRESSURE = 6, 10
STATE_VEL = list(range(7, 10)) + list(range(14, 20))  # base_vx/vy/wz, joint1..6 velocity
ACTION_VEL = [7, 8, 9]  # base_vx/vy/wz action
QUANTILES = [0.01, 0.10, 0.50, 0.90, 0.99]

ap = argparse.ArgumentParser()
ap.add_argument("--src", nargs="+", required=True, help="LeRobot dataset dir(s)")
ap.add_argument("--out-root", default=None,
                help="default: <parent of src>_{mode}_s{stride}/ next to the source folder")
ap.add_argument("--mode", choices=["variable", "uniform", "realtime"], default="variable")
ap.add_argument("--stride", type=int, default=3, help="max (variable) / fixed step in original frames")
ap.add_argument("--vmax-deg", type=float, nargs=6, default=[63, 36, 90, 108, 162, 162])
ap.add_argument("--amax-deg", type=float, nargs=6, default=[120, 60, 90, 96, 192, 192])
ap.add_argument("--margin", type=float, default=0.98, help="plan to margin * limits")
ap.add_argument("--no-acc-cap", action="store_true", help="variable: only velocity caps the rate")
ap.add_argument("--dwell", type=int, default=10,
                help="variable: output frames a step must be held before it may grow by 1")
ap.add_argument("--guard-pre", type=float, default=0.8, help="s before a suction event")
ap.add_argument("--guard-post", type=float, default=0.8, help="s after a suction event")
ap.add_argument("--pick-pressure", type=float, default=-30.0, help="pick = pressure below this")
ap.add_argument("--smooth-lam", type=float, default=1e-4, help="smoothing spline lambda (0 = gradient)")
ap.add_argument("--action", choices=["next", "same"], default="next")
ap.add_argument("--encoder", choices=["nvenc", "x264"], default="nvenc")
ap.add_argument("--qp", type=int, default=18)
ap.add_argument("--jobs", type=int, default=6, help="parallel video re-encodes")
ap.add_argument("--verify", type=int, default=2, help="episodes per dataset to pixel-check (0 = off)")
ap.add_argument("--overwrite", action="store_true")
args = ap.parse_args()

V = np.deg2rad(args.vmax_deg)
A = np.deg2rad(args.amax_deg)


# ---------------------------------------------------------------- frame planning
def smooth_va(q, fps):
    t = np.arange(len(q)) / fps
    if args.smooth_lam <= 0 or len(q) < 8:
        v = np.gradient(q, t, axis=0) if len(q) > 1 else np.zeros_like(q)
        return v, (np.gradient(v, t, axis=0) if len(q) > 1 else np.zeros_like(q))
    sp = [make_smoothing_spline(t, q[:, j], lam=args.smooth_lam) for j in range(q.shape[1])]
    return (np.stack([p.derivative(1)(t) for p in sp], 1),
            np.stack([p.derivative(2)(t) for p in sp], 1))


def protected_mask(state, fps):
    L = len(state)
    suc, p = state[:, SUCTION] > 0.5, state[:, PRESSURE]
    events = []
    falls = np.where(suc[:-1] & ~suc[1:])[0] + 1
    rises = np.where(~suc[:-1] & suc[1:])[0] + 1
    for f in falls:
        r = rises[rises > f]
        events.append(("place", int(f), int(r[0]) if len(r) else L - 1))
    hold = p < args.pick_pressure
    for i in np.where(~hold[:-1] & hold[1:])[0] + 1:
        events.append(("pick", int(i), int(i)))
    m = np.zeros(L, bool)
    pre, post = int(round(args.guard_pre * fps)), int(round(args.guard_post * fps))
    for _, a, b in events:
        m[max(a - pre, 0):min(b + post, L - 1) + 1] = True
    return m, events


def plan_variable(state, fps):
    L, S = len(state), args.stride
    q = state[:, POS].astype(np.float64)
    mask, events = protected_mask(state, fps)
    v, a = smooth_va(q, fps)
    with np.errstate(divide="ignore", invalid="ignore"):
        cap = np.minimum(float(S), np.min(args.margin * V / np.abs(v), 1))
        if not args.no_acc_cap:
            cap = np.minimum(cap, np.min(np.sqrt(args.margin * A / np.abs(a)), 1))
    kcap = np.clip(np.floor(np.nan_to_num(cap, nan=float(S)) + 1e-9), 1, S).astype(int)
    kcap[mask] = 1
    for f in range(L - 2, -1, -1):  # step down by <= 1 per output frame before slow parts / windows
        kcap[f] = min(kcap[f], kcap[min(f + kcap[f], L - 1)] + 1)
    starts = np.where(mask & ~np.r_[False, mask[:-1]])[0]
    vlim = args.margin * V / fps
    frames, f, k_prev, hold = [0], 0, 1, 0
    while f < L - 1:
        kc = kcap[f]
        if kc > k_prev:  # step up by 1 only after holding the current step --dwell frames
            k = k_prev + 1 if hold >= args.dwell else k_prev
        else:
            k = kc
        nxt = starts[starts > f]
        if len(nxt):
            k = min(k, int(nxt[0]) - f)  # land exactly on a window start
        k = min(k, L - 1 - f)
        while k > 1 and np.any(np.abs(q[f + k] - q[f]) > vlim):
            k -= 1
        hold = hold + 1 if k == k_prev else 0
        k_prev = k
        f += k
        frames.append(f)
    frames = np.asarray(frames)
    return frames, mask[frames], events


def local_rate(frames, L):
    if len(frames) == 1:
        return np.ones(1)
    fe = np.r_[frames[0] - (frames[1] - frames[0]), frames, frames[-1] + (frames[-1] - frames[-2])]
    return (fe[2:] - fe[:-2]) / 2.0


def plan_episode(state, src_fps):
    L, S = len(state), args.stride
    if args.mode == "variable":
        frames, prot, events = plan_variable(state, src_fps)
        rate = local_rate(frames, L)
    else:
        frames = np.arange(0, L, S)
        prot, events = np.zeros(len(frames), bool), []
        rate = np.full(len(frames), float(S) if args.mode == "uniform" else 1.0)
    if args.action == "next":
        a_src = np.r_[frames[1:] - 1, min(frames[-1] + max(int(round(local_rate(frames, L)[-1])), 1) - 1, L - 1)]
    else:
        a_src = frames
    return frames, a_src, rate, prot, events


# ---------------------------------------------------------------- helpers
def feature_stats(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    q = np.quantile(x, QUANTILES, axis=0)
    d = dict(min=x.min(0), max=x.max(0), mean=x.mean(0), std=x.std(0), count=np.array([len(x)]))
    for name, v in zip(["q01", "q10", "q50", "q90", "q99"], q):
        d[name] = v
    return d


def to_py(v):
    """numpy scalars / (nested object) arrays -> plain python, for pa.array."""
    if isinstance(v, np.ndarray):
        return to_py(v.item()) if v.ndim == 0 else [to_py(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    return v


def video_select(src, dst, frames, fps, L, w, h):
    """Decode src (yuv420p raw, no colour conversion), keep the listed frame numbers, encode."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if args.encoder == "nvenc":
        enc = ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "constqp", "-qp", str(args.qp)]
    else:
        enc = ["-c:v", "libx264", "-preset", "medium", "-qp", str(args.qp)]
    fsz = w * h * 3 // 2
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", src, "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"],
                           stdout=subprocess.PIPE, bufsize=fsz * 8)
    encp = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p",
                             "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", *enc, "-g", "2", "-bf", "0",
                             "-pix_fmt", "yuv420p", "-an", dst], stdin=subprocess.PIPE)
    i = ptr = 0
    while True:
        buf = dec.stdout.read(fsz)
        if len(buf) < fsz:
            break
        if ptr < len(frames) and i == frames[ptr]:
            encp.stdin.write(buf)
            ptr += 1
        i += 1
    dec.wait()
    encp.stdin.close()
    encp.wait()
    if dec.returncode or encp.returncode:
        raise RuntimeError(f"ffmpeg failed for {src}")
    if i != L or ptr != len(frames):
        raise RuntimeError(f"{src}: decoded {i} frames (parquet {L}), wrote {ptr}/{len(frames)}")
    n = int(subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0",
                            "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", dst],
                           check=True, capture_output=True, text=True).stdout.strip())
    if n != len(frames):
        raise RuntimeError(f"{dst}: {n} frames, expected {len(frames)}")


def decode(path, frames, w, h):
    sel = "+".join(f"eq(n\\,{f})" for f in frames)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", f"select='{sel}'",
                          "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w).astype(np.float64)


# ---------------------------------------------------------------- main
def process(src):
    src = os.path.abspath(src.rstrip("/"))
    name = os.path.basename(src)
    tag = f"{args.mode}_s{args.stride}"
    dst = os.path.join(args.out_root or (os.path.dirname(src) + f"_{tag}"), f"{name}_{tag}")
    if os.path.exists(dst):
        if not args.overwrite:
            print(f"[skip] {dst} exists (use --overwrite)")
            return
        shutil.rmtree(dst)
    t_start = time.time()
    info = json.load(open(os.path.join(src, "meta", "info.json")))
    assert info["codebase_version"].startswith("v3"), info["codebase_version"]
    src_fps = info["fps"]
    fps = src_fps / args.stride if args.mode == "realtime" else src_fps
    if abs(fps - round(fps)) > 1e-9:
        raise ValueError(f"fps {src_fps}/{args.stride} is not an integer")
    fps = int(round(fps))
    vkeys = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    h, w = info["features"][vkeys[0]]["shape"][:2]

    ep_files = sorted(glob.glob(os.path.join(src, "meta", "episodes", "*", "*.parquet")))
    eps = pa.concat_tables([pq.read_table(f) for f in ep_files]).to_pandas()
    eps = eps.sort_values("episode_index").reset_index(drop=True)
    data = pa.concat_tables([pq.read_table(f) for f in sorted(glob.glob(os.path.join(src, "data", "*", "*.parquet")))])
    ep_col = data["episode_index"].to_numpy()
    fi_col = data["frame_index"].to_numpy()
    state_all = np.stack(data["observation.state"].to_numpy(zero_copy_only=False)).astype(np.float32)
    action_all = np.stack(data["action"].to_numpy(zero_copy_only=False)).astype(np.float32)

    rows, act_rows, rates, prots, fmap, ep_ranges, video_jobs, ep_report = [], [], [], [], [], [], [], []
    g = 0
    for _, e in eps.iterrows():
        ei, L = int(e["episode_index"]), int(e["length"])
        r = np.where(ep_col == ei)[0]
        r = r[np.argsort(fi_col[r])]
        assert len(r) == L and np.array_equal(fi_col[r], np.arange(L)), f"episode {ei} not contiguous"
        frames, a_src, rate, prot, events = plan_episode(state_all[r].astype(np.float64), src_fps)
        n = len(frames)
        rows.append(r[frames]); act_rows.append(r[a_src]); rates.append(rate); prots.append(prot)
        fmap.append(dict(episode_index=np.full(n, ei), frame_index=np.arange(n),
                         original_frame_index=frames, rate=rate, protected=prot))
        ep_ranges.append((ei, g, g + n, n))
        g += n
        steps = np.diff(frames)
        ep_report.append(dict(episode=ei, src_frames=L, frames=n, src_seconds=L / src_fps, seconds=n / fps,
                              speedup=(L - 1) / max(n - 1, 1),
                              events=len(events), protected_frames=int(prot.sum()),
                              step_hist={int(k): int(c) for k, c in zip(*np.unique(steps, return_counts=True))}))
        for vk in vkeys:
            assert abs(float(e[f"videos/{vk}/from_timestamp"])) < 1e-6, "expects one episode per video file"
            rel = info["video_path"].format(video_key=vk, chunk_index=int(e[f"videos/{vk}/chunk_index"]),
                                            file_index=int(e[f"videos/{vk}/file_index"]))
            video_jobs.append(dict(src=os.path.join(src, rel), dst=os.path.join(dst, rel), frames=frames, L=L, ep=ei))
    rows, act_rows = np.concatenate(rows), np.concatenate(act_rows)
    rate, prot = np.concatenate(rates), np.concatenate(prots)
    state = state_all[rows].copy()
    action = action_all[act_rows].copy()
    state[:, STATE_VEL] *= rate[:, None].astype(np.float32)
    action[:, ACTION_VEL] *= rate[:, None].astype(np.float32)
    N = len(rows)
    frame_index = np.concatenate([np.arange(n) for *_, n in ep_ranges]).astype(np.int64)
    timestamp = (frame_index / fps).astype(np.float32)
    episode_index = ep_col[rows].astype(np.int64)
    index = np.arange(N, dtype=np.int64)
    task_index = data["task_index"].to_numpy()[rows].astype(np.int64)

    out = pa.table({
        "observation.state": pa.array(list(state), type=data.schema.field("observation.state").type),
        "action": pa.array(list(action), type=data.schema.field("action").type),
        "timestamp": pa.array(timestamp, pa.float32()),
        "frame_index": frame_index, "episode_index": episode_index,
        "index": index, "task_index": task_index,
    }).select(data.column_names)
    data_rel = info["data_path"].format(chunk_index=0, file_index=0)
    os.makedirs(os.path.dirname(os.path.join(dst, data_rel)), exist_ok=True)
    pq.write_table(out, os.path.join(dst, data_rel))
    pq.write_table(pa.table({k: np.concatenate([m[k] for m in fmap]) for k in fmap[0]}),
                   os.path.join(dst, "frame_map.parquet"))

    with ThreadPoolExecutor(args.jobs) as pool:
        futs = [pool.submit(video_select, j["src"], j["dst"], j["frames"], fps, j["L"], w, h) for j in video_jobs]
        for i, f in enumerate(futs):
            f.result()
            if (i + 1) % 50 == 0 or i + 1 == len(futs):
                print(f"  [{name}] videos {i + 1}/{len(futs)}", flush=True)

    # episodes meta + stats
    cols = {"observation.state": state, "action": action, "timestamp": timestamp,
            "frame_index": frame_index, "episode_index": episode_index, "index": index,
            "task_index": task_index}
    E = {c: [to_py(v) for v in eps[c]] for c in eps.columns}
    for i, (ei, a, b, n) in enumerate(ep_ranges):
        E["length"][i] = n
        E["data/chunk_index"][i] = 0
        E["data/file_index"][i] = 0
        E["dataset_from_index"][i] = a
        E["dataset_to_index"][i] = b
        E["meta/episodes/chunk_index"][i] = 0
        E["meta/episodes/file_index"][i] = 0
        for vk in vkeys:
            E[f"videos/{vk}/to_timestamp"][i] = n / fps
            E[f"stats/{vk}/count"][i] = [n]  # image mean/std kept from source
        for key, arr in cols.items():
            for sname, v in feature_stats(arr[a:b]).items():
                E[f"stats/{key}/{sname}"][i] = to_py(v)
    ep_rel = "meta/episodes/chunk-000/file-000.parquet"
    os.makedirs(os.path.dirname(os.path.join(dst, ep_rel)), exist_ok=True)
    ep_schema = pq.read_schema(ep_files[0])
    pq.write_table(pa.table({f.name: pa.array(E[f.name], type=f.type) for f in ep_schema}),
                   os.path.join(dst, ep_rel))
    stats = json.load(open(os.path.join(src, "meta", "stats.json")))
    for key, arr in cols.items():
        stats[key] = {k: v.tolist() for k, v in feature_stats(arr).items()}
    for vk in vkeys:
        stats[vk]["count"] = [N]
    json.dump(stats, open(os.path.join(dst, "meta", "stats.json"), "w"), indent=4)
    info["fps"] = fps
    info["total_frames"] = N
    for vk in vkeys:
        info["features"][vk]["info"]["video.fps"] = fps
        info["features"][vk]["info"]["video.encoder"] = "nvenc_constqp" if args.encoder == "nvenc" else "libx264_qp"
    json.dump(info, open(os.path.join(dst, "meta", "info.json"), "w"), indent=4, ensure_ascii=False)
    shutil.copy2(os.path.join(src, "meta", "tasks.parquet"), os.path.join(dst, "meta", "tasks.parquet"))
    if os.path.exists(os.path.join(src, "conversion_report.json")):
        shutil.copy2(os.path.join(src, "conversion_report.json"), os.path.join(dst, "conversion_report.json"))

    # motion check on the output (finite differences of the real joint samples, output time)
    fmap_all = np.concatenate([m["original_frame_index"] for m in fmap])
    vu, vu_prot, a99_out, a99_src, n_over = [], [], [], [], 0
    for ei, a, b, n in ep_ranges:
        q = state[a:b, POS].astype(np.float64)
        if n < 3:
            continue
        vd = np.diff(q, axis=0) * fps
        pr = prot[a:b - 1] & prot[a + 1:b]
        util = np.abs(vd) / V
        vu.append(util[~pr].max(0) if (~pr).any() else np.zeros(6))
        vu_prot.append(util[pr].max(0) if pr.any() else np.zeros(6))
        big = np.diff(fmap_all[a:b]) > 1  # steps that skip frames (step-1 rows are original motion)
        n_over += int((util[big & ~pr] > args.margin + 1e-6).any(1).sum())
        a99_out.append(np.percentile(np.abs(np.diff(vd, axis=0) * fps) / A, 99, axis=0))
        qs = state_all[ep_col == ei][:, POS].astype(np.float64)
        a99_src.append(np.percentile(np.abs(np.diff(qs, 2, axis=0) * src_fps ** 2) / A, 99, axis=0))
    motion = dict(peak_velocity_util_retimed=np.max(vu, 0).tolist(),
                  peak_velocity_util_protected=np.max(vu_prot, 0).tolist(),
                  retimed_steps_over_margin=n_over,
                  p99_fd_accel_util_output=np.max(a99_out, 0).tolist(),
                  p99_fd_accel_util_source=np.max(a99_src, 0).tolist())

    checks = []
    for j in video_jobs:
        if len([c for c in checks if c["episode"] == j["ep"]]) or len({c["episode"] for c in checks}) >= args.verify:
            continue
        fr, L = j["frames"], j["L"]
        ks = sorted({0, len(fr) // 3, len(fr) // 2, 2 * len(fr) // 3, len(fr) - 1})
        nv = decode(j["dst"], ks, w, h)
        o = decode(j["src"], [int(fr[k]) for k in ks], w, h)
        o2 = decode(j["src"], [min(int(fr[k]) + 1, L - 1) for k in ks], w, h)
        checks.append(dict(episode=j["ep"], video=os.path.relpath(j["dst"], dst), rows=ks,
                           mae_vs_mapped_orig=float(np.abs(o - nv).mean()),
                           mae_vs_next_orig=float(np.abs(o2 - nv).mean())))

    report = dict(src=src, dst=dst, params=vars(args), src_fps=src_fps, fps=fps,
                  src_frames=int(len(state_all)), frames=int(N),
                  speedup=float(len(state_all) / N) if args.mode != "realtime" else 1.0,
                  protected_fraction=float(prot.mean()), motion=motion,
                  image_alignment_check=checks, episodes=ep_report, seconds=time.time() - t_start)
    json.dump(report, open(os.path.join(dst, "subsample_report.json"), "w"), indent=2, ensure_ascii=False)
    print(f"[{name}] {len(state_all)} -> {N} frames (x{len(state_all) / N:.2f}), {src_fps} -> {fps} Hz, "
          f"{args.mode}, protected {prot.mean():.1%}, {report['seconds']:.0f}s -> {dst}")
    print("  retimed peak |v|/V :", np.round(motion["peak_velocity_util_retimed"], 2),
          " steps over margin:", n_over)
    print("  protected peak |v|/V:", np.round(motion["peak_velocity_util_protected"], 2))
    print("  p99 |a|/A out / src:", np.round(motion["p99_fd_accel_util_output"], 2),
          "/", np.round(motion["p99_fd_accel_util_source"], 2))
    for c in checks:
        print(f"  verify ep{c['episode']} {c['video']}: MAE vs mapped orig {c['mae_vs_mapped_orig']:.2f}, "
              f"vs next orig {c['mae_vs_next_orig']:.2f}")


if __name__ == "__main__":
    for s in args.src:
        process(s)
