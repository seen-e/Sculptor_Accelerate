#!/usr/bin/env python3
"""Measure the image <-> joint-state delay of LeRobot datasets by cross-correlating image motion
with joint motion, per episode and camera. Run it on the original and the retimed dataset and
compare: retiming should not change the delay (beyond the <= 1/2 source-frame image rounding).

  PY=/home/seene/miniconda3/envs/isaaclab/bin/python
  $PY check_sync.py --dataset openpi_sculptor_dataset/lerobot_20260918_fixed640_guard_12eps \
                              <retimed dataset dir> --eps 0 1 2 3

Signals (per row n -> n+1 of the dataset's own timeline):
  image motion = mean |I[n+1] - I[n]| (gray, 64x64) / (s[n+1] - s[n])
  joint motion = sum_j |q_j[n+1] - q_j[n]| (joints 0-5) / (s[n+1] - s[n])
s = source time (frame_map.parquet source_time_s for a retimed dataset, else frame / fps); both
signals are resampled onto a uniform source-time grid, so lags of original and retimed data are
directly comparable. lag = argmax of the cross-correlation over +-max-lag, parabola-refined.
lag > 0: the image changes AFTER the joints (image late / state early).
"""
import argparse
import glob
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", nargs="+", required=True)
ap.add_argument("--eps", type=int, nargs="*", help="episode indices (default: all)")
ap.add_argument("--cams", nargs="*", help="video keys or suffixes, e.g. cam0 cam9 (default: all)")
ap.add_argument("--max-lag", type=float, default=0.5, help="s")
ap.add_argument("--size", type=int, default=64, help="downscale for image motion")
ap.add_argument("--jobs", type=int, default=12)
ap.add_argument("--plot", default=None, help="PNG path for lag-vs-correlation curves")
ap.add_argument("--json", default=None, help="write all results here")
args = ap.parse_args()


def gray_frames(path, s):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", f"scale={s}:{s}:flags=area",
                          "-f", "rawvideo", "-pix_fmt", "gray", "-"], check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, s, s).astype(np.float32)


def xcorr_lag(a, b, K):
    """lag (rows, sub-frame) maximising corr(a[n], b[n + lag]), the curve, and the peak corr."""
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    n = len(a)
    lags = np.arange(-K, K + 1)
    c = np.array([np.mean(a[max(0, -l):n - max(0, l)] * b[max(0, l):n - max(0, -l)]) for l in lags])
    i = int(np.argmax(c))
    frac = 0.0
    if 0 < i < len(c) - 1:
        d = c[i - 1] - 2 * c[i] + c[i + 1]
        frac = 0.5 * (c[i - 1] - c[i + 1]) / d if d < 0 else 0.0
    return lags[i] + frac, lags, c, float(c[i])


def load(ds):
    ds = os.path.abspath(ds.rstrip("/"))
    info = json.load(open(os.path.join(ds, "meta", "info.json")))
    eps = pa.concat_tables([pq.read_table(f) for f in sorted(glob.glob(os.path.join(ds, "meta", "episodes", "*", "*.parquet")))]).to_pandas()
    data = pa.concat_tables([pq.read_table(f, columns=["observation.state", "episode_index", "frame_index"])
                             for f in sorted(glob.glob(os.path.join(ds, "data", "*", "*.parquet")))]).to_pandas()
    fm = os.path.join(ds, "frame_map.parquet")
    fm = pq.read_table(fm).to_pandas() if os.path.exists(fm) else None
    return ds, info, eps, data, fm


def main():
    results, curves = [], {}
    for ds_arg in args.dataset:
        ds, info, eps, data, fm = load(ds_arg)
        name = os.path.basename(ds)
        fps = info["fps"]
        vkeys = [k for k, v in info["features"].items() if v["dtype"] == "video"]
        if args.cams:
            vkeys = [k for k in vkeys if any(k == c or k.endswith("." + c) for c in args.cams)]
        sel = eps if args.eps is None else eps[eps.episode_index.isin(args.eps)]
        src_fps = fps  # retiming keeps fps; source grid = original frame rate
        K = int(round(args.max_lag * src_fps))
        jobs = []
        for _, e in sel.iterrows():
            ei = int(e["episode_index"])
            d = data[data.episode_index == ei].sort_values("frame_index")
            q = np.stack(d["observation.state"].to_numpy())[:, :6].astype(np.float64)
            if fm is not None:
                st = fm[fm.episode_index == ei].sort_values("frame_index").source_time_s.to_numpy()
            else:
                st = np.arange(len(q)) / fps
            for vk in vkeys:
                rel = info["video_path"].format(video_key=vk, chunk_index=int(e[f"videos/{vk}/chunk_index"]),
                                                file_index=int(e[f"videos/{vk}/file_index"]))
                jobs.append((ei, vk, os.path.join(ds, rel), q, st))

        def run(job):
            ei, vk, path, q, st = job
            img = gray_frames(path, args.size)
            n = min(len(img), len(q))
            q, img, st = q[:n], img[:n], st[:n]
            # motion per unit SOURCE time, at row midpoints, resampled to a uniform source grid
            ds_ = np.diff(st)
            ok = ds_ > 1e-6
            mid = 0.5 * (st[1:] + st[:-1])[ok]
            mq = (np.abs(np.diff(q, axis=0)).sum(1) / np.maximum(ds_, 1e-6))[ok]
            mi = (np.abs(np.diff(img, axis=0)).mean((1, 2)) / np.maximum(ds_, 1e-6))[ok]
            grid = np.arange(mid[0], mid[-1], 1.0 / src_fps)
            lag, lags, c, peak = xcorr_lag(np.interp(grid, mid, mq), np.interp(grid, mid, mi), K)
            return dict(dataset=name, episode=ei, cam=vk.split(".")[-1], lag_ms_source_time=float(lag / src_fps * 1000),
                        peak_corr=peak), (lags / src_fps, c)

        with ThreadPoolExecutor(args.jobs) as pool:
            for r, cv in pool.map(run, jobs):
                results.append(r)
                curves[(name, r["episode"], r["cam"])] = cv

    # report
    print(f"{'dataset':56s} {'cam':11s} {'eps':>4s} {'lag ms median':>14s} {'std':>7s} {'corr':>6s}")
    for name in dict.fromkeys(r["dataset"] for r in results):
        for cam in dict.fromkeys(r["cam"] for r in results if r["dataset"] == name):
            rr = [r for r in results if r["dataset"] == name and r["cam"] == cam]
            ls = np.array([r["lag_ms_source_time"] for r in rr])
            pc = np.array([r["peak_corr"] for r in rr])
            print(f"{name[:56]:56s} {cam:11s} {len(rr):4d} {np.median(ls):14.1f} {ls.std():7.1f} {np.median(pc):6.2f}")
    if len(args.dataset) == 2:
        a, b = [os.path.basename(os.path.abspath(d.rstrip("/"))) for d in args.dataset]
        diffs = []
        for r in results:
            if r["dataset"] != a:
                continue
            m = [x for x in results if x["dataset"] == b and x["episode"] == r["episode"] and x["cam"] == r["cam"]]
            if m:
                diffs.append(m[0]["lag_ms_source_time"] - r["lag_ms_source_time"])
        if diffs:
            diffs = np.array(diffs)
            print(f"\nlag difference 2nd - 1st per (episode, cam): median {np.median(diffs):.1f} ms, "
                  f"p90 |d| {np.percentile(np.abs(diffs), 90):.1f} ms, max |d| {np.abs(diffs).max():.1f} ms (n={len(diffs)})")
    if args.json:
        json.dump(results, open(args.json, "w"), indent=2)
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = list(dict.fromkeys(r["dataset"] for r in results))
        cams = list(dict.fromkeys(r["cam"] for r in results))
        fig, ax = plt.subplots(len(names), 1, figsize=(10, 3.2 * len(names)), squeeze=False, sharex=True)
        for i, nm in enumerate(names):
            for j, cam in enumerate(cams):
                cs = [curves[k] for k in curves if k[0] == nm and k[2] == cam]
                if cs:
                    lag = cs[0][0] * 1000
                    ax[i, 0].plot(lag, np.mean([c for _, c in cs], 0), label=cam, color=f"C{j}")
            ax[i, 0].axvline(0, color="k", lw=0.8)
            ax[i, 0].set_title(nm, fontsize=9)
            ax[i, 0].set_ylabel("corr (mean over eps)")
            ax[i, 0].legend(fontsize=7, ncol=4)
        ax[-1, 0].set_xlabel("lag ms, source time (>0 = image after joints)")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=110)
        print("plot ->", args.plot)


if __name__ == "__main__":
    main()
