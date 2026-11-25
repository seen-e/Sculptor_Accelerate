#!/usr/bin/env python3
"""Check a (retimed) LeRobot dataset against arm jerk limits and base acceleration limits.

  PY=/home/seene/miniconda3/envs/isaaclab/bin/python
  $PY check_limits.py openpi_sculptor_dataset_retime/lerobot_20260916_fixed640_guard_40eps_retime
  -> <dataset>/limit_check/{summary.png, epXXX_limits.png, limit_report.json}

Derivatives at 30 Hz, per episode:
  * "fd"  : plain finite differences (np.gradient applied 3x for jerk, 1x for base accel).
            Shows every step the trajectory really contains, plus sensor noise.
  * "sg"  : Savitzky-Golay derivative (--sg-window frames, cubic) - noise-robust, but smooths
            short jerk spikes (window 7 = 0.23 s) so it under-reports bang-bang switches.
Arm joints: observation.state dims 0-5 (rad). Base: observation.state 7-9 (vx, vy m/s, wz rad/s)
and action 7-9 (commanded). XY base acceleration is the norm of (ax, ay).
If frame_map.parquet exists (retime_lerobot.py output), rows are split into protected
(original 1x data) vs retimed rows.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

ap = argparse.ArgumentParser()
ap.add_argument("dataset")
ap.add_argument("--jmax-deg", type=float, nargs=6, default=[1400, 1200, 1200, 1800, 1800, 1800],
                help="arm joint jerk limits, deg/s^3")
ap.add_argument("--amax-deg", type=float, nargs=6, default=[120, 60, 90, 96, 192, 192],
                help="arm joint acceleration limits, deg/s^2 (reported only)")
ap.add_argument("--base-axy", type=float, default=0.343, help="base XY acceleration limit, m/s^2")
ap.add_argument("--base-aw", type=float, default=0.49, help="base yaw acceleration limit, rad/s^2")
ap.add_argument("--sg-window", type=int, default=7, help="Savitzky-Golay window, frames (odd)")
ap.add_argument("--episodes", type=int, nargs="*", default=None, help="only these episodes")
ap.add_argument("--out", default=None, help="default: <dataset>/limit_check")
ap.add_argument("--no-episode-plots", action="store_true")
args = ap.parse_args()

J = np.deg2rad(args.jmax_deg)
A = np.deg2rad(args.amax_deg)
CH = [f"J{i + 1}" for i in range(6)] + ["base_xy", "base_wz"]
LIM = np.r_[J, args.base_axy, args.base_aw]


def deriv(x, n, dt, method):
    if method == "sg":
        w = min(args.sg_window, len(x) - (1 - len(x) % 2))
        return savgol_filter(x, w, 3, deriv=n, delta=dt, axis=0, mode="interp")
    for _ in range(n):
        x = np.gradient(x, dt, axis=0)
    return x


def base_acc(v, dt, method):
    a = deriv(v, 1, dt, method)
    return np.hypot(a[:, 0], a[:, 1]), a[:, 2]


def episode(ep, state, action, prot, fps):
    dt = 1.0 / fps
    q = state[:, 0:6].astype(np.float64)
    r = {}
    for m in ("fd", "sg"):
        jerk = deriv(q, 3, dt, m)
        acc = deriv(q, 2, dt, m)
        bxy, bw = base_acc(state[:, 7:10].astype(np.float64), dt, m)
        cxy, cw = base_acc(action[:, 7:10].astype(np.float64), dt, m)
        r[m] = dict(jerk=jerk, acc=acc, util=np.column_stack([np.abs(jerk) / J, bxy / args.base_axy,
                                                               np.abs(bw) / args.base_aw]),
                    cmd_util=np.column_stack([cxy / args.base_axy, np.abs(cw) / args.base_aw]))
    return r


def stats(util, mask):
    if not mask.any():
        return None
    u = util[mask]
    return dict(peak=u.max(0).round(3).tolist(), p99=np.quantile(u, 0.99, axis=0).round(3).tolist(),
                exceed_frac=(u > 1).mean(0).round(4).tolist())


def plot_episode(path, ep, t, r, prot, sdot):
    fig, ax = plt.subplots(9, 1, figsize=(14, 17), sharex=True)
    spans = []
    if prot is not None:
        d = np.diff(np.r_[0, prot.astype(int), 0])
        spans = list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))
    for a in ax:
        for s0, s1 in spans:
            a.axvspan(t[s0], t[s1], color="tab:orange", alpha=0.12, lw=0)
    if sdot is not None:
        ax[0].plot(t, sdot, lw=0.8, color="k")
        ax[0].set_ylabel("speedup")
    ax[0].set_title(f"episode {ep}: |jerk|/Jmax and |base acc|/Amax  (grey = finite diff, "
                    f"blue = Savitzky-Golay {args.sg_window}f, orange = protected 1x rows)")
    for c in range(8):
        a = ax[c + 1]
        if c < 6:
            yf, ys = r["fd"]["jerk"][:, c] / J[c], r["sg"]["jerk"][:, c] / J[c]
        elif c == 6:
            yf, ys = r["fd"]["util"][:, 6], r["sg"]["util"][:, 6]
        else:
            yf, ys = r["fd"]["util"][:, 7], r["sg"]["util"][:, 7]
        a.plot(t, yf, lw=0.5, color="0.6")
        a.plot(t, ys, lw=0.9, color="tab:blue")
        a.axhline(1, color="r", lw=0.8, ls="--")
        if c < 6 or c == 7:
            a.axhline(-1, color="r", lw=0.8, ls="--")
        lim = max(1.5, min(np.quantile(np.abs(yf), 0.995) * 1.2, 20))
        a.set_ylim(-lim if c != 6 else 0, lim)
        a.set_ylabel(CH[c])
        a.text(1.002, 0.5, f"peak fd {np.abs(yf).max():.1f}\npeak sg {np.abs(ys).max():.1f}",
               transform=a.transAxes, fontsize=8, va="center")
    ax[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def plot_summary(path, reps, all_util, all_prot, has_prot):
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1])
    eps = [r["episode"] for r in reps]
    for i, m in enumerate(("fd", "sg")):
        ax = fig.add_subplot(gs[0, i])
        key = "retimed" if has_prot else "all"
        M = np.array([r[m][key]["peak"] if r[m][key] else [np.nan] * 8 for r in reps])
        im = ax.imshow(np.log10(np.maximum(M.T, 1e-3)), aspect="auto", cmap="RdYlGn_r", vmin=-1, vmax=1)
        ax.set_yticks(range(8), CH)
        ax.set_xticks(range(0, len(eps), max(1, len(eps) // 20)), eps[::max(1, len(eps) // 20)])
        ax.set_xlabel("episode")
        ax.set_title(f"peak utilisation per episode ({m}, {key} rows)\ncolour = log10(peak/limit); >0 = over limit")
        cb = fig.colorbar(im, ax=ax)
        cb.set_ticks([-1, 0, 1], labels=["0.1x", "1x", "10x"])
    for i, m in enumerate(("fd", "sg")):
        ax = fig.add_subplot(gs[1, i])
        u, p = all_util[m], all_prot
        xs = np.logspace(-1.5, 1.5, 200)

        def exceed(v):
            v = np.sort(v)
            return 1.0 - np.searchsorted(v, xs, side="right") / max(len(v), 1)
        for c, col in zip(range(8), plt.cm.tab10.colors):
            if has_prot:
                ax.plot(xs, exceed(u[~p, c]), color=col, lw=1.3, label=f"{CH[c]}")
                ax.plot(xs, exceed(u[p, c]), color=col, lw=0.9, ls=":")
            else:
                ax.plot(xs, exceed(u[:, c]), color=col, lw=1.3, label=CH[c])
        ax.axvline(1, color="r", ls="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-5, 1)
        ax.grid(True, which="major", alpha=0.3)
        ax.set_xlabel("|value| / limit")
        ax.set_ylabel("fraction of frames above x")
        ax.set_title(f"exceedance, all frames ({m})" + ("  solid = retimed rows, dotted = protected 1x rows" if has_prot else ""))
        ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    ds = args.dataset.rstrip("/")
    out = args.out or os.path.join(ds, "limit_check")
    os.makedirs(out, exist_ok=True)
    fps = json.load(open(os.path.join(ds, "meta", "info.json")))["fps"]
    files = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(os.path.join(ds, "data")) for f in fs
                   if f.endswith(".parquet"))
    df = pd.concat([pd.read_parquet(f, columns=["observation.state", "action", "episode_index", "frame_index"])
                    for f in files], ignore_index=True)
    fm_path = os.path.join(ds, "frame_map.parquet")
    has_prot = os.path.exists(fm_path)
    if has_prot:
        fm = pd.read_parquet(fm_path, columns=["episode_index", "frame_index", "protected", "path_speed"])
        df = df.merge(fm, on=["episode_index", "frame_index"], how="left", validate="1:1")
    df = df.sort_values(["episode_index", "frame_index"])

    reps, all_util, all_prot = [], {"fd": [], "sg": []}, []
    for ep, e in df.groupby("episode_index"):
        if args.episodes is not None and ep not in args.episodes:
            continue
        state, action = np.stack(e["observation.state"].values), np.stack(e["action"].values)
        if len(state) < args.sg_window + 2:
            continue
        prot = e["protected"].to_numpy(bool) if has_prot else None
        sdot = e["path_speed"].to_numpy() if has_prot else None
        r = episode(ep, state, action, prot, fps)
        rep = dict(episode=int(ep), frames=len(state))
        mask_all = np.ones(len(state), bool)
        for m in ("fd", "sg"):
            rep[m] = dict(all=stats(r[m]["util"], mask_all),
                          base_cmd_all=stats(r[m]["cmd_util"], mask_all),
                          acc_peak_util=(np.abs(r[m]["acc"]).max(0) / A).round(3).tolist())
            if has_prot:
                rep[m]["retimed"] = stats(r[m]["util"], ~prot)
                rep[m]["protected"] = stats(r[m]["util"], prot)
            all_util[m].append(r[m]["util"])
        all_prot.append(prot if has_prot else np.zeros(len(state), bool))
        reps.append(rep)
        if not args.no_episode_plots:
            plot_episode(os.path.join(out, f"ep{ep:03d}_limits.png"), ep, np.arange(len(state)) / fps, r, prot, sdot)
        s = rep["sg"]["retimed" if has_prot else "all"]
        print(f"ep {ep:3d}  sg peak " + " ".join(f"{c}={v:.2f}" for c, v in zip(CH, s["peak"])), flush=True)

    all_util = {m: np.vstack(v) for m, v in all_util.items()}
    all_prot = np.concatenate(all_prot)
    plot_summary(os.path.join(out, "summary.png"), reps, all_util, all_prot, has_prot)

    total = {}
    for m in ("fd", "sg"):
        total[m] = dict(all=stats(all_util[m], np.ones(len(all_prot), bool)))
        if has_prot:
            total[m]["retimed"] = stats(all_util[m], ~all_prot)
            total[m]["protected"] = stats(all_util[m], all_prot)
    json.dump(dict(dataset=os.path.abspath(ds), channels=CH,
                   limits=dict(jmax_deg=args.jmax_deg, base_axy=args.base_axy, base_aw=args.base_aw),
                   sg_window=args.sg_window, total=total, episodes=reps),
              open(os.path.join(out, "limit_report.json"), "w"), indent=1)
    print("\nchannels:", CH)
    for m in ("fd", "sg"):
        for k, v in total[m].items():
            print(f"[{m}] {k:9s} peak {v['peak']}\n{'':15s} p99  {v['p99']}\n{'':15s} >1   {v['exceed_frac']}")
    print("->", out)


if __name__ == "__main__":
    main()
