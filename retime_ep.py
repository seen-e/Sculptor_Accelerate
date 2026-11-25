#!/usr/bin/env python3
"""Speed up one EVT episode: original frames inside suction pick/place windows, up to
--frame-step x faster elsewhere, joint velocity / acceleration kept within limits.

  PY=/home/seene/miniconda3/envs/isaaclab/bin/python
  $PY retime_ep.py --ep 35                      # -> modified_v1/EP035/
  $PY retime_ep.py --ep 35 --frame-step 2 --guard-pre 1.0 --out modified_v2

Method
  * protected windows: place = suction_do_ch3 1->0->1 pulse, pick = fluid_pressure drops
    below --pick-pressure; each widened by --guard-pre/--guard-post, merged, snapped to
    original frames. Rows inside are copied unchanged (all 30 dims), even if over limits.
  * every stretch between windows: q(tau) = Q(s(tau)), Q = cubic spline through all original
    joint samples (path unchanged), s = original time. All 6 joints share one s(tau), so
    they stay synchronized. s(tau) is time-optimal (TOPP-RA style reachability on a grid)
    under |q'| <= V, |q''| <= A, s' <= frame_step, s' = 1 at window boundaries.
  * stretch durations are made whole frames (bisect the stretch speed cap), so output is a
    uniform 30 Hz timeline and window rows land on original frames.
  * non-joint dims outside windows come from the nearest original frame (original_frame).
"""
import argparse
import json
import os
import time

import numpy as np
from scipy.interpolate import CubicSpline, make_smoothing_spline

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "EVT五条轨迹_state修改前后_20261106", "01_修改前_原始state30")
FPS = 30.0
POS = slice(0, 6)
VEL = slice(14, 20)
SUCTION, PRESSURE = 6, 10

ap = argparse.ArgumentParser()
ap.add_argument("--ep", type=int, default=35)
ap.add_argument("--frame-step", type=float, default=3.0,
                help="max speedup outside protected windows (3 = every 3rd frame)")
ap.add_argument("--vmax-deg", type=float, nargs=6, default=[63, 36, 90, 108, 162, 162])
ap.add_argument("--amax-deg", type=float, nargs=6, default=[120, 60, 90, 96, 192, 192])
ap.add_argument("--margin", type=float, default=0.98, help="plan to margin * limits")
ap.add_argument("--guard-pre", type=float, default=0.8, help="s before a suction event")
ap.add_argument("--guard-post", type=float, default=0.8, help="s after a suction event")
ap.add_argument("--max-expand", type=float, default=1.0,
                help="max s a window edge is pushed outward to reach a within-limits frame")
ap.add_argument("--pick-pressure", type=float, default=-30.0, help="pick = pressure below this")
ap.add_argument("--smooth-lam", type=float, default=1e-4,
                help="smoothing spline lambda for the joint path outside windows (0 = exact interpolation)")
ap.add_argument("--blend", type=float, default=0.5,
                help="s over which the smoothed path is blended back onto the window edge frames")
ap.add_argument("--subdiv", type=int, default=4, help="planning grid points per original frame")
ap.add_argument("--out", default="modified_v1")
ap.add_argument("--no-plot", action="store_true")
args = ap.parse_args()

V = np.deg2rad(args.vmax_deg)
A = np.deg2rad(args.amax_deg)


def base_path(t, q):
    """Joint path vs original time: smoothing spline (30 Hz joint data has acceleration noise
    well above the limits), or exact interpolation with --smooth-lam 0. Called as f(s, nu)."""
    if args.smooth_lam <= 0:
        cs = CubicSpline(t, q)
        return lambda s, nu=0: cs(s, nu)
    sp = [make_smoothing_spline(t, q[:, j], lam=args.smooth_lam) for j in range(q.shape[1])]
    der = [[p, p.derivative(1), p.derivative(2)] for p in sp]
    return lambda s, nu=0: np.stack([d[nu](s) for d in der], -1)


def stretch_path(base, t, q, fa, fb):
    """base plus a quintic blend that removes its offset to the original frames fa / fb,
    so the path meets the protected rows exactly."""
    L = min(args.blend, 0.5 * (t[fb] - t[fa]))
    ends = [(q[fa] - base(t[fa]), t[fa], 1.0), (q[fb] - base(t[fb]), t[fb], -1.0)]

    def f(s, nu=0):
        s = np.asarray(s, dtype=np.float64)
        out = base(s, nu)
        for d, s0, sgn in ends:
            r = (s - s0) * sgn / L
            m = (r >= 0) & (r < 1)
            r = np.where(m, r, 0.0)
            if nu == 0:
                w = np.where(m, 1 - (10 * r ** 3 - 15 * r ** 4 + 6 * r ** 5), 0.0)
            elif nu == 1:
                w = np.where(m, -(30 * r ** 2 - 60 * r ** 3 + 30 * r ** 4) * sgn / L, 0.0)
            else:
                w = np.where(m, -(60 * r - 180 * r ** 2 + 120 * r ** 3) / L ** 2, 0.0)
            out = out + np.multiply.outer(w, d)
        return out
    return f


def protected_windows(t, state, cs):
    """Merged [start, end] frame ranges around suction pick/place events. Edges are pushed
    outward (<= --max-expand s) to frames where the original motion is within limits, so the
    s' = 1 hand-over at the edge is feasible."""
    suc, p = state[:, SUCTION], state[:, PRESSURE]
    events = []
    falls = np.where(np.diff(suc) < 0)[0] + 1
    rises = np.where(np.diff(suc) > 0)[0] + 1
    for f in falls:
        r = rises[rises > f]
        events.append(("place", t[f], t[r[0]] if len(r) else t[-1]))
    hold = (p < args.pick_pressure).astype(int)
    for i in np.where(np.diff(hold) == 1)[0] + 1:
        events.append(("pick", t[i], t[i]))
    safe = (np.all(np.abs(cs(t, 1)) <= V * args.margin, axis=1)
            & np.all(np.abs(cs(t, 2)) <= A * args.margin, axis=1))
    n_exp = int(round(args.max_expand * FPS))
    spans = []
    for _, a, b in events:
        fa = max(int(np.searchsorted(t, a - args.guard_pre, side="right")) - 1, 0)
        fb = min(int(np.searchsorted(t, b + args.guard_post)), len(t) - 1)
        for _ in range(n_exp):
            if safe[fa] or fa == 0:
                break
            fa -= 1
        for _ in range(n_exp):
            if safe[fb] or fb == len(t) - 1:
                break
            fb += 1
        spans.append((fa, fb))
    frames = []
    for fa, fb in sorted(spans):
        if frames and fa <= frames[-1][1]:
            frames[-1][1] = max(frames[-1][1], fb)
        else:
            frames.append([fa, fb])
    return events, [tuple(f) for f in frames]


def x_range(G, h):
    """min / max of x over the 2D polygon {(x, u): G @ (x, u) <= h}, None if empty."""
    i, j = np.triu_indices(len(h), 1)
    det = G[i, 0] * G[j, 1] - G[j, 0] * G[i, 1]
    ok = np.abs(det) > 1e-12
    det = np.where(ok, det, 1.0)
    x = (h[i] * G[j, 1] - h[j] * G[i, 1]) / det
    u = (G[i, 0] * h[j] - G[j, 0] * h[i]) / det
    pts = np.stack([x[ok], u[ok]], 1)
    feas = np.all(pts @ G.T <= h + 1e-9 * (1 + np.abs(h)), axis=1)
    if not feas.any():
        return None
    return pts[feas, 0].min(), pts[feas, 0].max()


def topp(s, d1, d2, cap, fix_start, fix_end):
    """Time-optimal x = s'^2 on grid s. Returns x, u (= s'' per segment), #fallback points."""
    n = len(s)
    Vp, Ap = V * args.margin, A * args.margin
    with np.errstate(divide="ignore"):
        xhi = np.minimum(cap ** 2, np.min((Vp / np.abs(d1)) ** 2, axis=1))
    xlo = np.zeros(n)
    Alim = np.tile(Ap, (n, 1))
    for k, fixed in ((0, fix_start), (n - 1, fix_end)):
        if fixed:  # window boundary: s' = 1, allow the original acceleration there
            xlo[k] = xhi[k] = 1.0
            Alim[k] = np.maximum(Ap, np.abs(d2[k]) + 1e-9)
    fallback = 0

    def acc_rows(k, dl):
        """|q''| <= A at both ends of segment k (constant s'' = u, x_{k+1} = x + 2 dl u)."""
        g = np.concatenate([np.stack([d2[k], d1[k]], 1),
                            np.stack([d2[k + 1], 2 * dl * d2[k + 1] + d1[k + 1]], 1)])
        h = np.concatenate([Alim[k], Alim[k + 1]])
        return np.vstack([g, -g]), np.concatenate([h, h])

    # backward: K[k] = states at k from which K[k+1] is reachable
    Klo, Khi = xlo.copy(), xhi.copy()
    for k in range(n - 2, -1, -1):
        dl = s[k + 1] - s[k]
        G, h = acc_rows(k, dl)
        G = np.vstack([G, [[1, 0], [-1, 0], [1, 2 * dl], [-1, -2 * dl]]])
        h = np.concatenate([h, [xhi[k], -xlo[k], Khi[k + 1], -Klo[k + 1]]])
        r = x_range(G, h)
        if r is None:
            fallback += 1
            continue  # keep [xlo, xhi]; forward pass clips
        Klo[k], Khi[k] = r

    # forward: greedy max acceleration while staying inside K
    x = np.empty(n)
    u = np.zeros(n - 1)
    x[0] = Khi[0]
    for k in range(n - 1):
        dl = s[k + 1] - s[k]
        lo, hi = (Klo[k + 1] - x[k]) / (2 * dl), (Khi[k + 1] - x[k]) / (2 * dl)
        G, h = acc_rows(k, dl)
        rhs, b = h - G[:, 0] * x[k], G[:, 1]  # b u <= rhs
        pos, neg = b > 1e-12, b < -1e-12
        ulo = max(lo, (rhs[neg] / b[neg]).max(initial=-np.inf))
        uhi = min(hi, (rhs[pos] / b[pos]).min(initial=np.inf))
        if ulo > uhi + 1e-12:
            fallback += 1
            uhi = hi if uhi > hi else max(uhi, lo)
        u[k] = uhi
        x[k + 1] = max(x[k] + 2 * dl * u[k], 1e-10)
    return x, u, fallback


def stretch_duration(s, x):
    sd = np.sqrt(x)
    return np.sum(2 * np.diff(s) / (sd[:-1] + sd[1:]))


def retime_stretch(cs, t, fa, fb, fix_start, fix_end):
    """Plan original frames fa..fb; duration made a whole number of output frames."""
    s = np.concatenate([np.linspace(t[k], t[k + 1], args.subdiv, endpoint=False) for k in range(fa, fb)]
                       + [[t[fb]]])
    d1, d2 = cs(s, 1), cs(s, 2)
    x, u, fb_n = topp(s, d1, d2, args.frame_step, fix_start, fix_end)
    D = stretch_duration(s, x)
    M = max(1, int(np.ceil(D * FPS - 1e-6)))
    target = M / FPS
    lo, hi = 1e-3, args.frame_step  # bisect the speed cap so the stretch lasts exactly M frames
    for _ in range(25):
        if abs(D - target) < 2e-5:
            break
        mid = 0.5 * (lo + hi)
        x, u, fb_n = topp(s, d1, d2, mid, fix_start, fix_end)
        D = stretch_duration(s, x)
        lo, hi = (mid, hi) if D > target else (lo, mid)  # too slow -> raise the cap
    c = target / D  # residual, ~1
    return s, x, u, M, c, fb_n


def sample_stretch(s, x, u, M, c, taus):
    """s, s', s'' at local times taus (0..M/FPS), constant s'' per grid segment, time scaled by c."""
    sd = np.sqrt(x)
    seg_t = 2 * np.diff(s) / (sd[:-1] + sd[1:])
    t0 = np.concatenate([[0.0], np.cumsum(seg_t)])
    tl = np.asarray(taus) / c
    k = np.clip(np.searchsorted(t0, tl, side="right") - 1, 0, len(seg_t) - 1)
    dt = tl - t0[k]
    ss = s[k] + sd[k] * dt + 0.5 * u[k] * dt ** 2
    sdot = (sd[k] + u[k] * dt) / c
    sdd = u[k] / c ** 2
    return np.minimum(ss, s[-1]), sdot, sdd, (t0, seg_t)


def main():
    ep = f"EP{args.ep:03d}"
    src = np.load(os.path.join(SRC, ep, "state30.npz"))
    t, state = src["time_s"].astype(np.float64), src["state"].astype(np.float64)
    names = [str(n) for n in src["state_names"]]
    n = len(t)
    base = base_path(t, state[:, POS])
    events, windows = protected_windows(t, state, base)

    # stretches cover the episode: ("prot", fa, fb) rows copied, ("move", fa, fb) retimed
    stretches, f = [], 0
    for wa, wb in windows:
        if wa > f:
            stretches.append(("move", f, wa))
        stretches.append(("prot", wa, wb))
        f = wb
    if f < n - 1:
        stretches.append(("move", f, n - 1))

    rows, src_t, sdots, sdds, kinds = [], [], [], [], []  # per output frame
    report, dense = [], {"move_v": [], "move_a": []}
    t_start = time.time()
    for si, (kind, fa, fb) in enumerate(stretches):
        has_prev = si > 0
        has_next = si < len(stretches) - 1
        if kind == "prot":
            first = fa + 1 if has_prev and stretches[si - 1][0] == "prot" else fa  # shared frame
            for k in range(first, fb + 1):
                rows.append(("orig", k))
                src_t.append(t[k]); sdots.append(1.0); sdds.append(0.0); kinds.append(1)
            report.append(dict(kind="protected", t0=float(t[fa]), t1=float(t[fb]),
                               frames=int(fb - fa + 1)))
            continue
        fix_s = has_prev and stretches[si - 1][0] == "prot"
        fix_e = has_next and stretches[si + 1][0] == "prot"
        cs = stretch_path(base, t, state[:, POS], fa, fb)
        s, x, u, M, c, nfb = retime_stretch(cs, t, fa, fb, fix_s, fix_e)
        taus = np.arange(M + 1) / FPS
        ss, sdot, sdd, (t0, seg_t) = sample_stretch(s, x, u, M, c, taus)
        # frames shared with a protected neighbour are emitted there as original rows
        for m in range(1 if fix_s else 0, M if fix_e else M + 1):
            rows.append(("spline", ss[m], cs)); src_t.append(ss[m]); sdots.append(sdot[m]); sdds.append(sdd[m])
            kinds.append(0)
        # dense check of the continuous retimed motion inside this stretch
        td = np.linspace(0, M / FPS, max(50, M * 20))
        s_d, sd_d, sdd_d, _ = sample_stretch(s, x, u, M, c, td)
        dense["move_v"].append(np.abs(cs(s_d, 1) * sd_d[:, None]).max(0))
        dense["move_a"].append(np.abs(cs(s_d, 2) * sd_d[:, None] ** 2 + cs(s_d, 1) * sdd_d[:, None]).max(0))
        report.append(dict(kind="retimed", t0=float(t[fa]), t1=float(t[fb]),
                           orig_duration=float(t[fb] - t[fa]), new_duration=M / FPS,
                           speedup=float((t[fb] - t[fa]) / (M / FPS)), residual_time_scale=float(c),
                           infeasible_grid_points=int(nfb)))

    # build output state
    N = len(rows)
    out = np.empty((N, state.shape[1]))
    orig_frame = np.empty(N, dtype=np.int64)
    acc = np.empty((N, 6))
    for i, r in enumerate(rows):
        if r[0] == "orig":
            k = r[1]
            out[i] = state[k]
            orig_frame[i] = k
            acc[i] = base(t[k], 2)
        else:
            sv, cs = r[1], r[2]
            k = int(np.clip(np.searchsorted(t, sv), 1, n - 1))
            k = k - 1 if sv - t[k - 1] < t[k] - sv else k
            out[i] = state[k]
            orig_frame[i] = k
            out[i, POS] = cs(sv)
            out[i, VEL] = cs(sv, 1) * sdots[i]
            acc[i] = cs(sv, 2) * sdots[i] ** 2 + cs(sv, 1) * sdds[i]
    time_s = np.arange(N) / FPS
    protected = np.array(kinds, dtype=bool)
    action = np.vstack([out[1:, POS], out[-1:, POS]])  # action_k = joint position of frame k+1

    # checks
    vmax_move = np.max(dense["move_v"], 0) if dense["move_v"] else np.zeros(6)
    amax_move = np.max(dense["move_a"], 0) if dense["move_a"] else np.zeros(6)
    fd_v = np.abs(np.diff(out[:, POS], axis=0) * FPS)
    summary = dict(
        episode=args.ep, original_frames=int(n), new_frames=int(N),
        original_duration=float(t[-1] - t[0]), new_duration=float(time_s[-1]),
        speedup=float((t[-1] - t[0]) / time_s[-1]),
        protected_original_seconds=float(sum(r["t1"] - r["t0"] for r in report if r["kind"] == "protected")),
        retimed_peak_velocity_rad_s=vmax_move.tolist(), velocity_limit_rad_s=V.tolist(),
        velocity_utilization=(vmax_move / V).tolist(),
        retimed_peak_acceleration_rad_s2=amax_move.tolist(), acceleration_limit_rad_s2=A.tolist(),
        acceleration_utilization=(amax_move / A).tolist(),
        protected_peak_velocity_rad_s=np.abs(np.gradient(state[:, POS], t, axis=0))[
            np.isin(np.arange(n), orig_frame[protected])].max(0).tolist(),
        output_fd_peak_velocity_rad_s=fd_v.max(0).tolist(),
        infeasible_grid_points=int(sum(r.get("infeasible_grid_points", 0) for r in report)),
        endpoint_error_rad=float(np.abs(out[-1, POS] - state[-1, POS]).max()),
        compute_seconds=time.time() - t_start,
    )

    out_dir = os.path.join(HERE, args.out, ep)
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "state30.npz"), time_s=time_s, state=out.astype(np.float32),
             state_names=np.array(names), original_frame=orig_frame, source_time_s=np.array(src_t),
             protected=protected, path_speed=np.array(sdots), joint_acceleration_rad_s2=acc,
             action=action.astype(np.float32))
    header = ["frame", "time_s", "original_frame", "source_time_s", "protected"] + names
    with open(os.path.join(out_dir, "state30.csv"), "w", encoding="utf-8-sig") as fcsv:
        fcsv.write(",".join(header) + "\n")
        for i in range(N):
            fcsv.write(",".join([str(i), repr(float(time_s[i])), str(orig_frame[i]), repr(float(src_t[i])),
                                 str(int(protected[i]))] + [repr(float(v)) for v in out[i]]) + "\n")
    with open(os.path.join(out_dir, "参数与结果.json"), "w", encoding="utf-8") as fj:
        json.dump(dict(params=vars(args), events=[dict(kind=k, t0=float(a), t1=float(b)) for k, a, b in events],
                       summary=summary, stretches=report), fj, ensure_ascii=False, indent=2)

    print(f"[{ep}] {t[-1] - t[0]:.2f}s -> {time_s[-1]:.2f}s  (x{summary['speedup']:.3f}), "
          f"{n} -> {N} frames, protected {summary['protected_original_seconds']:.1f}s in {len(windows)} windows")
    print("  retimed peak |v| / V :", np.round(vmax_move / V, 3))
    print("  retimed peak |a| / A :", np.round(amax_move / A, 3))
    print("  infeasible grid points:", summary["infeasible_grid_points"], " compute %.1fs" % summary["compute_seconds"])
    print("  ->", out_dir)

    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
        for a in ax:
            for i0 in np.where(np.diff(np.r_[0, protected.astype(int)]) == 1)[0]:
                i1 = i0 + np.argmin(protected[i0:]) if not protected[i0:].all() else N - 1
                a.axvspan(time_s[i0], time_s[i1], color="orange", alpha=0.2, lw=0)
        ax[0].plot(time_s, sdots, "k", lw=1)
        ax[0].set_ylabel("speedup  ds/dτ")
        ax[0].set_title(f"{ep}: {t[-1]:.1f}s -> {time_s[-1]:.1f}s, orange = protected (original rows)")
        for j in range(6):
            ax[1].plot(time_s, out[:, 14 + j] / V[j], lw=0.8, label=f"J{j + 1}")
            ax[2].plot(time_s, acc[:, j] / A[j], lw=0.8, label=f"J{j + 1}")
        for a, lab in ((ax[1], "v / Vmax"), (ax[2], "a / Amax")):
            a.axhline(1, color="r", ls="--", lw=0.8)
            a.axhline(-1, color="r", ls="--", lw=0.8)
            a.set_ylabel(lab)
            a.legend(ncol=6, fontsize=8, loc="upper right")
        ax[2].set_ylim(-2, 2)
        ax[2].set_xlabel("new time (s)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "retime.png"), dpi=110)


if __name__ == "__main__":
    main()
