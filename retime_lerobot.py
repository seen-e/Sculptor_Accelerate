#!/usr/bin/env python3
"""retime_ep.py applied to a whole LeRobot v3.0 dataset (openpi_sculptor_dataset).

  PY=/home/seene/miniconda3/envs/isaaclab/bin/python
  $PY retime_lerobot.py --src openpi_sculptor_dataset/lerobot_20260918_fixed640_guard_12eps
  $PY retime_lerobot.py --src openpi_sculptor_dataset/lerobot_*        # all datasets
  -> openpi_sculptor_dataset_retime/<name>_retime/

EE hover cut (before retiming, --no-dwell-cut to disable): runs of >= --dwell-min s (0.3) where
  EE linear speed < --dwell-lin (0.01 m/s) AND EE angular speed < --dwell-ang (0.1 rad/s) AND the
  base is still (|v| < 0.02 m/s, |wz| < 0.05 rad/s), outside protected windows. EE speeds come from
  URDF forward kinematics of elfin_joint1-6 (arm-base frame). Only the first and last frame of a
  hover are kept; the gap is compressed to one frame interval and the planner below crosses it
  within the limits. Rows inside a cut use the earlier image until the joints are >= half way to
  the later frame, then the later image (frame_map.in_cut marks them).
Joint motion: exactly the retime_ep.py method (same functions and defaults):
  protected windows (suction place pulse / pick pressure drop, +-0.8 s) keep the original rows
  at 1x; between them q(tau) = Q(s(tau)) on a smoothed spline path, s(tau) time-optimal under
  |q'| <= V, |q''| <= A, s' <= --frame-step; output is a uniform 30 Hz timeline.

Per output row n (time n / 30, source time s_n on the original timeline):
  * joints 0-5  : Q(s_n) (protected rows: original values)
  * joint vel   : Q'(s_n) * s'   (dims 14-19)
  * base_vx/vy/wz state (7-9): nearest original frame * s'   (retime_ep.py did not scale these);
                  with --base-axy/--base-aw: smoothed original base velocity at s_n * s' instead
  * other dims  : nearest original frame (original_frame = round(s_n * 30))
  * images      : the same nearest original frame -> image time offset |s_n - frame/30| <= 1/60 s
                  of original time (0 on protected rows). Images are never interpolated.
  * action      : joints = joints of row n+1 (the next target; protected->protected pairs keep
                  the recorded action), dims 6-9 = original action row round(s_{n+1}*30) - 1,
                  base velocity actions * s' (with base limits: smoothed original command at
                  s_{n+1} - 1/30 * s').
Optional limits (off by default; without them the output is exactly as before):
  --jmax-deg J1..J6  arm jerk, deg/s^3: after topp() each stretch is re-planned as a sparse LP in s'^2
                     with |a_k - a_(k-1)| <= Jmax * dt per grid step (also across the protected-window
                     boundaries) on top of the velocity/acceleration rows (see jerk_lp). Rows that the
                     original data make impossible use a penalised slack and are counted in the report.
  --base-axy, --base-aw  base acceleration (m/s^2, rad/s^2): the retimed base velocity is vb(s) * s', its
                     acceleration vb'(s) s'^2 + vb(s) s'' is constrained in the same planner as the joints.
  The report and plots then add 30 Hz finite-difference |jerk|/Jmax and base |acc|/limit.
Outputs: LeRobot dataset + frame_map.parquet (source_time_s, original_frame_index, path_speed,
protected) + retime_report.json (per-episode speedup, limit utilisation, image offset)
+ plots/epXXX_retime.png (same figure as retime_ep.py: speedup, v/Vmax, a/Amax; --no-plot to skip).
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.interpolate import CubicSpline, make_smoothing_spline
from scipy.ndimage import uniform_filter1d
from retime_config import parse_config_args

POS = slice(0, 6)
VEL = slice(14, 20)
BASE_VEL = slice(7, 10)
ACTION_BASE_VEL = slice(7, 10)
SUCTION, PRESSURE = 6, 10
QUANTILES = [0.01, 0.10, 0.50, 0.90, 0.99]

ap = argparse.ArgumentParser(allow_abbrev=False)
ap.add_argument("--src", nargs="+", required=True, help="LeRobot dataset dir(s)")
ap.add_argument("--out-root", default=None, help="default: <parent of src>_retime/")
# --- retime_ep.py parameters (same defaults) ---
ap.add_argument("--frame-step", type=float, default=3.0, help="max speedup outside protected windows")
ap.add_argument("--vmax-deg", type=float, nargs=6, default=[63, 36, 90, 108, 162, 162])
ap.add_argument("--amax-deg", type=float, nargs=6, default=[120, 60, 90, 96, 192, 192])
ap.add_argument("--margin", type=float, default=0.98)
ap.add_argument("--guard-pre", type=float, default=0.8)
ap.add_argument("--guard-post", type=float, default=0.8)
ap.add_argument("--max-expand", type=float, default=1.0)
ap.add_argument("--pick-pressure", type=float, default=-30.0)
ap.add_argument("--smooth-lam", type=float, default=1e-4)
ap.add_argument("--blend", type=float, default=0.5)
ap.add_argument("--subdiv", type=int, default=4)
# --- optional extra limits (all off unless given; without them the output is unchanged) ---
ap.add_argument("--jmax-deg", type=float, nargs=6, default=None,
                help="arm joint jerk limits, deg/s^3 (e.g. 1400 1200 1200 1800 1800 1800)")
ap.add_argument("--base-axy", type=float, default=None, help="base XY acceleration limit |(ax, ay)|, m/s^2 (e.g. 0.343)")
ap.add_argument("--base-aw", type=float, default=None, help="base yaw acceleration limit, rad/s^2 (e.g. 0.49)")
ap.add_argument("--base-smooth-lam", type=float, default=1e-4,
                help="smoothing spline lambda for the base velocity path used by --base-axy/--base-aw")
# --- EE hover cut (frames removed before retiming; all conditions must hold) ---
ap.add_argument("--no-dwell-cut", action="store_true", help="do not cut EE hovers")
ap.add_argument("--dwell-lin", type=float, default=0.01, help="EE linear speed below this, m/s")
ap.add_argument("--dwell-ang", type=float, default=0.1, help="EE angular speed below this, rad/s")
ap.add_argument("--dwell-base-v", type=float, default=0.02, help="base |(vx, vy)| below this, m/s")
ap.add_argument("--dwell-base-w", type=float, default=0.05, help="base |wz| below this, rad/s")
ap.add_argument("--dwell-min", type=float, default=0.3, help="minimum hover duration, s")
# --- dataset / video ---
ap.add_argument("--workers", type=int, default=16, help="parallel episode retiming processes")
ap.add_argument("--encoder", choices=["nvenc", "x264"], default="nvenc")
ap.add_argument("--qp", type=int, default=18)
ap.add_argument("--jobs", type=int, default=6, help="parallel video re-encodes")
ap.add_argument("--verify", type=int, default=2, help="episodes per dataset to pixel-check (0 = off)")
ap.add_argument("--overwrite", action="store_true")
ap.add_argument("--no-plot", action="store_true", help="skip the per-episode plots/epXXX_retime.png")
args = parse_config_args(ap, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'config.yaml'))

V = np.deg2rad(args.vmax_deg)
A = np.deg2rad(args.amax_deg)
JMAX = None if args.jmax_deg is None else np.deg2rad(args.jmax_deg)
BASE_ON = args.base_axy is not None or args.base_aw is not None
N_DIR = 8  # base XY |a| <= Axy as a 16-gon: projections on 8 directions in [0, pi), both signs
BASE_DIRS = np.stack([np.cos(np.arange(N_DIR) * np.pi / N_DIR), np.sin(np.arange(N_DIR) * np.pi / N_DIR)], 1)
FPS = 30.0  # set per dataset in process(); workers get it via retime_episode's argument


# =========================== retime_ep.py core (unchanged logic) ===========================
def base_path(t, q):
    if args.smooth_lam <= 0:
        cs = CubicSpline(t, q)
        return lambda s, nu=0: cs(s, nu)
    sp = [make_smoothing_spline(t, q[:, j], lam=args.smooth_lam) for j in range(q.shape[1])]
    der = [[p, p.derivative(1), p.derivative(2)] for p in sp]
    return lambda s, nu=0: np.stack([d[nu](s) for d in der], -1)


def stretch_path(base, t, q, fa, fb):
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


def protected_windows(t, state, cs, fps):
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
    n_exp = int(round(args.max_expand * fps))
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


def base_vel_path(t, v):
    """Smoothed base velocity (vx, vy, wz) on the planning timeline t: f(s, nu) -> (..., 3), nu = 0 or 1."""
    sp = [make_smoothing_spline(t, v[:, c], lam=args.base_smooth_lam) for c in range(3)]
    der = [[p, p.derivative(1)] for p in sp]
    return lambda s, nu=0: np.stack([d[nu](s) for d in der], -1)


def constraint_cols(cs, bp, s):
    """Constraint columns c on grid s:  rate_c = d1[:, c] * s',  accel_c = d2[:, c] * s'^2 + d1[:, c] * s''.
    Columns 0-5: arm joints (Q', Q''), |rate| <= Vp, |accel| <= Ap.
    With base limits (bp = base_vel_path): the retimed base velocity is vb(s) * s', so its acceleration is
    vb'(s) s'^2 + vb(s) s'' -> the same form with d1 = vb, d2 = vb' and no velocity bound. XY uses
    N_DIR projections (a 2*N_DIR-gon circumscribing the circle, so the limit is shrunk by cos(pi/2N_DIR))."""
    d1, d2 = cs(s, 1), cs(s, 2)
    Vp, Ap = V * args.margin, A * args.margin
    if bp is None:
        return d1, d2, Vp, Ap
    vb, ab = bp(s, 0), bp(s, 1)
    c1, c2, lv, la = [d1], [d2], [Vp], [Ap]
    if args.base_axy is not None:
        c1.append(vb[:, :2] @ BASE_DIRS.T)
        c2.append(ab[:, :2] @ BASE_DIRS.T)
        lv.append(np.full(N_DIR, np.inf))
        la.append(np.full(N_DIR, args.base_axy * args.margin * np.cos(np.pi / (2 * N_DIR))))
    if args.base_aw is not None:
        c1.append(vb[:, 2:3])
        c2.append(ab[:, 2:3])
        lv.append([np.inf])
        la.append([args.base_aw * args.margin])
    return np.hstack(c1), np.hstack(c2), np.concatenate(lv), np.concatenate(la)


def topp_limits(s, d1, d2, Vp, Ap, cap, fix_start, fix_end):
    """Per-node bounds shared by topp() and jerk_lp(): s'^2 in [xlo, xhi], |accel| <= Alim (relaxed to the
    original acceleration at fixed protected-boundary nodes, where s' = 1)."""
    n = len(s)
    with np.errstate(divide="ignore"):
        xhi = np.minimum(cap ** 2, np.min((Vp / np.abs(d1)) ** 2, axis=1))
    xlo = np.zeros(n)
    Alim = np.tile(Ap, (n, 1))
    for k, fixed in ((0, fix_start), (n - 1, fix_end)):
        if fixed:
            xlo[k] = xhi[k] = 1.0
            Alim[k] = np.maximum(Ap, np.abs(d2[k]) + 1e-9)
    return xlo, xhi, Alim


def topp(s, d1, d2, Vp, Ap, cap, fix_start, fix_end):
    """Time-optimal path speed on grid s (x = s'^2 at nodes, u = s'' per segment) under
    |d1 s'| <= Vp, |d2 s'^2 + d1 s''| <= Ap, s' <= cap: backward controllable sets, greedy forward pass."""
    n = len(s)
    xlo, xhi, Alim = topp_limits(s, d1, d2, Vp, Ap, cap, fix_start, fix_end)
    fallback = 0

    def acc_rows(k, dl):
        g = np.concatenate([np.stack([d2[k], d1[k]], 1),
                            np.stack([d2[k + 1], 2 * dl * d2[k + 1] + d1[k + 1]], 1)])
        h = np.concatenate([Alim[k], Alim[k + 1]])
        keep = np.abs(g).max(1) > 1e-12  # 0*x + 0*u <= h (> 0) always holds, e.g. base standing still
        g, h = g[keep], h[keep]
        return np.vstack([g, -g]), np.concatenate([h, h])

    Klo, Khi = xlo.copy(), xhi.copy()
    for k in range(n - 2, -1, -1):
        dl = s[k + 1] - s[k]
        G, h = acc_rows(k, dl)
        G = np.vstack([G, [[1, 0], [-1, 0], [1, 2 * dl], [-1, -2 * dl]]])
        h = np.concatenate([h, [xhi[k], -xlo[k], Khi[k + 1], -Klo[k + 1]]])
        r = x_range(G, h)
        if r is None:
            fallback += 1
            continue
        Klo[k], Khi[k] = r

    x = np.empty(n)
    u = np.zeros(n - 1)
    x[0] = Khi[0]
    for k in range(n - 1):
        dl = s[k + 1] - s[k]
        lo, hi = (Klo[k + 1] - x[k]) / (2 * dl), (Khi[k + 1] - x[k]) / (2 * dl)
        G, h = acc_rows(k, dl)
        rhs, b = h - G[:, 0] * x[k], G[:, 1]
        pos, neg = b > 1e-12, b < -1e-12
        ulo = max(lo, (rhs[neg] / b[neg]).max(initial=-np.inf))
        uhi = min(hi, (rhs[pos] / b[pos]).min(initial=np.inf))
        if ulo > uhi + 1e-12:
            fallback += 1
            uhi = hi if uhi > hi else max(uhi, lo)
        u[k] = uhi
        x[k + 1] = max(x[k] + 2 * dl * u[k], 1e-10)
    return x, u, fallback


def jerk_lp(s, d1, d2, Vp, Ap, cap, fix_start, fix_end, x_ref):
    """Path speed with the arm jerk limit (--jmax-deg), as one sparse LP in x = s'^2 (s'' = dx/ds / 2).
    Constraints: s'^2 <= xhi, all acceleration columns (joints and base) at both ends of every segment as
    in topp(), and for the joints |a_k - a_(k-1)| <= Jmax * dt_(k-1), a_k = acceleration at the start of
    segment k. dt is taken from x_ref, the topp() solution: that is the pointwise fastest profile under
    the velocity/acceleration limits, so any LP solution is slower, its real dt is longer and the jerk
    rows are conservative. At a fixed (protected) end the original motion (s' = 1, s'' = 0, joint
    acceleration d2) is the neighbour, so the transitions in and out of protected windows are limited too.
    Objective: minimise the duration linearised at x_ref. Jerk rows carry a penalised slack, so the LP
    stays feasible where the original data already break a limit. Returns x, number of jerk rows over."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix
    n = len(s)
    xlo, xhi, Alim = topp_limits(s, d1, d2, Vp, Ap, cap, fix_start, fix_end)
    ds = np.diff(s)
    al = d2[:-1] - d1[:-1] / (2 * ds[:, None])  # a_k (start of segment k) = al_k x_k + be_k x_(k+1)
    be = d1[:-1] / (2 * ds[:, None])
    ga = -d1[1:] / (2 * ds[:, None])  # end of segment k = ga_k x_k + de_k x_(k+1)
    de = d2[1:] + d1[1:] / (2 * ds[:, None])
    rows, cols, vals, lb, ub = [], [], [], [], []
    r0 = 0

    def add(ci, vi, lo, hi):  # ci, vi: lists of (m,) arrays (same m) -> m rows
        nonlocal r0
        m = len(lo)
        for c, v in zip(ci, vi):
            rows.append(r0 + np.arange(m))
            cols.append(c)
            vals.append(v)
        lb.append(lo)
        ub.append(hi)
        r0 += m

    k = np.arange(n - 1)
    for c in range(d1.shape[1]):  # acceleration, both segment ends
        for cf0, cf1, lim in ((al[:, c], be[:, c], Alim[:-1, c]), (ga[:, c], de[:, c], Alim[1:, c])):
            m = (np.abs(cf0) > 1e-12) | (np.abs(cf1) > 1e-12)
            add([k[m], k[m] + 1], [cf0[m], cf1[m]], -lim[m], lim[m])
    nv = n
    Jp = JMAX * args.margin
    sx = np.sqrt(np.maximum(x_ref, 1e-10))
    dt = 2 * ds / (sx[:-1] + sx[1:])  # dt_k = duration of segment k in x_ref
    nj = 0
    for j in range(6):
        if n >= 3:  # a_k - a_(k-1), k = 1 .. n-2
            kk = np.arange(1, n - 1)
            band = Jp[j] * dt[kk - 1]
            sl = nv + nj + np.arange(len(kk))
            add([kk - 1, kk, kk + 1, sl], [-al[kk - 1, j], al[kk, j] - be[kk - 1, j], be[kk, j], -np.ones(len(kk))],
                np.full(len(kk), -np.inf), band)
            add([kk - 1, kk, kk + 1, sl], [-al[kk - 1, j], al[kk, j] - be[kk - 1, j], be[kk, j], np.ones(len(kk))],
                -band, np.full(len(kk), np.inf))
            nj += len(kk)
        for fixed, kk, a_orig, band in ((fix_start, 0, d2[0, j], Jp[j] * ds[0]),
                                         (fix_end, n - 2, d2[n - 1, j], Jp[j] * dt[n - 2])):
            if fixed:
                sl = np.array([nv + nj])
                add([np.array([kk]), np.array([kk + 1]), sl], [al[kk:kk + 1, j], be[kk:kk + 1, j], -np.ones(1)],
                    np.array([-np.inf]), np.array([a_orig + band]))
                add([np.array([kk]), np.array([kk + 1]), sl], [al[kk:kk + 1, j], be[kk:kk + 1, j], np.ones(1)],
                    np.array([a_orig - band]), np.array([np.inf]))
                nj += 1
    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(r0, nv + nj)).tocsr()
    # duration T = sum 2 ds / (sqrt x_k + sqrt x_k+1); dT/dx_k ~ -(ds_(k-1) + ds_k) / (4 x_k^1.5)
    dsn = np.r_[ds, 0] + np.r_[0, ds]
    w = dsn / (4 * np.maximum(x_ref, 1e-2) ** 1.5)
    cost = np.r_[-w / w.max(), np.full(nj, 1e3)]
    res = milp(cost, constraints=LinearConstraint(A, np.concatenate(lb), np.concatenate(ub)),
               bounds=Bounds(np.r_[xlo, np.zeros(nj)], np.r_[xhi, np.full(nj, np.inf)]))
    if res.x is None:
        return None, -1
    x = np.clip(res.x[:nv], np.maximum(xlo, 1e-10), xhi)
    slack = res.x[nv:]
    return x, int(np.sum(slack > 1e-6))


def stretch_duration(s, x):
    sd = np.sqrt(x)
    return np.sum(2 * np.diff(s) / (sd[:-1] + sd[1:]))


def retime_stretch(cs, bp, t, fa, fb, fix_start, fix_end, fps):
    s = np.concatenate([np.linspace(t[k], t[k + 1], args.subdiv, endpoint=False) for k in range(fa, fb)]
                       + [[t[fb]]])
    d1, d2, Vp, Ap = constraint_cols(cs, bp, s)
    jerk = JMAX is not None

    def plan(cap):
        x, u, fb_n = topp(s, d1, d2, Vp, Ap, cap, fix_start, fix_end)
        if not jerk:
            return x, u, fb_n, 0
        xj, nj = jerk_lp(s, d1, d2, Vp, Ap, cap, fix_start, fix_end, x)
        if xj is None:  # LP failed: keep the velocity/acceleration-only profile, count every row as over
            return x, u, fb_n, len(s)
        return xj, np.diff(xj) / (2 * np.diff(s)), fb_n, nj

    x, u, fb_n, jb_n = plan(args.frame_step)
    D = stretch_duration(s, x)
    M = max(1, int(np.ceil(D * fps - 1e-6)))
    target = M / fps
    lo, hi = 1e-3, args.frame_step
    Dlo, Dhi = np.inf, D
    for it in range(25):
        if abs(D - target) < (2e-5 if not jerk else 2e-4):
            break
        mid = 0.5 * (lo + hi)
        if jerk and np.isfinite(Dlo) and it % 3 != 2:  # regula falsi in 1/cap (D ~ linear), LP is costly
            y = 1 / hi + (target - Dhi) * (1 / lo - 1 / hi) / (Dlo - Dhi)
            if 1 / hi < y < 1 / lo:
                mid = 1 / y
        x, u, fb_n, jb_n = plan(mid)
        D = stretch_duration(s, x)
        if D > target:
            lo, Dlo = mid, D
        else:
            hi, Dhi = mid, D
    c = target / D
    return s, x, u, M, c, fb_n, jb_n


def sample_stretch(s, x, u, M, c, taus):
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
# ============================================================================================


# ----------------------- EE kinematics (URDF: SE01 Sculptor EVT, arm chain only) ---------------------
# elfin_joint1..6 origins / axes from "SE01-A0-000 Sculptor EVT_URDF_1010(simplified).SLDASM_cam30.urdf"
# (all rpy = 0). FK flange position equals observation.state flange_x/y/z (arm-base frame).
ARM_CHAIN = [((0, 0, 0.509107490819241), (0, 0, 1)),
             ((0.16, 0, 0.415892509180758), (0, 1, 0)),
             ((0, 0, 0.779997490819395), (0, 1, 0)),
             ((-0.150000000126904, 0, 0.780002034048233), (0, 0, 1)),
             ((0, 0, 0), (0, -1, 0)),
             ((8.30753166865296e-05, -6.80971368529359e-05, 0.237499807309987),
              (-6.06617424888621e-05, 0, 0.999999998160076))]


def _rot_batch(ax, th):
    ax = np.asarray(ax, float) / np.linalg.norm(ax)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    s, c = np.sin(th)[:, None, None], np.cos(th)[:, None, None]
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def ee_speeds(q, fps):
    """EE linear (m/s) and angular (rad/s) speed in the arm-base frame, per frame."""
    n = len(q)
    qd = np.gradient(q, 1.0 / fps, axis=0) if n > 1 else np.zeros_like(q)
    R = np.tile(np.eye(3), (n, 1, 1))
    p = np.zeros((n, 3))
    w = np.zeros((n, 3))
    for j, (o, ax) in enumerate(ARM_CHAIN):
        p = p + R @ np.asarray(o, float)
        z = R @ (np.asarray(ax, float) / np.linalg.norm(ax))
        w += z * qd[:, j:j + 1]
        R = R @ _rot_batch(ax, q[:, j])
    lin = np.linalg.norm(np.gradient(p, 1.0 / fps, axis=0), axis=1) if n > 1 else np.zeros(n)
    return lin, np.linalg.norm(w, axis=1)


def find_dwells(state, fps, protected):
    """Hover runs: EE linear < --dwell-lin AND EE angular < --dwell-ang AND base still, for at least
    --dwell-min s, outside protected windows. Returns kept original frame indices and the runs."""
    L = len(state)
    if args.no_dwell_cut or L < 10:
        return np.arange(L), []
    lin, ang = ee_speeds(state[:, POS], fps)
    lin, ang = uniform_filter1d(lin, 5), uniform_filter1d(ang, 5)
    base_still = (np.hypot(state[:, 7], state[:, 8]) < args.dwell_base_v) & (np.abs(state[:, 9]) < args.dwell_base_w)
    low = (lin < args.dwell_lin) & (ang < args.dwell_ang) & base_still & ~protected
    n_min = max(3, int(round(args.dwell_min * fps)))
    keep = np.ones(L, bool)
    runs, i = [], 0
    while i < L:
        if low[i]:
            j = i
            while j < L and low[j]:
                j += 1
            if j - i >= n_min:
                keep[i + 1:j - 1] = False  # keep the first and last hover frame
                runs.append((i, j - 1))
            i = j
        else:
            i += 1
    return np.where(keep)[0], runs


def retime_episode(state, action, fps, plot_path=None, title=""):
    """One episode -> retimed rows. state (L, 30), action (L, 10), original timeline i / fps.
    EE hovers are cut first (find_dwells), then the remaining frames are retimed as in retime_ep.py.
    plot_path: also save the retime_ep.py-style figure (speedup, v/Vmax, a/Amax) there."""
    L_full = len(state)
    t_full = np.arange(L_full) / fps
    if L_full < 10:  # too short to plan: copy
        return dict(state=state.astype(np.float32), action=action.astype(np.float32),
                    orig_frame=np.arange(L_full), src_t=t_full, sdot=np.ones(L_full),
                    protected=np.ones(L_full, bool), in_cut=np.zeros(L_full, bool),
                    report=dict(src_frames=L_full, frames=L_full, src_seconds=L_full / fps, seconds=L_full / fps,
                                speedup=1.0, note="too short, copied"))
    base_full = base_path(t_full, state[:, POS])
    events, windows_full = protected_windows(t_full, state, base_full, fps)
    prot_full = np.zeros(L_full, bool)
    for wa, wb in windows_full:
        prot_full[wa:wb + 1] = True
    kept, dwells = find_dwells(state, fps, prot_full)
    gap_after = np.r_[np.diff(kept) > 1, False]  # interval k -> k+1 (reduced) is a cut hover
    if dwells:
        state_r = state[kept]
        t = np.arange(len(kept)) / fps  # cut hovers are compressed to one frame interval
        base = base_path(t, state_r[:, POS])
        windows = [(int(np.searchsorted(kept, wa)), int(np.searchsorted(kept, wb))) for wa, wb in windows_full]
    else:
        state_r, t, base, windows = state, t_full, base_full, windows_full
    L = len(state_r)
    # base limits: smoothed base velocity paths (state for the constraint, action for the output action)
    bp = base_vel_path(t, state_r[:, BASE_VEL]) if BASE_ON else None
    bpa = base_vel_path(t, action[kept][:, ACTION_BASE_VEL]) if BASE_ON else None

    stretches, f = [], 0
    for wa, wb in windows:
        if wa > f:
            stretches.append(("move", f, wa))
        stretches.append(("prot", wa, wb))
        f = wb
    if f < L - 1:
        stretches.append(("move", f, L - 1))

    rows, src_t, sdots, kinds = [], [], [], []
    dense_v, dense_a, n_infeasible, n_jerk = [], [], 0, 0
    for si, (kind, fa, fb) in enumerate(stretches):
        has_prev, has_next = si > 0, si < len(stretches) - 1
        if kind == "prot":
            first = fa + 1 if has_prev and stretches[si - 1][0] == "prot" else fa
            for k in range(first, fb + 1):
                rows.append(("orig", k)); src_t.append(t[k]); sdots.append(1.0); kinds.append(1)
            continue
        fix_s = has_prev and stretches[si - 1][0] == "prot"
        fix_e = has_next and stretches[si + 1][0] == "prot"
        cs = stretch_path(base, t, state_r[:, POS], fa, fb)
        s, x, u, M, c, nfb, njb = retime_stretch(cs, bp, t, fa, fb, fix_s, fix_e, fps)
        n_infeasible += nfb
        n_jerk += njb
        ss, sdot, sdd, _ = sample_stretch(s, x, u, M, c, np.arange(M + 1) / fps)
        for m in range(1 if fix_s else 0, M if fix_e else M + 1):
            rows.append(("spline", ss[m], cs, sdd[m])); src_t.append(ss[m]); sdots.append(sdot[m]); kinds.append(0)
        td = np.linspace(0, M / fps, max(50, M * 20))
        s_d, sd_d, sdd_d, _ = sample_stretch(s, x, u, M, c, td)
        dense_v.append(np.abs(cs(s_d, 1) * sd_d[:, None]).max(0))
        dense_a.append(np.abs(cs(s_d, 2) * sd_d[:, None] ** 2 + cs(s_d, 1) * sdd_d[:, None]).max(0))

    N = len(rows)
    out = np.empty((N, state.shape[1]))
    orig_r = np.empty(N, dtype=np.int64)  # image frame, reduced index
    in_cut = np.zeros(N, bool)
    acc = np.empty((N, 6))
    sdots = np.asarray(sdots)
    for i, r in enumerate(rows):
        if r[0] == "orig":
            out[i] = state_r[r[1]]
            orig_r[i] = r[1]
            acc[i] = base(t[r[1]], 2)
            continue
        sv, cs = r[1], r[2]
        qv = cs(sv)
        k0 = int(np.clip(np.searchsorted(t, sv, side="right") - 1, 0, L - 2))
        if gap_after[k0]:  # inside a cut hover: image by joint progress k0 -> k0+1
            d = state_r[k0 + 1, POS] - state_r[k0, POS]
            prog = float(np.dot(qv - state_r[k0, POS], d) / max(np.dot(d, d), 1e-12))
            k = k0 + 1 if prog >= 0.5 else k0
            in_cut[i] = True
        else:
            k = k0 + 1 if (t[k0 + 1] - sv) < (sv - t[k0]) else k0
        orig_r[i] = k
        out[i] = state_r[k]
        out[i, POS] = qv
        out[i, VEL] = cs(sv, 1) * sdots[i]
        out[i, BASE_VEL] = (bp(sv) if BASE_ON else state_r[k, BASE_VEL]) * sdots[i]
        acc[i] = cs(sv, 2) * sdots[i] ** 2 + cs(sv, 1) * r[3]
    orig_r = np.maximum.accumulate(orig_r)  # images never go backwards
    orig = kept[orig_r]
    src_c = np.asarray(src_t)                        # planning time (hovers compressed)
    src_t = np.interp(src_c, t, t_full[kept])        # original time
    prot = np.asarray(kinds, dtype=bool)

    # action: joints = next row's joints (protected->protected original pairs keep the recorded
    # action); dims 6-9 = original action just before the next row's image frame
    a_idx = np.empty(N, dtype=np.int64)
    a_idx[:-1] = np.clip(orig[1:] - 1, orig[:-1], L_full - 1)
    a_idx[-1] = L_full - 1
    act = action[a_idx].astype(np.float64)
    act[:, ACTION_BASE_VEL] *= sdots[:, None]
    keep_rec = np.r_[prot[:-1] & prot[1:] & (np.diff(orig) == 1), prot[-1] and orig[-1] == L_full - 1]
    nxt = np.vstack([out[1:, POS], out[-1:, POS]])
    act[~keep_rec, :6] = nxt[~keep_rec]
    if BASE_ON:  # base action from the smoothed original command just before the next row, * s'
        ta = np.clip(np.r_[src_c[1:], src_c[-1]] - 1.0 / fps, 0.0, t[-1])
        act[~keep_rec, ACTION_BASE_VEL] = (bpa(ta) * sdots[:, None])[~keep_rec]

    fd_v = np.abs(np.diff(out[:, POS], axis=0)) * fps
    img_off = np.abs(src_c - t[orig_r])
    report = dict(
        src_frames=int(L_full), frames=int(N), src_seconds=L_full / fps, seconds=N / fps,
        speedup=float((L_full - 1) / max(N - 1, 1)), events=len(events), windows=len(windows),
        protected_rows=int(prot.sum()), infeasible_grid_points=int(n_infeasible),
        dwell_cuts=len(dwells), dwell_removed_frames=int(L_full - len(kept)),
        dwell_removed_seconds=float((L_full - len(kept)) / fps),
        retimed_peak_v_util=(np.max(dense_v, 0) / V).tolist() if dense_v else [0] * 6,
        retimed_peak_a_util=(np.max(dense_a, 0) / A).tolist() if dense_a else [0] * 6,
        output_fd_peak_v_util=(fd_v.max(0) / V).tolist(),
        **extra_limits_report(out, act, fps, n_jerk, prot),
        image_offset_max_s=float(img_off[~in_cut].max()) if (~in_cut).any() else 0.0,
        cut_rows=int(in_cut.sum()),
        cut_state_vs_image_max_deg=float(np.rad2deg(np.abs(out[in_cut, :6] - state[orig[in_cut], :6]).max()))
        if in_cut.any() else 0.0,
        endpoint_error_rad=float(np.abs(out[-1, POS] - state[-1, POS]).max()))
    if plot_path:
        cut_times = [i / fps for i in np.where(np.diff(orig) > 1)[0] if kept_gap(orig[i], orig[i + 1], kept)]
        plot_episode(plot_path, title, np.arange(N) / fps, sdots, out[:, VEL], acc, prot, L_full / fps, cut_times,
                     *extra_limits_series(out, fps))
    return dict(state=out.astype(np.float32), action=act.astype(np.float32), orig_frame=orig,
                src_t=src_t, sdot=sdots, protected=prot, in_cut=in_cut, report=report)


def extra_limits_series(out, fps):
    """Output rows at 30 Hz, finite differences: |jerk| / Jmax (N, 6) if --jmax-deg, base |a_xy| / Axy and
    |a_wz| / Aw (N, 2) if a base limit is set (an unset one reads 0). None when off."""
    dt = 1.0 / fps
    ju = bu = None
    if JMAX is not None and len(out) > 3:
        j = out[:, POS]
        for _ in range(3):
            j = np.gradient(j, dt, axis=0)
        ju = np.abs(j) / JMAX
    if BASE_ON and len(out) > 1:
        bu = base_acc_util(out[:, BASE_VEL], dt)
    return ju, bu


def base_acc_util(v, dt):
    a = np.gradient(v, dt, axis=0)
    return np.column_stack([np.hypot(a[:, 0], a[:, 1]) / (args.base_axy or np.inf),
                            np.abs(a[:, 2]) / (args.base_aw or np.inf)])


def extra_limits_report(out, act, fps, n_jerk, prot):
    """Report entries for the optional limits: peaks over all rows and over retimed (non-protected) rows;
    protected rows are original 1x data and can break the limits on their own."""
    ju, bu = extra_limits_series(out, fps)
    r = {}
    mv = ~prot if (~prot).any() else np.ones(len(prot), bool)
    if ju is not None:
        r.update(jerk_dead_ends=int(n_jerk), output_fd_peak_j_util=ju.max(0).tolist(),
                 output_fd_peak_j_util_retimed=ju[mv].max(0).tolist(),
                 output_fd_j_exceed_frac_retimed=(ju[mv] > 1).mean(0).tolist())
    if bu is not None:
        ba = base_acc_util(act[:, ACTION_BASE_VEL], 1.0 / fps) if len(act) > 1 else np.zeros((len(act), 2))
        r.update(output_fd_peak_base_acc_util=bu.max(0).tolist(),
                 output_fd_peak_base_acc_util_retimed=bu[mv].max(0).tolist(),
                 action_fd_peak_base_acc_util_retimed=ba[mv].max(0).tolist())
    return r


def kept_gap(a, b, kept):
    """True if original frames a -> b straddle a cut hover (frames between them were removed)."""
    ia, ib = np.searchsorted(kept, a), np.searchsorted(kept, b)
    return ib == ia + 1 and b - a > 1


def plot_episode(path, title, time_s, sdots, vel, acc, protected, src_seconds, cut_times=(),
                 jerk_util=None, base_util=None):
    """Same layout as retime_ep.py's retime.png (+ |jerk|/Jmax, base |acc|/limit rows when those limits are on)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N = len(time_s)
    nrow = 3 + (jerk_util is not None) + (base_util is not None)
    fig, ax = plt.subplots(nrow, 1, figsize=(16, 10 + 3 * (nrow - 3)), sharex=True)
    starts = np.where(np.diff(np.r_[0, protected.astype(int)]) == 1)[0]
    for a in ax:
        for i0 in starts:
            i1 = i0 + np.argmin(protected[i0:]) if not protected[i0:].all() else N - 1
            a.axvspan(time_s[i0], time_s[i1], color="orange", alpha=0.2, lw=0)
    for a in ax:
        for tc in cut_times:
            a.axvline(tc, color="#7a7974", lw=0.8, ls=":")
    ax[0].plot(time_s, sdots, "k", lw=1)
    ax[0].set_ylabel("speedup  ds/dτ")
    ax[0].set_title(f"{title}: {src_seconds:.1f}s -> {time_s[-1]:.1f}s "
                    f"(x{src_seconds / max(time_s[-1], 1e-9):.2f}), orange = protected (original rows), "
                    f"dotted = EE hover cut ({len(cut_times)})")
    for j in range(6):
        ax[1].plot(time_s, vel[:, j] / V[j], lw=0.8, label=f"J{j + 1}")
        ax[2].plot(time_s, acc[:, j] / A[j], lw=0.8, label=f"J{j + 1}")
    for a, lab in ((ax[1], "v / Vmax"), (ax[2], "a / Amax")):
        a.axhline(1, color="r", ls="--", lw=0.8)
        a.axhline(-1, color="r", ls="--", lw=0.8)
        a.set_ylabel(lab)
        a.legend(ncol=6, fontsize=8, loc="upper right")
    ax[2].set_ylim(-2, 2)
    r = 3
    if jerk_util is not None:
        for j in range(6):
            ax[r].plot(time_s, jerk_util[:, j], lw=0.8, label=f"J{j + 1}")
        ax[r].set_ylabel("|jerk| / Jmax (30 Hz fd)")
        ax[r].set_ylim(0, max(1.5, min(float(np.quantile(jerk_util, 0.999)) * 1.2, 5)))
        r += 1
    if base_util is not None:
        ax[r].plot(time_s, base_util[:, 0], lw=0.8, label="base |a_xy| / Axy")
        ax[r].plot(time_s, base_util[:, 1], lw=0.8, label="base |a_wz| / Aw")
        ax[r].set_ylabel("base |acc| / limit (30 Hz fd)")
        ax[r].set_ylim(0, max(1.5, min(float(base_util.max()) * 1.1, 5)))
    for a in ax[3:]:
        a.axhline(1, color="r", ls="--", lw=0.8)
        a.legend(ncol=6, fontsize=8, loc="upper right")
    ax[-1].set_xlabel("new time (s)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


# =========================== dataset I/O ===========================
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
    if isinstance(v, np.ndarray):
        return to_py(v.item()) if v.ndim == 0 else [to_py(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    return v


def encoder_args(kind):
    if kind == "nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "constqp", "-qp", str(args.qp)]
    return ["-c:v", "libx264", "-preset", "medium", "-qp", str(args.qp)]


def video_select_once(src, dst, frames, fps, L, w, h, kind):
    """Decode src as raw yuv420p, write frames[0], frames[1], ... (non-decreasing, repeats allowed).
    Returns None on success, else an error string."""
    fsz = w * h * 3 // 2
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", src, "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"],
                           stdout=subprocess.PIPE, bufsize=fsz * 8)
    encp = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p",
                             "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", *encoder_args(kind), "-g", "2", "-bf", "0",
                             "-pix_fmt", "yuv420p", "-an", dst], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    i = ptr = 0
    broken = False
    try:
        while True:
            buf = dec.stdout.read(fsz)
            if len(buf) < fsz:
                break
            while ptr < len(frames) and frames[ptr] == i:
                encp.stdin.write(buf)
                ptr += 1
            i += 1
    except BrokenPipeError:
        broken = True
    dec.kill() if broken else None
    dec.stdout.close()
    dec.wait()
    try:
        encp.stdin.close()
    except BrokenPipeError:
        broken = True
    err = encp.stderr.read().decode(errors="replace").strip()
    encp.wait()
    if broken or encp.returncode:
        return f"encoder ({kind}) failed: {err.splitlines()[-1] if err else 'broken pipe'}"
    if dec.returncode:
        return f"decoder failed for {src}"
    if i != L or ptr != len(frames):
        raise RuntimeError(f"{src}: decoded {i} frames (parquet {L}), wrote {ptr}/{len(frames)}")
    n = int(subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0",
                            "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", dst],
                           check=True, capture_output=True, text=True).stdout.strip())
    if n != len(frames):
        return f"{dst}: {n} frames, expected {len(frames)}"
    return None


def video_select(src, dst, frames, fps, L, w, h):
    """nvenc (retried: GeForce cards limit concurrent NVENC sessions, e.g. when two conversions
    run at once), then libx264 as fallback. Returns the encoder used."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    plan = (["nvenc"] * 4 + ["x264"]) if args.encoder == "nvenc" else ["x264"]
    err = None
    for attempt, kind in enumerate(plan):
        err = video_select_once(src, dst, frames, fps, L, w, h, kind)
        if err is None:
            if attempt:
                print(f"  [video] {os.path.basename(os.path.dirname(os.path.dirname(dst)))}/"
                      f"{os.path.basename(dst)} ok with {kind} after {attempt} retries", flush=True)
            return kind
        if kind == "nvenc":
            time.sleep(2 + 3 * attempt)
    raise RuntimeError(f"{src}: {err}")


def decode(path, frames, w, h):
    sel = "+".join(f"eq(n\\,{f})" for f in sorted(set(frames)))
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", f"select='{sel}'",
                          "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                         check=True, capture_output=True).stdout
    img = np.frombuffer(raw, np.uint8).reshape(-1, h, w).astype(np.float64)
    lut = {f: img[i] for i, f in enumerate(sorted(set(frames)))}
    return np.stack([lut[f] for f in frames])


def process(src):
    src = os.path.abspath(src.rstrip("/"))
    name = os.path.basename(src)
    final = os.path.join(args.out_root or (os.path.dirname(src) + "_retime"), f"{name}_retime")
    if os.path.exists(final):
        if not args.overwrite:
            print(f"[skip] {final} exists (use --overwrite)")
            return
        shutil.rmtree(final)
    dst = final + ".partial"  # renamed to final only when everything succeeded
    if os.path.exists(dst):
        shutil.rmtree(dst)
    t_start = time.time()
    info = json.load(open(os.path.join(src, "meta", "info.json")))
    assert info["codebase_version"].startswith("v3"), info["codebase_version"]
    fps = int(info["fps"])
    vkeys = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    h, w = info["features"][vkeys[0]]["shape"][:2]

    ep_files = sorted(glob.glob(os.path.join(src, "meta", "episodes", "*", "*.parquet")))
    eps = pa.concat_tables([pq.read_table(f) for f in ep_files]).to_pandas()
    eps = eps.sort_values("episode_index").reset_index(drop=True)
    data = pa.concat_tables([pq.read_table(f) for f in sorted(glob.glob(os.path.join(src, "data", "*", "*.parquet")))])
    ep_col = data["episode_index"].to_numpy()
    fi_col = data["frame_index"].to_numpy()
    task_col = data["task_index"].to_numpy()
    state_all = np.stack(data["observation.state"].to_numpy(zero_copy_only=False)).astype(np.float64)
    action_all = np.stack(data["action"].to_numpy(zero_copy_only=False)).astype(np.float64)

    ep_rows = []
    for _, e in eps.iterrows():
        ei, L = int(e["episode_index"]), int(e["length"])
        r = np.where(ep_col == ei)[0]
        r = r[np.argsort(fi_col[r])]
        assert len(r) == L and np.array_equal(fi_col[r], np.arange(L)), f"episode {ei} not contiguous"
        ep_rows.append((ei, L, r))

    print(f"[{name}] retiming {len(ep_rows)} episodes on {args.workers} workers ...", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        futs = [pool.submit(retime_episode, state_all[r], action_all[r], float(fps),
                            None if args.no_plot else os.path.join(dst, "plots", f"ep{ei:03d}_retime.png"),
                            f"{name} ep{ei}") for ei, _, r in ep_rows]
        res = []
        for i, f in enumerate(futs):
            res.append(f.result())
            if (i + 1) % 20 == 0 or i + 1 == len(futs):
                print(f"  [{name}] retimed {i + 1}/{len(futs)}", flush=True)

    ep_ranges, video_jobs, g = [], [], 0
    for (ei, L, r), rr, (_, e) in zip(ep_rows, res, eps.iterrows()):
        n = len(rr["orig_frame"])
        ep_ranges.append((ei, g, g + n, n))
        g += n
        for vk in vkeys:
            assert abs(float(e[f"videos/{vk}/from_timestamp"])) < 1e-6, "expects one episode per video file"
            rel = info["video_path"].format(video_key=vk, chunk_index=int(e[f"videos/{vk}/chunk_index"]),
                                            file_index=int(e[f"videos/{vk}/file_index"]))
            video_jobs.append(dict(src=os.path.join(src, rel), dst=os.path.join(dst, rel),
                                   frames=rr["orig_frame"], L=L, ep=ei))
    state = np.concatenate([rr["state"] for rr in res])
    action = np.concatenate([rr["action"] for rr in res])
    N = len(state)
    frame_index = np.concatenate([np.arange(n) for *_, n in ep_ranges]).astype(np.int64)
    timestamp = (frame_index / fps).astype(np.float32)
    episode_index = np.concatenate([np.full(n, ei) for ei, _, _, n in ep_ranges]).astype(np.int64)
    index = np.arange(N, dtype=np.int64)
    task_index = np.concatenate([task_col[r][rr["orig_frame"]] for (_, _, r), rr in zip(ep_rows, res)]).astype(np.int64)

    out_columns = {
        "observation.state": pa.array(list(state), type=data.schema.field("observation.state").type),
        "action": pa.array(list(action), type=data.schema.field("action").type),
        "timestamp": pa.array(timestamp, pa.float32()),
        "frame_index": frame_index, "episode_index": episode_index,
        "index": index, "task_index": task_index,
    }
    # Commander state is categorical: select the same source row as the image,
    # retaining its Arrow type rather than interpolating integer state codes.
    sampled_cols = {}
    if "observation.commander_state" in data.column_names:
        source_rows = np.concatenate([r[rr["orig_frame"]] for (_, _, r), rr in zip(ep_rows, res)])
        key = "observation.commander_state"
        out_columns[key] = data[key].take(pa.array(source_rows, type=pa.int64()))
        sampled_cols[key] = np.asarray(out_columns[key].to_pylist())
    out = pa.table(out_columns).select(data.column_names)
    data_rel = info["data_path"].format(chunk_index=0, file_index=0)
    os.makedirs(os.path.dirname(os.path.join(dst, data_rel)), exist_ok=True)
    pq.write_table(out, os.path.join(dst, data_rel))
    pq.write_table(pa.table(dict(
        episode_index=episode_index, frame_index=frame_index,
        source_time_s=np.concatenate([rr["src_t"] for rr in res]),
        original_frame_index=np.concatenate([rr["orig_frame"] for rr in res]),
        path_speed=np.concatenate([rr["sdot"] for rr in res]),
        protected=np.concatenate([rr["protected"] for rr in res]),
        in_cut=np.concatenate([rr["in_cut"] for rr in res]))), os.path.join(dst, "frame_map.parquet"))

    with ThreadPoolExecutor(args.jobs) as pool:
        futs = [pool.submit(video_select, j["src"], j["dst"], j["frames"], fps, j["L"], w, h) for j in video_jobs]
        used = []
        for i, f in enumerate(futs):
            used.append(f.result())
            if (i + 1) % 50 == 0 or i + 1 == len(futs):
                print(f"  [{name}] videos {i + 1}/{len(futs)}", flush=True)

    cols = {"observation.state": state, "action": action, "timestamp": timestamp,
            "frame_index": frame_index, "episode_index": episode_index, "index": index,
            "task_index": task_index, **sampled_cols}
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
                stat_key = f"stats/{key}/{sname}"
                if stat_key in E:  # v3 datasets may omit optional quantiles
                    E[stat_key][i] = to_py(v)
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
    info["total_frames"] = N
    for vk in vkeys:
        video_info = info["features"][vk]["info"]
        video_info["video.encoder"] = ("nvenc_constqp" if all(u == "nvenc" for u in used)
                                       else "libx264" if all(u == "x264" for u in used)
                                       else "mixed_nvenc_libx264_qp")
        video_info["video.qp"] = args.qp
        video_info["video.lossless"] = args.qp == 0 and all(u == "x264" for u in used)
        video_info["video.crf"] = None
        video_info["video.g"] = 2
        video_info["video.preset"] = ("medium" if all(u == "x264" for u in used)
                                     else "p4" if all(u == "nvenc" for u in used) else "mixed")
    json.dump(info, open(os.path.join(dst, "meta", "info.json"), "w"), indent=4, ensure_ascii=False)
    shutil.copy2(os.path.join(src, "meta", "tasks.parquet"), os.path.join(dst, "meta", "tasks.parquet"))
    if os.path.exists(os.path.join(src, "conversion_report.json")):
        shutil.copy2(os.path.join(src, "conversion_report.json"), os.path.join(dst, "conversion_report.json"))

    checks, seen = [], set()
    for j in video_jobs:
        if j["ep"] in seen or len(seen) >= args.verify:
            continue
        seen.add(j["ep"])
        fr, L = j["frames"], j["L"]
        ks = sorted({0, len(fr) // 3, len(fr) // 2, 2 * len(fr) // 3, len(fr) - 1})
        nv = decode(j["dst"], ks, w, h)
        o = decode(j["src"], [int(fr[k]) for k in ks], w, h)
        o2 = decode(j["src"], [min(int(fr[k]) + 1, L - 1) for k in ks], w, h)
        checks.append(dict(episode=j["ep"], video=os.path.relpath(j["dst"], dst), rows=ks,
                           mae_vs_mapped_orig=float(np.abs(o - nv).mean()),
                           mae_vs_next_orig=float(np.abs(o2 - nv).mean())))

    reps = [dict(episode=ei, **rr["report"]) for (ei, _, _), rr in zip(ep_rows, res)]
    pv = np.max([r["retimed_peak_v_util"] for r in reps if "retimed_peak_v_util" in r], 0)
    pa_ = np.max([r["retimed_peak_a_util"] for r in reps if "retimed_peak_a_util" in r], 0)
    extra = {}
    for key in ("output_fd_peak_j_util_retimed", "output_fd_peak_base_acc_util_retimed",
                "action_fd_peak_base_acc_util_retimed"):
        vals = [r[key] for r in reps if key in r]
        if vals:
            extra[key] = np.max(vals, 0).tolist()
    if JMAX is not None:
        extra["jerk_dead_ends"] = int(sum(r.get("jerk_dead_ends", 0) for r in reps))
    off = max(r.get("image_offset_max_s", 0) for r in reps)
    cut_n = sum(r.get("dwell_cuts", 0) for r in reps)
    cut_s = sum(r.get("dwell_removed_seconds", 0.0) for r in reps)
    cut_dev = max(r.get("cut_state_vs_image_max_deg", 0.0) for r in reps)
    src_sec, out_sec = len(state_all) / fps, N / fps
    report = dict(src=src, dst=dst, params=vars(args), fps=fps, src_frames=int(len(state_all)), frames=int(N),
                  src_seconds=src_sec, seconds=out_sec, speedup=src_sec / out_sec,
                  retimed_peak_v_util=pv.tolist(), retimed_peak_a_util=pa_.tolist(),
                  image_offset_max_s=off, dwell_cuts=cut_n, dwell_removed_seconds=cut_s,
                  cut_state_vs_image_max_deg=cut_dev, **extra, image_alignment_check=checks, episodes=reps,
                  seconds_compute=time.time() - t_start)
    report["dst"] = final
    json.dump(report, open(os.path.join(dst, "retime_report.json"), "w"), indent=2, ensure_ascii=False)
    os.rename(dst, final)
    dst = final
    print(f"[{name}] {src_sec:.0f}s -> {out_sec:.0f}s (x{src_sec / out_sec:.2f}), {len(state_all)} -> {N} frames, "
          f"{report['seconds_compute']:.0f}s -> {dst}")
    print("  retimed peak |v|/V:", np.round(pv, 3), "  |a|/A:", np.round(pa_, 3))
    if "output_fd_peak_j_util_retimed" in extra:
        print("  retimed rows peak |jerk|/Jmax (30 Hz fd):", np.round(extra["output_fd_peak_j_util_retimed"], 3),
              f"  jerk dead ends left: {extra['jerk_dead_ends']}")
    if "output_fd_peak_base_acc_util_retimed" in extra:
        print("  retimed rows peak base |a_xy|/Axy, |a_wz|/Aw (30 Hz fd): state",
              np.round(extra["output_fd_peak_base_acc_util_retimed"], 3),
              " action", np.round(extra["action_fd_peak_base_acc_util_retimed"], 3))
    print(f"  max image time offset {off * 1000:.1f} ms (outside hover cuts)")
    print(f"  EE hover cuts: {cut_n}, removed {cut_s:.1f}s of original time, "
          f"state-vs-image in cut rows max {cut_dev:.2f} deg")
    for c in checks:
        print(f"  verify ep{c['episode']} {c['video']}: MAE vs mapped orig {c['mae_vs_mapped_orig']:.2f}, "
              f"vs next orig {c['mae_vs_next_orig']:.2f}")


if __name__ == "__main__":
    for s in args.src:
        process(s)
