"""Generate an interactive HTML report (Plotly) for a LeRobot dataset:
per-episode joint / pressure / suction plots + suction pick & place timing tables.

Usage:
  PY=/home/seene/miniconda3/envs/isaaclab/bin/python
  $PY make_suction_report.py <dataset_dir> [-o out.html] [--pick-pressure -30]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from suction_events import DEFAULT_PLOTLY, detect_cycles, r3, spans, write_html

ap = argparse.ArgumentParser()
ap.add_argument("dataset")
ap.add_argument("-o", "--out", default=None)
ap.add_argument("--pick-pressure", type=float, default=-30.0,
                help="pressure below this = box attached (same default as retime_lerobot.py)")
ap.add_argument("--weak-pressure", type=float, default=-22.0,
                help="fallback threshold for weak grips that never go below --pick-pressure (idle vacuum ~ -16)")
ap.add_argument("--plotly-js", default=DEFAULT_PLOTLY,
                help="local plotly.min.js to inline (offline HTML); falls back to the CDN if missing")
args = ap.parse_args()

root = Path(args.dataset)
info = json.loads((root / "meta/info.json").read_text())
fps = info["fps"]
SN = info["features"]["observation.state"]["names"]
AN = info["features"]["action"]["names"]
J = [SN.index(f"elfin_joint{i}") for i in range(1, 7)]
JV = [SN.index(f"elfin_joint{i}_velocity") for i in range(1, 7)]
SUC_S, PRES = SN.index("suction_do_ch3"), SN.index("fluid_pressure")
FL = [SN.index(n) for n in ("flange_x", "flange_y", "flange_z")]
BASE = [SN.index(n) for n in ("base_vx", "base_vy", "base_wz")]
SUC_A = AN.index("suction_on")

df = pd.concat([pd.read_parquet(p) for p in sorted((root / "data").rglob("*.parquet"))])
fm_path = root / "frame_map.parquet"
fm = pd.read_parquet(fm_path) if fm_path.exists() else None
rr_path = root / "retime_report.json"
rr = json.loads(rr_path.read_text()) if rr_path.exists() else None
rr_eps = {e["episode"]: e for e in rr["episodes"]} if rr else {}
VMAX = np.radians(rr["params"]["vmax_deg"]) if rr else None
AMAX = np.radians(rr["params"]["amax_deg"]) if rr else None


episodes = []
for ep, e in df.groupby("episode_index"):
    e = e.sort_values("frame_index")
    s = np.stack(e["observation.state"].values).astype(float)
    a = np.stack(e["action"].values).astype(float)
    n = len(e)
    t = np.arange(n) / fps
    p = s[:, PRES]
    src_t = None
    prot = []
    rate = None
    sdot, cut_t = None, []
    if fm is not None:
        f = fm[fm.episode_index == ep].sort_values("frame_index")
        if len(f) == n:
            src_t = f["source_time_s"].to_numpy()
            prot = [(int(i), int(j)) for i, j in spans(f["protected"].to_numpy())]
            # local speed-up factor = d(source time)/d(output time), 0.5 s moving average
            k = max(int(fps * 0.5), 1)
            ds = np.gradient(src_t, t)
            rate = np.convolve(ds, np.ones(k) / k, mode="same")
            sdot = f["path_speed"].to_numpy()
            cut = f["in_cut"].to_numpy().astype(bool)
            cut_t = t[np.where(np.diff(cut.astype(int)) != 0)[0] + 1].tolist() if cut.any() else []
    ev = detect_cycles(t, p, a[:, SUC_A], s[:, SUC_S], fps, args.pick_pressure, args.weak_pressure, src_t=src_t)
    cycles, orphan_picks = ev["cycles"], ev["orphan_picks"]
    episodes.append(dict(
        ep=int(ep), n=n, dur=round(n / fps, 3),
        src_dur=None if src_t is None else round(float(src_t[-1]), 3),
        speedup=rr_eps.get(int(ep), {}).get("speedup"),
        t=r3(t),
        src_t=None if src_t is None else r3(src_t),
        joints=[r3(np.degrees(s[:, j])) for j in J],
        sdot=None if sdot is None else r3(sdot), cut_t=cut_t,
        vutil=None if VMAX is None else [r3(s[:, j] / VMAX[i]) for i, j in enumerate(JV)],
        autil=None if AMAX is None else [r3(np.gradient(s[:, j], 1 / fps) / AMAX[i]) for i, j in enumerate(JV)],
        pres=r3(p), suc_a=a[:, SUC_A].astype(int).tolist(), suc_s=s[:, SUC_S].astype(int).tolist(),
        flange=[r3(s[:, j]) for j in FL], base=[r3(s[:, j]) for j in BASE],
        prot=prot, **ev,
    ))
    print(f"ep{ep:03d}: {n} frames, {len(cycles)} place, missing pick {sum(c['pick'] is None for c in cycles)}, orphan holds {orphan_picks}, hold s {[c['hold_s'] for c in cycles]}")

meta = dict(
    name=root.name, path=str(root), fps=fps, pick_pressure=args.pick_pressure, weak_pressure=args.weak_pressure,
    task=pd.read_parquet(root / "meta/tasks.parquet").index.tolist(),
    params=rr["params"] if rr else None,
    total=dict(src_s=rr["src_seconds"], s=rr["seconds"], speedup=rr["speedup"]) if rr else None,
)
write_html(meta, episodes, args.out or root.parent / f"{root.name}_吸盘时间交互报告.html", args.plotly_js)
