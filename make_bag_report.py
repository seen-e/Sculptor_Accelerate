"""Interactive suction pick / place HTML report straight from a raw ROS2 mcap bag (no conversion needed).

Uses only standard message types (custom sculptor_* types are not needed):
  /joint_states                          elfin_joint1..6 position / velocity
  /xp_fluid/pressure_raw                 vacuum pressure
  /io_controller/do/cmd/ch3              suction command   (like LeRobot action.suction_on)
  /io_controller/do/status/ch3           suction DO status (like LeRobot state.suction_do_ch3)
  /sculptor/teleop/state/arm_flange_pose flange xyz
  /chassis/body_twist                    base vx / vy / wz
Everything is resampled onto a uniform --fps grid (sample-and-hold, like the LeRobot conversion).

Usage (ROS python):
  source /opt/ros/jazzy/setup.bash
  python3 make_bag_report.py bag_261115_194539 [more bags ...] [-o out.html]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from suction_events import DEFAULT_PLOTLY, detect_cycles, idle_baseline, r3, write_html

ap = argparse.ArgumentParser()
ap.add_argument("bags", nargs="+")
ap.add_argument("-o", "--out", default=None)
ap.add_argument("--fps", type=float, default=30.0)
ap.add_argument("--pick-pressure", default="auto",
                help="'auto' = idle baseline + --pick-offset (this robot idles at ~-33, the LeRobot data at ~-16)")
ap.add_argument("--pick-offset", type=float, default=-10.0)
ap.add_argument("--weak-offset", type=float, default=-5.0)
ap.add_argument("--vmax-deg", type=float, nargs=6, default=[63, 36, 90, 108, 162, 162],
                help="joint limits for v/Vmax (default = retime1.2_guard0.1 run)")
ap.add_argument("--amax-deg", type=float, nargs=6, default=[144.0, 72.0, 108.0, 115.2, 230.4, 230.4])
ap.add_argument("--plotly-js", default=DEFAULT_PLOTLY)
args = ap.parse_args()

TOPICS = {
    "joint": "/joint_states", "pres": "/xp_fluid/pressure_raw",
    "cmd": "/io_controller/do/cmd/ch3", "status": "/io_controller/do/status/ch3",
    "flange": "/sculptor/teleop/state/arm_flange_pose", "twist": "/chassis/body_twist",
}
JN = [f"elfin_joint{i}" for i in range(1, 7)]
VMAX, AMAX = np.radians(args.vmax_deg), np.radians(args.amax_deg)


def read_bag(path):
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=str(path), storage_id="mcap"), rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    want = [v for v in TOPICS.values() if v in types]
    r.set_filter(rosbag2_py.StorageFilter(topics=want))
    cls = {k: get_message(types[k]) for k in want}
    log = {k: ([], []) for k in TOPICS}
    inv = {v: k for k, v in TOPICS.items()}
    while r.has_next():
        tp, data, t = r.read_next()          # bag receive time (ns)
        m = deserialize_message(data, cls[tp])
        k = inv[tp]
        if k == "joint":
            idx = [m.name.index(n) for n in JN]
            v = list(m.velocity) if len(m.velocity) else [np.nan] * len(m.name)
            val = [m.position[i] for i in idx] + [v[i] for i in idx]
        elif k == "pres":
            val = m.fluid_pressure
        elif k in ("cmd", "status"):
            val = float(m.data)
        elif k == "flange":
            q = m.pose.position; val = [q.x, q.y, q.z]
        else:
            tw = m.twist; val = [tw.linear.x, tw.linear.y, tw.angular.z]
        log[k][0].append(t); log[k][1].append(val)
    return {k: (np.array(ts, dtype=np.int64), np.array(vs, dtype=float)) for k, (ts, vs) in log.items() if ts}


def hold(ts, vs, grid):
    """sample-and-hold onto grid (first value back-filled)."""
    i = np.clip(np.searchsorted(ts, grid, side="right") - 1, 0, len(ts) - 1)
    return vs[i]


episodes, metas = [], []
for bi, bag in enumerate(args.bags):
    bag = Path(bag)
    L = read_bag(bag)
    t0 = min(v[0][0] for v in L.values())
    t1 = max(v[0][-1] for v in L.values())
    n = int((t1 - t0) / 1e9 * args.fps) + 1
    grid = t0 + (np.arange(n) / args.fps * 1e9).astype(np.int64)
    t = np.arange(n) / args.fps
    js = hold(*L["joint"], grid)
    pos, vel = js[:, :6], js[:, 6:]
    if np.isnan(vel).all():
        vel = np.gradient(pos, 1 / args.fps, axis=0)
    p = hold(*L["pres"], grid)
    cmd = hold(*L["cmd"], grid).astype(int)
    status = hold(*L["status"], grid).astype(int) if "status" in L else cmd
    flange = hold(*L["flange"], grid) if "flange" in L else np.zeros((n, 3))
    base = hold(*L["twist"], grid) if "twist" in L else np.zeros((n, 3))

    base_p = idle_baseline(p, cmd)
    pick = float(args.pick_pressure) if args.pick_pressure != "auto" else base_p + args.pick_offset
    weak = pick - args.pick_offset + args.weak_offset if args.pick_pressure != "auto" else base_p + args.weak_offset
    ev = detect_cycles(t, p, cmd, status, args.fps, pick, weak)
    sm = bag / "session_meta.json"
    task = json.loads(sm.read_text()).get("case_description", "") if sm.exists() else ""
    metas.append(dict(base=base_p, pick=pick, weak=weak, task=task))
    episodes.append(dict(
        ep=bi, label=bag.name, n=n, dur=round(n / args.fps, 3), src_dur=None, speedup=None,
        t=r3(t), src_t=None,
        joints=[r3(np.degrees(pos[:, j])) for j in range(6)],
        sdot=None, cut_t=[], prot=[],
        vutil=[r3(vel[:, j] / VMAX[j]) for j in range(6)],
        autil=[r3(np.gradient(vel[:, j], 1 / args.fps) / AMAX[j]) for j in range(6)],
        pres=r3(p), suc_a=cmd.tolist(), suc_s=status.tolist(),
        flange=[r3(flange[:, j]) for j in range(3)], base=[r3(base[:, j]) for j in range(3)],
        **ev,
    ))
    print(f"{bag.name}: {n / args.fps:.1f}s, idle {base_p:.0f} -> pick < {pick:.0f} (weak < {weak:.0f}), "
          f"{len(ev['cycles'])} place, hold s {[c['hold_s'] for c in ev['cycles']]}, orphan {ev['orphan_picks']}")

m0 = metas[0]
thr = "、".join(f"{m['pick']:.0f}" for m in metas)
meta = dict(
    title="吸盘吸附 / 放置时间交互报告（原始 bag）",
    name=", ".join(Path(b).name for b in args.bags), path=", ".join(str(Path(b).resolve()) for b in args.bags),
    fps=args.fps, pick_pressure=m0["pick"], weak_pressure=m0["weak"], task=[m0["task"]] if m0["task"] else [],
    xlabel="bag 时间 (s)（从录制开始算，原始速度，未加速）",
    vutil_title="v / Vmax（/joint_states 速度，限位取 retime1.2 参数）",
    autil_title="a / Amax（由速度差分）",
    base_title="底盘速度 /chassis/body_twist",
    signal_names=["/io_controller/do/cmd/ch3（吸盘指令）", "/io_controller/do/status/ch3（×0.9 错开）"],
    extra_tiles=[["空吸基线气压", f"{m0['base']:.0f}", ""], ["吸附判定阈值", f"< {m0['pick']:.0f}", ""]],
    rules=[
        f"<b>吸附时刻（pick）</b>：气压 <code>/xp_fluid/pressure_raw</code> 降到 <b>{thr}</b> 以下。这台机器空吸基线约 "
        f"<b>{m0['base']:.0f}</b>（LeRobot 数据集里是 -16），所以阈值按“基线 {args.pick_offset:+.0f}”自动设定，不能直接用 -30；"
        f"气压停在阈值附近的弱吸附用“基线 {args.weak_offset:+.0f}”兜底。吸住箱子时约 -45 ~ -81。",
        "<b>放置时刻（place）</b>：吸盘指令 <code>/io_controller/do/cmd/ch3</code> 从 1 变 0；<b>放置结束</b> = 恢复为 1。"
        "<code>/io_controller/do/status/ch3</code> 一般滞后约 0.07~0.1 s。",
        "<b>吸附持续</b> = 放置时刻 − 吸附时刻；<b>释放脉冲</b> = 放置结束 − 放置时刻；<b>泄压用时</b> = 放置时刻 → 气压回升到 &gt; -5。",
        f"所有话题按 bag 接收时间对齐，以 {args.fps:.0f} Hz 采样保持重采样（和 LeRobot 转换一致）。气压传感器约 17 Hz，"
        "所以时刻精度约 ±0.06 s。这是原始未加速数据，没有 retime 的 speedup / 保护窗口两栏。",
    ],
)
out = args.out or Path(args.bags[0]).resolve().parent / f"{Path(args.bags[0]).name}_吸盘时间交互报告.html"
write_html(meta, episodes, out, args.plotly_js)
