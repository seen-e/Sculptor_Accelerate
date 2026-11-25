# 机械臂轨迹加速（Retime）工具

把遥操作采集的 LeRobot v3.0 数据集"加速"：吸盘取/放附近的保护窗口保持原速，其余部分在
**关节速度 / 加速度限制**内最多加速 `--frame-step` 倍，输出仍为 30 Hz 的完整 LeRobot 数据集
（state / action / 视频全部重建并对齐）。

所有脚本都用 isaaclab 环境的 Python 运行：

```bash
PY=/home/seene/miniconda3/envs/isaaclab/bin/python
cd /home/seene/data_test2
```

## 脚本一览

| 脚本 | 作用 | 输入 |
| --- | --- | --- |
| [retime_lerobot.py](retime_lerobot.py) | **主脚本**：整个 LeRobot 数据集做时间最优加速（含 EE 悬停剪切） | LeRobot 数据集目录 |
| [subsample_lerobot.py](subsample_lerobot.py) | 只抽帧的加速（不插值，每帧都是原始帧） | LeRobot 数据集目录 |
| [retime_ep.py](retime_ep.py) | 单条 EVT 轨迹加速（算法原型，retime_lerobot.py 沿用其核心） | `EVT五条轨迹_.../01_修改前_原始state30/EPxxx/state30.npz` |
| [check_sync.py](check_sync.py) | 检查图像与关节状态的时间延迟，对比加速前后 | LeRobot 数据集目录 |
| [isaac_show_lerobot.py](isaac_show_lerobot.py) | 在 Isaac Sim 里回放某个 episode，旁边显示数据集视频帧 | LeRobot 数据集目录 |
| [isaac_show_ep.py](isaac_show_ep.py) | 在 Isaac Sim 里对比单条 EVT 轨迹加速前后 | EVT 五条轨迹目录 |

---

## 一、速度 / 加速度限制在哪里修改

### 方式 1：命令行参数（推荐，不用改代码）

```bash
$PY retime_lerobot.py --src openpi_sculptor_dataset/lerobot_20260918_fixed640_guard_12eps \
    --vmax-deg 63 36 90 108 162 162 \
    --amax-deg 120 60 90 96 192 192 \
    --margin 0.98 \
    --frame-step 3.0
```

| 参数 | 含义 | 默认值 |
| --- | --- | --- |
| `--vmax-deg` | 6 个关节（elfin_joint1~6）的**最大角速度**，单位 **deg/s**，必须给 6 个数 | `63 36 90 108 162 162` |
| `--amax-deg` | 6 个关节的**最大角加速度**，单位 **deg/s²**，必须给 6 个数 | `120 60 90 96 192 192` |
| `--margin` | 实际规划用 `margin × 限值`，留安全余量 | `0.98` |
| `--frame-step` | 保护窗口外的**最大加速倍数**（3 = 最多 3 倍速） | `3.0` |

> 速度/加速度限值只约束机械臂 6 个关节；保护窗口内的帧原样保留，不受限值约束。

### 方式 2：直接改代码里的默认值

各脚本中对应的 `argparse` 默认值位置：

- [retime_lerobot.py:61-64](retime_lerobot.py#L61-L64) — `--frame-step`、`--vmax-deg`、`--amax-deg`、`--margin`
- [subsample_lerobot.py:59-62](subsample_lerobot.py#L59-L62) — `--stride`（最大抽帧步长）、`--vmax-deg`、`--amax-deg`、`--margin`
- [retime_ep.py:38-42](retime_ep.py#L38-L42) — 同 retime_lerobot.py

代码内部会把 deg 转成 rad 使用（[retime_lerobot.py:89-90](retime_lerobot.py#L89-L90)：`V = np.deg2rad(args.vmax_deg)`，`A = np.deg2rad(args.amax_deg)`），一般不需要动这里。

### 其他相关参数（retime_lerobot.py）

| 参数 | 含义 | 默认值 |
| --- | --- | --- |
| `--guard-pre` / `--guard-post` | 吸盘事件前 / 后保护窗口长度（秒），窗口内保持 1 倍速 | `0.8` / `0.8` |
| `--pick-pressure` | 气压低于该值视为"取"事件 | `-30.0` |
| `--no-dwell-cut` | 关闭 EE 悬停剪切 | 默认开启剪切 |
| `--dwell-lin` / `--dwell-ang` | 末端线速度 < m/s 且角速度 < rad/s 视为悬停 | `0.01` / `0.1` |
| `--dwell-base-v` / `--dwell-base-w` | 底盘同时静止的阈值 | `0.02` / `0.05` |
| `--dwell-min` | 悬停至少持续多少秒才剪掉 | `0.3` |

---

## 二、数据路径怎么用

### 输入：`--src`

`--src` 指向一个（或多个）**LeRobot v3.0 数据集目录**，目录下应有 `meta/info.json`、`data/`、`videos/`：

```text
openpi_sculptor_dataset/
└── lerobot_20260918_fixed640_guard_12eps/
    ├── meta/        (info.json, episodes/, stats.json, tasks.parquet)
    ├── data/
    └── videos/
```

```bash
# 处理单个数据集
$PY retime_lerobot.py --src openpi_sculptor_dataset/lerobot_20260918_fixed640_guard_12eps

# 一次处理多个 / 全部数据集（shell 通配符展开成多个路径）
$PY retime_lerobot.py --src openpi_sculptor_dataset/lerobot_*

# 任意位置的数据集，用绝对路径即可
$PY retime_lerobot.py --src /path/to/my_lerobot_dataset
```

### 输出：`--out-root`

- **不指定**时，输出到 `<src 的上级目录>_retime/<数据集名>_retime/`，例如：
  `openpi_sculptor_dataset/lerobot_xxx` → `openpi_sculptor_dataset_retime/lerobot_xxx_retime/`
- **指定** `--out-root /some/dir` 时，输出到 `/some/dir/<数据集名>_retime/`
- 输出目录已存在会跳过，加 `--overwrite` 覆盖。处理过程中先写到 `*.partial`，全部成功后才改名。

输出内容：

```text
lerobot_xxx_retime/
├── meta/ data/ videos/     加速后的完整 LeRobot 数据集，可直接用于训练
├── frame_map.parquet       每个新帧对应的原始时间 source_time_s、原始帧号、速度倍率、是否保护帧
├── retime_report.json      每个 episode 的加速比、速度/加速度利用率、图像时间偏差
└── plots/epXXX_retime.png  每个 episode 的加速倍率、v/Vmax、a/Amax 曲线（--no-plot 跳过）
```

### 运行性能相关参数

| 参数 | 含义 | 默认值 |
| --- | --- | --- |
| `--workers` | 并行处理 episode 的进程数 | `16` |
| `--jobs` | 并行视频重编码数 | `6` |
| `--encoder` | `nvenc`（需 NVIDIA GPU）或 `x264`（CPU） | `nvenc` |
| `--qp` | 视频编码质量，越小越好 | `18` |
| `--verify` | 每个数据集抽查多少个 episode 做像素校验，0 关闭 | `2` |

### 其他脚本的路径

- **subsample_lerobot.py**：与 retime_lerobot.py 相同，用 `--src` / `--out-root`，默认输出到 `<src 上级>_<mode>_s<stride>/<数据集名>_<mode>_s<stride>/`（如 `openpi_sculptor_dataset_variable_s3/`）。
- **retime_ep.py**：输入路径写死在 [retime_ep.py:30](retime_ep.py#L30)（`SRC = EVT五条轨迹_state修改前后_20261106/01_修改前_原始state30`），用 `--ep 35` 选轨迹，`--out` 指定输出目录（默认 `modified_v1/EPxxx/`）。换数据源需改第 30 行。
- **check_sync.py / isaac_show_lerobot.py**：用 `--dataset` 指定数据集目录，例如：

```bash
ORIG=openpi_sculptor_dataset/lerobot_20260918_fixed640_guard_12eps
RT=openpi_sculptor_dataset_retime/lerobot_20260918_fixed640_guard_12eps_retime

# 对比加速前后图像-关节延迟
$PY check_sync.py --dataset $ORIG $RT --eps 0 1 2 3

# Isaac Sim 中按原始时间对齐回放，检查轨迹是否一致
$PY isaac_show_lerobot.py --dataset $RT --ref $ORIG --ep 0 --sync source --start 30 --end 60
```

> 注意：isaac_show_*.py 中的机器人模型路径 `ROBOT_USD` 写死为 `/home/seene/Tele_Isaac/...`，换机器时需要修改。

---

## 三、常用示例

```bash
# 默认参数加速全部数据集
$PY retime_lerobot.py --src openpi_sculptor_dataset/lerobot_*

# 更保守：最多 2 倍速，速度/加速度限值留 10% 余量，保护窗口加长到 1 秒
$PY retime_lerobot.py --src openpi_sculptor_dataset/lerobot_20260918_fixed640_guard_12eps \
    --frame-step 2 --margin 0.9 --guard-pre 1.0 --guard-post 1.0 \
    --out-root /data/retime_conservative --overwrite

# 没有 NVIDIA GPU 时用 CPU 编码
$PY retime_lerobot.py --src ... --encoder x264
```
