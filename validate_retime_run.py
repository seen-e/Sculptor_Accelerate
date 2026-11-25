"""Audit one retiming experiment and produce a self-contained results folder."""
import html
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


def read_table(root, folder):
    import pyarrow as pa
    return pa.concat_tables([pq.read_table(p) for p in sorted((root / folder).rglob('*.parquet'))])


def fd_stats(q, fps, vmax, amax, protected=None):
    v = np.diff(q, axis=0) * fps
    a = np.diff(q, n=2, axis=0) * fps ** 2
    result = {}
    for name, values, limit, order in [('velocity', v, vmax, 1), ('acceleration', a, amax, 2)]:
        util = np.abs(values) / limit
        masks = {'all': np.ones(len(util), bool)}
        if protected is not None:
            stencils = np.stack([protected[k:k + len(util)] for k in range(order + 1)])
            masks.update(protected=stencils.all(0), retimed=(~stencils).all(0),
                         boundary=stencils.any(0) & ~stencils.all(0))
        result[name] = {key: dict(peak=util[m].max(0).tolist(),
                                     exceed_samples=int((util[m] > 1.0001).any(1).sum()),
                                     samples=int(m.sum()))
                        for key, m in masks.items() if m.any()}
    return result


def probe_video(job):
    path, frames = job
    p = subprocess.run(['ffprobe', '-v', 'error', '-threads', '1', '-count_frames',
                        '-select_streams', 'v:0', '-show_entries',
                        'stream=nb_read_frames,width,height,avg_frame_rate', '-of', 'json', str(path)],
                       check=True, capture_output=True, text=True)
    assert not p.stderr.strip(), (str(path), p.stderr)
    stream = json.loads(p.stdout)['streams'][0]
    assert int(stream['nb_read_frames']) == frames, (path, frames, stream)
    assert (stream['width'], stream['height'], stream['avg_frame_rate']) == (256, 256, '30/1')
    return str(path)


def comparison(src, dst, info, source_ep, out_ep, output):
    keys = ['observation.images.surround', 'observation.images.cam9']
    inputs = []
    for base, ep in [(src, source_ep), (dst, out_ep)]:
        for key in keys:
            path = base / info['video_path'].format(video_key=key,
                       chunk_index=int(ep[f'videos/{key}/chunk_index']),
                       file_index=int(ep[f'videos/{key}/file_index']))
            inputs.extend(['-i', str(path)])
    original_s, retimed_s = source_ep['length'] / 30, out_ep['length'] / 30
    filters = []
    for i in range(4):
        filters.append(f'[{i}:v]setpts=PTS-STARTPTS,scale=400:400,setsar=1[v{i}]')
    for side, start, duration in [('left', 0, original_s), ('right', 2, retimed_s)]:
        name = 'Original' if side == 'left' else 'Retimed'
        filters.append(f'[v{start}][v{start+1}]vstack=inputs=2,pad=400:864:0:64:color=0x142030,'
                       f"drawtext=text='{name}  {duration:.2f}s':x=14:y=10:fontsize=23:fontcolor=white,"
                       f"drawtext=text='top surround / bottom cam9':x=14:y=40:fontsize=16:fontcolor=white,"
                       f'tpad=stop_mode=clone:stop_duration={max(0, original_s-duration)+1}[{side}]')
    filters.append(f'[left][right]hstack=inputs=2,'
                   f"drawtext=text='RETIME FINISHED':x=420:y=100:fontsize=26:fontcolor=yellow:"
                   f"box=1:boxcolor=black@0.7:enable='gte(t,{retimed_s:.6f})'[out]")
    script = output / 'comparison_filter.txt'
    script.write_text(';\n'.join(filters))
    video = output / 'comparison.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-filter_complex_threads', '1', *inputs,
                    '-filter_complex_script', str(script), '-map', '[out]', '-frames:v',
                    str(source_ep['length']), '-r', '30', '-an', '-c:v', 'libx264', '-threads', '2',
                    '-crf', '22', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video)], check=True)
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', str(retimed_s / 2), '-i', str(video),
                    '-frames:v', '1', str(output / 'comparison_preview.png')], check=True)
    return dict(episode=int(source_ep['episode_index']), source_seconds=original_s,
                retimed_seconds=retimed_s, speedup=original_s / retimed_s)


def main():
    run = Path(sys.argv[1])
    launch = json.loads((run / 'launch.json').read_text())
    src = Path(launch['source'])
    dst = next(p for p in (run / 'output').iterdir() if (p / 'retime_report.json').exists())
    output = run / 'results'
    output.mkdir(exist_ok=True)
    report = json.loads((dst / 'retime_report.json').read_text())
    info = json.loads((dst / 'meta/info.json').read_text())
    original, retimed = read_table(src, 'data'), read_table(dst, 'data')
    eps_src = read_table(src, 'meta/episodes').to_pandas().set_index('episode_index')
    eps_out = read_table(dst, 'meta/episodes').to_pandas().set_index('episode_index')
    fm = pq.read_table(dst / 'frame_map.parquet').to_pandas()
    odf, rdf = original.to_pandas(), retimed.to_pandas()
    assert original.schema.equals(retimed.schema, check_metadata=False)
    assert len(rdf) == info['total_frames'] == len(fm)
    assert len(eps_out) == info['total_episodes'] == 34
    vmax, amax = np.deg2rad(report['params']['vmax_deg']), np.deg2rad(report['params']['amax_deg'])
    audit = []
    video_jobs = []
    for ei, e in eps_out.iterrows():
        a = odf[odf.episode_index == ei].sort_values('frame_index')
        b = rdf[rdf.episode_index == ei].sort_values('frame_index')
        f = fm[fm.episode_index == ei].sort_values('frame_index')
        n = len(b)
        indices = f.original_frame_index.to_numpy()
        np.testing.assert_array_equal(b.frame_index, np.arange(n))
        np.testing.assert_allclose(b.timestamp, np.arange(n) / info['fps'], atol=2e-6)
        np.testing.assert_array_equal(b.episode_index, f.episode_index)
        np.testing.assert_array_equal(b.frame_index, f.frame_index)
        assert n == e['length']
        assert np.all(np.diff(indices) >= 0) and indices.min() >= 0 and indices.max() < len(a)
        assert np.all(np.diff(f.source_time_s) >= -1e-8)
        np.testing.assert_array_equal(b['observation.commander_state'], a['observation.commander_state'].to_numpy()[indices])
        assert e['stats/observation.commander_state/count'].tolist() == [n]
        commander = b['observation.commander_state'].to_numpy()
        for stat, value in [('min', commander.min()), ('max', commander.max()),
                            ('mean', commander.mean()), ('std', commander.std())]:
            np.testing.assert_allclose(e[f'stats/observation.commander_state/{stat}'], [value])
        state_a, state_b = np.stack(a['observation.state']), np.stack(b['observation.state'])
        action_b = np.stack(b['action'])
        assert state_b.shape == (n, 30) and action_b.shape == (n, 10)
        assert np.isfinite(state_b).all() and np.isfinite(action_b).all()
        prot = f.protected.to_numpy()
        np.testing.assert_array_equal(state_b[prot], state_a[indices[prot]])
        np.testing.assert_allclose(state_b[[0, -1], :6], state_a[[0, -1], :6], atol=1e-6)
        for vk, feature in info['features'].items():
            if feature['dtype'] != 'video':
                continue
            assert abs(e[f'videos/{vk}/to_timestamp'] - n / info['fps']) < 1e-6
            path = dst / info['video_path'].format(video_key=vk,
                       chunk_index=int(e[f'videos/{vk}/chunk_index']),
                       file_index=int(e[f'videos/{vk}/file_index']))
            video_jobs.append((path, n))
        audit.append(dict(episode=int(ei), source=fd_stats(state_a[:, :6].astype(float), 30, vmax, amax),
                          output=fd_stats(state_b[:, :6].astype(float), 30, vmax, amax, prot)))
    np.testing.assert_array_equal(rdf['index'], np.arange(len(rdf)))
    stats = json.loads((dst / 'meta/stats.json').read_text())
    assert stats['observation.commander_state']['count'] == [len(rdf)]
    print('Parquet, mapping, protected rows, endpoints and metadata passed', flush=True)
    with ThreadPoolExecutor(3) as pool:
        videos = list(pool.map(probe_video, video_jobs))
    print(f'Fully decoded and counted {len(videos)} output videos', flush=True)
    reps = report['episodes']
    bad = [r['episode'] for r in reps if max(r['retimed_peak_a_util']) > 1.0001 or max(r['retimed_peak_v_util']) > 1.0001]
    infeasible = [r['episode'] for r in reps if r['infeasible_grid_points']]
    finite_difference_bad = [r['episode'] for r in audit if r['output']['acceleration']['all']['exceed_samples']]
    fd_groups = {}
    for origin in ('source', 'output'):
        for group in ('all', 'retimed', 'protected', 'boundary'):
            values = [r[origin]['acceleration'][group] for r in audit if group in r[origin]['acceleration']]
            if values:
                fd_groups[f'{origin}/{group}'] = dict(
                    episodes_exceed=sum(v['exceed_samples'] > 0 for v in values),
                    peak=max(max(v['peak']) for v in values),
                    exceed_samples=sum(v['exceed_samples'] for v in values),
                    samples=sum(v['samples'] for v in values))
    med = np.median([r['speedup'] for r in reps])
    representative = min((r for r in reps if r['episode'] not in bad), key=lambda r: abs(r['speedup'] - med))['episode']
    source_ep = dict(eps_src.loc[representative], episode_index=representative)
    out_ep = dict(eps_out.loc[representative], episode_index=representative)
    example = comparison(src, dst, info, source_ep, out_ep, output)
    summary = dict(status='LIMIT_REVIEW_REQUIRED', source=str(src), output=str(dst),
                   structural_validation_passed=True, videos_decoded=len(videos),
                   commander_mapping_passed=True, episodes=34, src_frames=len(odf), frames=len(rdf),
                   source_seconds=report['src_seconds'], seconds=report['seconds'], speedup=report['speedup'],
                   time_saved_percent=100*(1-report['seconds']/report['src_seconds']),
                   dense_limit_exceeded_episodes=bad, infeasible_grid_episodes=infeasible,
                   finite_difference_acceleration_exceeded_episodes=finite_difference_bad,
                   finite_difference_groups=fd_groups,
                   dense_peak_velocity_util=report['retimed_peak_v_util'],
                   dense_peak_acceleration_util=report['retimed_peak_a_util'], comparison=example,
                   derivative_method='forward first and second differences of stored joint angles at 30 Hz; masks require whole stencil',
                   audit=audit)
    (output / 'validation.json').write_text(json.dumps(summary, indent=2))
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    episodes = [r['episode'] for r in reps]
    axes[0].bar(np.array(episodes)-.2, [r['src_seconds'] for r in reps], width=.4, label='Original')
    axes[0].bar(np.array(episodes)+.2, [r['seconds'] for r in reps], width=.4, label='Retimed')
    axes[0].set_ylabel('Duration (s)'); axes[0].legend()
    peaks = [max(r['retimed_peak_a_util']) for r in reps]
    axes[1].bar(episodes, peaks, color=['#d94c47' if p>1.0001 else '#378e87' for p in peaks])
    axes[1].axhline(1, color='red', linestyle='--', label='Configured acceleration limit')
    axes[1].set(xlabel='Episode index (zero based)', ylabel='Dense acceleration / limit')
    axes[1].legend()
    fig.savefig(output / 'summary.png', dpi=130); plt.close(fig)
    rows = ''.join(f"<tr><td>{r['episode']}</td><td>{r['src_frames']}</td><td>{r['frames']}</td>"
                   f"<td>{r['src_seconds']:.2f}</td><td>{r['seconds']:.2f}</td>"
                   f"<td>{r['src_frames']/r['frames']:.3f}</td><td>{max(r['retimed_peak_a_util']):.3f}</td>"
                   f"<td>{r['infeasible_grid_points']}</td></tr>" for r in reps)
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>34 条轨迹加速实验报告</title>
<style>body{{font:16px/1.7 system-ui,sans-serif;max-width:1060px;margin:36px auto;padding:0 24px;color:#173047;background:#f6f8fb}}
section{{background:white;border-radius:12px;padding:22px;margin:18px 0}}h1,h2{{line-height:1.4}}.warn{{border-left:6px solid #d94c47}}
table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border-bottom:1px solid #ddd;padding:6px;text-align:right}}
img,video{{max-width:100%}}code{{overflow-wrap:anywhere}}.metric{{font-size:25px;font-weight:700}}</style>
<h1>34 条轨迹加速实验报告</h1><p>2026-11-23 · 20261102 单箱搬运数据 · 第一次默认参数基线</p>
<section><div class="metric">{report['src_seconds']:.2f} 秒 → {report['seconds']:.2f} 秒 ｜ {report['speedup']:.3f} 倍速</div>
<p>共 34 条轨迹、5 路相机；17245 帧 → {len(rdf)} 帧，节省 {summary['time_saved_percent']:.2f}% 时间。输出仍为 30 Hz。
本次没有发生悬停剪切，加速来自重新分配轨迹执行时间。总时长按帧数 / 30 计算。</p></section>
<section class="warn"><h2>结论：已跑通，但关节限制验证未通过</h2>
<p>结构、映射与视频检查通过；第 9 号轨迹（从 0 编号）的第二关节加速度峰值约为上限的 1.959 倍，约 117.54°/s²，上限为 60°/s²。
峰值出现在原始时间 8.0 秒、离开保护窗口的连接位置。该段求解记录了不可行网格点，现有算法仍使用回退值继续输出。</p>
<p>第 1、9 号轨迹各记录 2 个不可行网格点。其余轨迹的密集采样解析速度、加速度检查没有超过配置硬上限，
但这不等于完整输出或真实机器人验证通过。30 Hz 关节角二阶差分另有 {len(finite_difference_bad)} 条轨迹超限，见 validation.json；
该检查包括原样保留的保护帧以及连接处，与解析曲线检查口径不同。</p>
<p>当前结果保留作实验基线。下一步应修复不可行求解的回退处理与保护窗口连接策略，再重新验证，而不能只依赖平均加速倍率。</p></section>
<section><h2>检查范围</h2><ul><li>34 条轨迹：帧数、时间戳、状态/动作维度及有限值、连续索引、原始帧映射、首尾关节角、保护帧原样保留。</li>
<li>commander_state：按原始图像帧映射取整数值，保持 Arrow 类型，并更新全局与逐轨迹统计。另用 0→1→2 的合成数据完成回归测试，旧格式也通过。</li>
<li>170 段输出视频：逐帧解码计数，确认 256×256、30 fps，与每条轨迹帧数一致。另抽查 3 条轨迹各 1 路相机的 5 个位置验证图像映射。</li>
<li>本报告未开展 Isaac Sim 动力学、碰撞、力矩或真实机器人验证。jerk 和底盘加速度约束本次未启用。</li></ul></section>
<section><h2>输出离散点的加速度：区分来源</h2>
<p>对关节角作 30 Hz 二阶差分，每个结果使用连续 3 帧；只有 3 帧均处于同一区域才归入该区域，否则归入连接处。原始数据与输出使用同一检查口径。</p>
<table><tr><th>区域</th><th>超限轨迹数</th><th>峰值 / 上限</th><th>超限采样位置数</th></tr>
{''.join(f'<tr><td>{label}</td><td>{fd_groups[key]["episodes_exceed"]}</td><td>{fd_groups[key]["peak"]:.3f}</td><td>{fd_groups[key]["exceed_samples"]}</td></tr>' for key, label in [('source/all','原始数据全部'),('output/all','加速后全部'),('output/retimed','加速后：纯加速区间'),('output/protected','加速后：原样保护区间'),('output/boundary','加速后：连接处')])}
</table><p>二阶差分检查反映输出离散点的变化。它可能漏掉帧间很短的尖峰，因此纯加速区间差分通过不能消除第 9 号轨迹解析曲线的超限。</p></section>
<section><h2>加速前后对比</h2><p>示例为第 {representative} 号轨迹：{example['source_seconds']:.2f} 秒 → {example['retimed_seconds']:.2f} 秒，
{example['speedup']:.3f} 倍速。左侧原始、右侧加速，同时开始且均按 30 fps 播放；右侧结束后定格，上方 surround、下方 cam9。
这是实采画面的重采样对比，不是仿真或实际执行后重新拍摄的结果。示例选择接近中位倍率的轨迹，异常轨迹另见下图与表格。</p>
<video controls preload="metadata" poster="comparison_preview.png" src="comparison.mp4"></video></section>
<section><h2>34 条轨迹结果</h2><img src="summary.png" alt="轨迹时长与加速度检查">
<table><thead><tr><th>轨迹</th><th>原帧数</th><th>新帧数</th><th>原秒数</th><th>新秒数</th><th>倍率</th><th>加速度/上限</th><th>不可行点</th></tr></thead><tbody>{rows}</tbody></table></section>
<section><h2>参数与复现</h2><p>最大局部倍率 3；margin=0.98；速度上限 [63,36,90,108,162,162]°/s；
加速度上限 [120,60,90,96,192,192]°/s²；保护窗口前后各 0.8 秒；样条平滑 λ=0.0001；网格细分 4；视频 libx264、QP=18（有损）。</p>
<p>输入：<code>{html.escape(str(src))}</code></p><p>输出：<code>{html.escape(str(dst))}</code></p>
<p>执行命令和代码 SHA256 见 launch.json；逐帧映射见输出中的 frame_map.parquet；完整检查见 <a href="validation.json">validation.json</a>。
此批为 20261102 的 34 条数据，与用户提供的历史 496 条报告不同。</p></section></html>'''
    (output / 'report.html').write_text(page, encoding='utf-8')
    (output / 'launch.json').write_text(json.dumps(launch, indent=2))
    (output / 'retime_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k != 'audit'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
