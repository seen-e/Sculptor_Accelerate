"""Compare original joint samples with the exact path used by a recorded retiming run.

All comparisons use source time, not execution time. Between recorded samples,
the straight-line reference is a diagnostic convention, not ground truth.
"""
import hashlib
import html
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def table(root, folder):
    return pa.concat_tables([pq.read_table(p) for p in sorted((root / folder).rglob('*.parquet'))]).to_pandas()


def reconstruct(m, state, fps):
    t = np.arange(len(state)) / fps
    q = state[:, :6].astype(float)
    base = m.base_path(t, q)
    _, windows = m.protected_windows(t, state, base, fps)
    protected = np.zeros(len(t), bool)
    for a, b in windows:
        protected[a:b+1] = True
    _, dwells = m.find_dwells(state, fps, protected)
    assert not dwells, 'This source-time comparison requires a run without dwell cuts'
    stretches, start = [], 0
    for a, b in windows:
        if a > start:
            stretches.append((start, a))
        start = b
    if start < len(t)-1:
        stretches.append((start, len(t)-1))
    paths = [(a, b, m.stretch_path(base, t, q, a, b)) for a, b in stretches]

    def evaluate(times):
        # Protected regions are shown as straight connections between original
        # samples; the actual pipeline emits their recorded rows unchanged.
        result = np.column_stack([np.interp(times, t, q[:, j]) for j in range(6)])
        moving = np.zeros(len(times), bool)
        for a, b, path in paths:
            mask = (times >= t[a]) & (times <= t[b])
            result[mask] = path(times[mask])
            moving[mask] = True
        return result, moving

    fit, _ = evaluate(t)
    fit[protected] = q[protected]
    td = np.linspace(t[0], t[-1], (len(t)-1)*20+1)
    dense, moving = evaluate(td)
    reference = np.column_stack([np.interp(td, t, q[:, j]) for j in range(6)])
    k = np.minimum(np.searchsorted(t, td, side='right')-1, len(t)-2)
    low, high = np.minimum(q[k], q[k+1]), np.maximum(q[k], q[k+1])
    local_excess = np.maximum(np.maximum(low-dense, dense-high), 0)
    local_excess[~moving] = 0
    return dict(t=t, q=q, base=base, fit=fit, protected=protected, td=td,
                dense=dense, moving=moving, reference=reference, windows=windows,
                local_excess=local_excess, evaluate=evaluate)


def metrics(r):
    err = np.rad2deg(r['fit'] - r['q'])
    move = ~r['protected']
    absolute = np.abs(err[move])
    max_row, max_joint = np.unravel_index(np.argmax(np.abs(err)), err.shape)
    base_error = np.rad2deg(r['base'](r['t'])-r['q'])
    global_excess = np.maximum(np.maximum(r['q'].min(0)-r['dense'], r['dense']-r['q'].max(0)), 0)
    joints = []
    for j in range(6):
        joints.append(dict(joint=j+1, max_abs_error_deg=float(absolute[:, j].max()),
                           mean_abs_error_deg=float(absolute[:, j].mean()),
                           p99_abs_error_deg=float(np.quantile(absolute[:, j], .99)),
                           max_local_envelope_excess_deg=float(np.rad2deg(r['local_excess'][:, j].max())),
                           max_global_range_excess_deg=float(np.rad2deg(global_excess[:, j].max()))))
    return dict(max_abs_error_deg=float(absolute.max()), mean_abs_error_deg=float(absolute.mean()),
                p99_abs_error_deg=float(np.quantile(absolute, .99)),
                max_error_joint=int(max_joint+1), max_error_source_time=float(r['t'][max_row]),
                max_base_fit_error_all_nodes_deg=float(np.abs(base_error).max()),
                max_dense_vs_linear_reference_deg=float(np.rad2deg(np.abs(r['dense']-r['reference'])[r['moving']]).max()),
                max_local_envelope_excess_deg=float(np.rad2deg(r['local_excess'].max())),
                max_global_range_excess_deg=float(np.rad2deg(global_excess.max())),
                protected_nodes=int(r['protected'].sum()), unprotected_nodes=int(move.sum()), joints=joints)


def figures(output, ei, r, stats):
    t, td = r['t'], r['td']
    q, fit = np.rad2deg(r['q']), np.rad2deg(r['dense'])
    error = np.rad2deg(r['dense']-r['reference'])
    node_error = np.rad2deg(r['fit']-r['q'])
    fig, axes = plt.subplots(6, 2, figsize=(15, 15), sharex=True, constrained_layout=True)
    for j in range(6):
        left, right = axes[j]
        left.plot(t, q[:, j], color='#2570aa', lw=1.5, label='Original samples (connected)')
        left.plot(td, fit[:, j], color='#ec7b24', lw=1, ls='--', label='Path used for retiming')
        left.set_ylabel(f'J{j+1} angle (deg)')
        right.plot(td, error[:, j], color='#ec7b24', lw=.8, label='Vs linear sample reference')
        right.scatter(t[~r['protected']], node_error[~r['protected'], j], color='#2570aa', s=3,
                      label='Error at original samples', zorder=3)
        right.axhline(0, color='black', lw=.5)
        right.set_ylabel(f'J{j+1} difference (deg)')
        right.set_title(f"Original-node max = {stats['joints'][j]['max_abs_error_deg']:.4f} deg", fontsize=10)
        for ax in (left, right):
            ax.grid(alpha=.2)
            for a, b in r['windows']:
                ax.axvspan(t[a], t[b], color='gray', alpha=.15)
    axes[0, 0].legend(fontsize=8); axes[0, 1].legend(fontsize=8)
    axes[-1, 0].set_xlabel('Original source time (s)'); axes[-1, 1].set_xlabel('Original source time (s)')
    fig.suptitle(f'Episode {ei}: original vs fitted joint angles\nGrey = protected original rows; time axes are aligned to the source', fontsize=15)
    fig.savefig(output / f'ep{ei:03d}_all_joints.png', dpi=125); plt.close(fig)

    j = stats['max_error_joint']-1
    center = stats['max_error_source_time']
    lo, hi = max(t[0], center-.6), min(t[-1], center+.6)
    md, mn = (td >= lo) & (td <= hi), (t >= lo) & (t <= hi)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, constrained_layout=True)
    axes[0].plot(t[mn], q[mn, j], 'o-', color='#2570aa', ms=4, lw=1, label='Original samples')
    axes[0].plot(td[md], fit[md, j], color='#ec7b24', lw=2, label='Fitted path + endpoint correction')
    axes[0].plot(td[md], np.rad2deg(r['base'](td[md])[:, j]), color='#888888', ls=':', label='Base smoothing spline')
    axes[0].set_ylabel(f'J{j+1} angle (deg)'); axes[0].legend(fontsize=9)
    axes[1].plot(td[md], error[md, j], color='#ec7b24', label='Fitted - linear reference')
    axes[1].scatter(t[mn], node_error[mn, j], color='#2570aa', s=18, label='Fitted - recorded sample', zorder=3)
    axes[1].axhline(0, color='black', lw=.7)
    axes[1].set(xlabel='Original source time (s)', ylabel='Difference (deg)'); axes[1].legend(fontsize=9)
    for ax in axes:
        ax.axvline(center, color='#ba3e3e', ls='--', lw=.8)
        ax.grid(alpha=.2)
        for a, b in r['windows']:
            ax.axvspan(t[a], t[b], color='gray', alpha=.15)
        ax.set_xlim(lo, hi)
    fig.suptitle(f"Episode {ei}, J{j+1}: largest original-node difference = {stats['max_abs_error_deg']:.4f} deg\nAt source time {center:.3f} s; grey = protected rows", fontsize=13)
    fig.savefig(output / f'ep{ei:03d}_detail.png', dpi=140); plt.close(fig)
    np.savez_compressed(output / f'ep{ei:03d}_joint_comparison.npz', source_time_s=t,
                        original_angle_deg=q, fitted_at_source_nodes_deg=np.rad2deg(r['fit']),
                        protected=r['protected'], dense_source_time_s=td, fitted_dense_angle_deg=fit,
                        linear_reference_dense_deg=np.rad2deg(r['reference']))


def main():
    run = Path(sys.argv[1])
    output = run / 'joint_fit_comparison'
    output.mkdir(exist_ok=True)
    launch = json.loads((run/'launch.json').read_text())
    script = Path(__file__).with_name('retime_lerobot.py')
    assert hashlib.sha256(script.read_bytes()).hexdigest() == launch['code_sha256']
    dst = next(p for p in (run/'output').iterdir() if (p/'retime_report.json').exists())
    report = json.loads((dst/'retime_report.json').read_text())
    sys.argv = ['retime_lerobot.py', '--src', 'unused']
    import retime_lerobot as m
    for key, val in report['params'].items():
        setattr(m.args, key, val)
    m.V = np.deg2rad(m.args.vmax_deg); m.A = np.deg2rad(m.args.amax_deg)
    source, result = table(Path(launch['source']), 'data'), table(dst, 'data')
    mapping = pq.read_table(dst/'frame_map.parquet').to_pandas()
    reconstructed, summaries = {}, []
    for ei in sorted(source.episode_index.unique()):
        raw = source[source.episode_index == ei].sort_values('frame_index')
        r = reconstruct(m, np.stack(raw['observation.state']), report['fps'])
        stats = dict(episode=int(ei), **metrics(r))
        # Verify the reconstruction against actual output positions using the
        # recorded mapping, so this is not a newly chosen fitting method.
        f = mapping[mapping.episode_index == ei].sort_values('frame_index')
        out = result[result.episode_index == ei].sort_values('frame_index')
        predicted, _ = r['evaluate'](f.source_time_s.to_numpy())
        p = f.protected.to_numpy()
        predicted[p] = r['q'][f.original_frame_index.to_numpy()[p]]
        actual = np.stack(out['observation.state'])[:, :6].astype(float)
        np.testing.assert_allclose(predicted, actual, rtol=0, atol=3e-7)
        stats['reconstruction_max_error_deg'] = float(np.rad2deg(np.abs(predicted-actual).max()))
        reconstructed[int(ei)] = r; summaries.append(stats)
    worst = max(summaries, key=lambda x: x['max_abs_error_deg'])['episode']
    selected = list(dict.fromkeys([0, 9, 22, worst]))
    for item in sorted(summaries, key=lambda x: -x['max_abs_error_deg']):
        if len(selected) >= 4:
            break
        if item['episode'] not in selected:
            selected.append(item['episode'])
    chosen = [next(s for s in summaries if s['episode'] == ei) for ei in selected]
    for stats in chosen:
        figures(output, stats['episode'], reconstructed[stats['episode']], stats)
    summary = dict(selected_episodes=selected, worst_original_node_episode=worst,
                   code_sha256=launch['code_sha256'], params=report['params'], episodes=summaries,
                   definitions={'node_error':'Corrected fitted path minus recorded angle at the same original source time, excluding protected nodes',
                                'local_envelope':'Dense path outside the range of the two adjacent recorded samples; includes smoothing displacement, not proof of a new extremum',
                                'dense_reference':'Piecewise linear connection of samples; actual inter-sample physical motion is unknown',
                                'dense_resolution':'20 subdivisions per original 30 Hz interval; numerical scan, not an analytic bound'})
    (output/'comparison_metrics.json').write_text(json.dumps(summary, indent=2))
    rows = ''.join(f"<tr><td>{s['episode']}</td><td>{s['mean_abs_error_deg']:.5f}</td><td>{s['p99_abs_error_deg']:.5f}</td>"
                   f"<td>{s['max_abs_error_deg']:.5f}</td><td>J{s['max_error_joint']} / {s['max_error_source_time']:.3f}s</td>"
                   f"<td>{s['max_local_envelope_excess_deg']:.5f}</td></tr>" for s in chosen)
    sections = []
    for s in chosen:
        ei = s['episode']
        jointrows = ''.join(f"<tr><td>J{j['joint']}</td><td>{j['mean_abs_error_deg']:.5f}</td><td>{j['p99_abs_error_deg']:.5f}</td><td>{j['max_abs_error_deg']:.5f}</td>"
                            f"<td>{j['max_local_envelope_excess_deg']:.5f}</td><td>{j['max_global_range_excess_deg']:.5f}</td></tr>" for j in s['joints'])
        sections.append(f'''<section id="ep{ei}"><h2>Episode {ei}</h2><p>最大原始节点偏差 {s['max_abs_error_deg']:.5f}°，位于 J{s['max_error_joint']}，原始时间 {s['max_error_source_time']:.3f} 秒。</p>
<a href="ep{ei:03d}_detail.png"><img src="ep{ei:03d}_detail.png" alt="最大偏差局部放大"></a>
<details><summary>展开六个关节的全程对比图与数据</summary><a href="ep{ei:03d}_all_joints.png"><img src="ep{ei:03d}_all_joints.png" alt="六关节全程对比"></a></details>
<table><tr><th>关节</th><th>平均绝对偏差°</th><th>P99偏差°</th><th>最大偏差°</th><th>相邻点范围外°</th><th>全程角度范围外°</th></tr>{jointrows}</table>
<p><a href="ep{ei:03d}_joint_comparison.npz">下载对比数组 NPZ</a>（包含原始角度、同一时刻拟合角度、密集曲线和保护标记）</p></section>''')
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>关节角拟合前后比较</title>
<style>body{{font:16px/1.75 system-ui,sans-serif;color:#183047;background:#f4f7fb;max-width:1150px;margin:30px auto;padding:0 22px}}section{{background:white;margin:20px 0;padding:24px;border-radius:12px}}h1,h2{{line-height:1.4}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border-bottom:1px solid #ddd;padding:8px;text-align:right}}img{{width:100%;height:auto}}summary{{cursor:pointer;color:#17629b}}.note{{border-left:5px solid #e49d2c}}a{{color:#17629b}}</style>
<h1>关节角拟合前后比较</h1><p>2026-11-24 · 20261102 数据集 · 全部 34 条扫描，展示 4 条代表轨迹</p>
<section><h2>比较的是同一个原始时刻</h2><p>蓝色是原始采样关节角，橙色是本次加速实际采用的拟合路径，包含端点修正。横轴统一为原始时间 s，避免把“执行更快”误当成“路径偏离”。
灰色区域为保护窗口，实际输出保留原始帧；图中的直线只用于连接离散采样点。局部图的灰色点线是端点修正前的基础平滑样条。</p>
<p>使用既有实验的同一份代码及参数重建曲线：λ=0.0001，端点混合长度至多 0.5 秒。已逐条对照 frame_map 与真实输出关节角，误差在 float32 舍入量级。
本批没有删除悬停点。所有角度单位为度，episode 从 0 编号。</p></section>
<section><h2>抽取结果</h2><p>选择第一条 Episode 0、此前加速度异常的 Episode 9、视频示例 Episode 22，以及全批原始节点拟合偏差最大的 Episode {worst}；如重合则补入下一条偏差较大的轨迹。</p>
<table><tr><th>Episode</th><th>平均绝对偏差°</th><th>P99偏差°</th><th>最大偏差°</th><th>最大偏差位置</th><th>相邻点范围外°</th></tr>{rows}</table>
<p>前三项偏差只统计非保护原始节点的六关节数据，避免保护帧的零偏差稀释结果。<a href="comparison_metrics.json">全部 34 条指标 JSON</a></p></section>
<section class="note"><h2>如何理解“过冲”指标</h2><p>“相邻点范围外”表示曲线超出相邻两个原始角度的区间。它可能来自平滑偏移或点间峰值，不能直接认定为新增过冲或碰撞。
“全程角度范围外”只检查是否超出该关节整条原始轨迹的最大/最小角度。即使不超出，也可能改变局部避让。</p>
<p>点间误差图以原始点的直线连接为参照，真实采样间运动未知；每个原始间隔细分 20 段进行数值检查，不是严格连续上界。
这些角度对比尚不能证明避障距离或运动细节足够保留，也不评估实机跟踪过冲。Episode 9 的加速度超限属于另一个指标，角度偏差小不代表加速度合格。</p></section>
{''.join(sections)}</html>'''
    (output/'report.html').write_text(page, encoding='utf-8')
    print(json.dumps(dict(selected=chosen, worst_episode=worst,
                         max_reconstruction_error_deg=max(s['reconstruction_max_error_deg'] for s in summaries)), indent=2), flush=True)


if __name__ == '__main__':
    main()
