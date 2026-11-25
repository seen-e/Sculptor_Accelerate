"""Integration regression for commander mapping and v3 metadata without quantiles."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def main():
    template, root = map(Path, sys.argv[1:3])
    root.mkdir(parents=True, exist_ok=False)
    info = json.loads((template / 'meta/info.json').read_text())
    schema = pq.read_schema(next((template / 'data').rglob('*.parquet')))
    source_eps = pq.read_table(next((template / 'meta/episodes').rglob('*.parquet'))).slice(0, 1)
    length = 90
    q = np.linspace(0, 0.15, length)
    state = np.zeros((length, 30), np.float32)
    state[:, 0] = q
    action = np.zeros((length, 10), np.float32)
    action[:, 0] = np.r_[q[1:], q[-1]]
    commander = np.repeat([0, 1, 2], length // 3)
    video = root / 'synthetic.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                    'testsrc2=size=256x256:rate=30', '-frames:v', str(length),
                    '-c:v', 'libx264', '-threads', '1', '-qp', '0', str(video)], check=True)
    results = []
    for variant in ('commander', 'legacy'):
        src = root / variant
        (src / 'meta').mkdir(parents=True)
        variant_info = json.loads(json.dumps(info))
        values = {'observation.state': list(state), 'action': list(action),
                  'observation.commander_state': commander, 'timestamp': np.arange(length) / 30,
                  'frame_index': np.arange(length), 'episode_index': np.zeros(length, np.int64),
                  'index': np.arange(length), 'task_index': np.zeros(length, np.int64)}
        fields = [f for f in schema if variant == 'commander' or f.name != 'observation.commander_state']
        data_path = src / info['data_path'].format(chunk_index=0, file_index=0)
        data_path.parent.mkdir(parents=True)
        pq.write_table(pa.table({f.name: pa.array(values[f.name], type=f.type) for f in fields}), data_path)
        e = source_eps.to_pylist()[0]
        e.update(episode_index=0, length=length, dataset_from_index=0, dataset_to_index=length)
        for key, feature in info['features'].items():
            if feature['dtype'] != 'video':
                continue
            e[f'videos/{key}/chunk_index'] = 0
            e[f'videos/{key}/file_index'] = 0
            e[f'videos/{key}/from_timestamp'] = 0.
            e[f'videos/{key}/to_timestamp'] = length / 30
            path = src / info['video_path'].format(video_key=key, chunk_index=0, file_index=0)
            path.parent.mkdir(parents=True)
            shutil.copy2(video, path)
        ep_fields = [f for f in source_eps.schema if variant == 'commander' or 'commander_state' not in f.name]
        ep_path = src / 'meta/episodes/chunk-000/file-000.parquet'
        ep_path.parent.mkdir(parents=True)
        pq.write_table(pa.table({f.name: pa.array([e[f.name]], type=f.type) for f in ep_fields}), ep_path)
        variant_info.update(total_frames=length, total_episodes=1, splits={'train': '0:1'})
        if variant == 'legacy':
            variant_info['features'].pop('observation.commander_state')
        (src / 'meta/info.json').write_text(json.dumps(variant_info))
        stats = json.loads((template / 'meta/stats.json').read_text())
        if variant == 'legacy':
            stats.pop('observation.commander_state')
        (src / 'meta/stats.json').write_text(json.dumps(stats))
        shutil.copy2(template / 'meta/tasks.parquet', src / 'meta/tasks.parquet')
        cmd = [sys.executable, str(Path(__file__).with_name('retime_lerobot.py')), '--src', str(src),
               '--out-root', str(root / 'output'), '--workers', '1', '--jobs', '1', '--encoder', 'x264',
               '--qp', '0', '--verify', '1', '--no-plot', '--no-dwell-cut']
        subprocess.run(cmd, check=True)
        dst = root / 'output' / (variant + '_retime')
        out = pq.read_table(next((dst / 'data').rglob('*.parquet')))
        fm = pq.read_table(dst / 'frame_map.parquet')
        assert out.column_names == [f.name for f in fields]
        assert len(out) < length, 'Fixture must actually skip source frames'
        if variant == 'commander':
            expected = commander[fm['original_frame_index'].to_numpy()]
            np.testing.assert_array_equal(out['observation.commander_state'].to_numpy(), expected)
            assert set(expected) == {0, 1, 2}
            assert out.schema.field('observation.commander_state').type == schema.field('observation.commander_state').type
            ep = pq.read_table(next((dst / 'meta/episodes').rglob('*.parquet'))).to_pylist()[0]
            assert ep['stats/observation.commander_state/count'] == [len(out)]
            np.testing.assert_allclose(ep['stats/observation.commander_state/mean'], [expected.mean()])
            np.testing.assert_allclose(ep['stats/observation.commander_state/std'], [expected.std()])
        report = json.loads((dst / 'retime_report.json').read_text())
        assert report['image_alignment_check'][0]['mae_vs_mapped_orig'] == 0
        results.append(dict(variant=variant, input_frames=length, output_frames=len(out), passed=True))
    (root / 'result.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results))


if __name__ == '__main__':
    main()
