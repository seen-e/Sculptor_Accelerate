"""Exercise the actual entrypoint's YAML/CLI parsing and retiming behavior."""
import contextlib
import io
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'retime_lerobot.py'
sys.path.insert(0, str(ROOT))


def load(*options):
    with patch.object(sys, 'argv', [str(SCRIPT), *map(str, options)]):
        return runpy.run_path(str(SCRIPT), run_name='config_test')


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.config = Path(self.tmp.name) / 'experiment.yaml'

    def write(self, content):
        self.config.write_text(content, encoding='utf-8')
        return self.config

    def test_default_file_matches_builtin_values(self):
        configured = vars(load('--src', 'unused')['args']).copy()
        builtin = vars(load('--no-config', '--src', 'unused')['args']).copy()
        self.assertEqual(configured.pop('config'), str(ROOT / 'config/config.yaml'))
        builtin.pop('config')
        configured.pop('no_config')
        builtin.pop('no_config')
        self.assertEqual(configured, builtin)

    def test_yaml_only_source_and_cli_override(self):
        self.write('src: [dataset_a, dataset_b]\nframe_step: 1.5\nencoder: x264\n')
        args = load('--config', self.config)['args']
        self.assertEqual(args.src, ['dataset_a', 'dataset_b'])
        self.assertEqual(args.frame_step, 1.5)
        self.assertEqual(args.encoder, 'x264')
        args = load('--config', self.config, '--src', 'cli_dataset', '--frame-step', '2')['args']
        self.assertEqual(args.src, ['cli_dataset'])
        self.assertEqual(args.frame_step, 2.)

    def test_boolean_overrides_both_directions(self):
        for initial in [False, True]:
            self.write(f'src: [unused]\nno_plot: {str(initial).lower()}\n'
                       f'no_dwell_cut: {str(initial).lower()}\noverwrite: {str(initial).lower()}\n')
            flags = ('--plot', '--dwell-cut', '--no-overwrite') if initial else (
                '--no-plot', '--no-dwell-cut', '--overwrite')
            args = load('--config', self.config, *flags)['args']
            self.assertEqual([args.no_plot, args.no_dwell_cut, args.overwrite], [not initial]*3)

    def test_invalid_config_fails_before_processing(self):
        cases = {
            'typo': ('frame_stpe: 2', 'unknown config keys'),
            'duplicate': ('margin: 0.9\nmargin: 0.8', 'duplicate config key'),
            'boolean': ('no_plot: "false"', 'expected true or false'),
            'list_size': ('vmax_deg: [1, 2]', 'exactly 6'),
            'list_type': ('src: data', 'non-empty list'),
            'int_type': ('subdiv: 4.5', 'expected int'),
            'choice': ('encoder: invalid', 'expected one of'),
            'null': ('margin: null', 'null is not allowed'),
            'nan': ('margin: .nan', 'finite numbers'),
            'structure': ('- not_a_mapping', 'must be a YAML mapping'),
            'syntax': ('margin: [', 'cannot load config'),
        }
        for name, (content, message) in cases.items():
            with self.subTest(name=name):
                self.write(content)
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
                    load('--config', self.config, '--src', 'unused')
                self.assertEqual(error.exception.code, 2)
                self.assertIn(message, stderr.getvalue())

    def test_missing_explicit_file_and_required_source(self):
        for options in [('--config', self.config, '--src', 'unused'), ()]:
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    load(*options)
                self.assertEqual(error.exception.code, 2)

    def test_default_file_is_relative_to_script(self):
        original = os.getcwd()
        try:
            os.chdir(self.tmp.name)
            args = load('--src', 'relative_dataset')['args']
            self.assertEqual(args.config, str(ROOT / 'config/config.yaml'))
            self.assertEqual(args.src, ['relative_dataset'])
        finally:
            os.chdir(original)

    def test_help_does_not_require_valid_config(self):
        self.write('invalid: [')
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), self.assertRaises(SystemExit) as error:
            load('--config', self.config, '--help')
        self.assertEqual(error.exception.code, 0)
        self.assertIn('--no-config', stdout.getvalue())

    def test_retiming_equal_with_defaults_and_changes_with_yaml(self):
        state = np.zeros((90, 30), dtype=np.float32)
        state[:, 0] = np.linspace(0, .15, len(state))
        action = np.zeros((90, 10), dtype=np.float32)
        action[:, 0] = np.r_[state[1:, 0], state[-1, 0]]
        configured = load('--src', 'unused', '--no-dwell-cut')
        builtin = load('--no-config', '--src', 'unused', '--no-dwell-cut')
        a = configured['retime_episode'](state, action, 30.)
        b = builtin['retime_episode'](state, action, 30.)
        for key in a:
            if key == 'report':
                self.assertEqual(a[key], b[key])
            else:
                np.testing.assert_array_equal(a[key], b[key])
        self.assertLess(len(a['state']), len(state))
        self.write('src: [unused]\nframe_step: 2.0\nno_dwell_cut: true\n')
        slower = load('--config', self.config)['retime_episode'](state, action, 30.)
        self.assertGreater(len(slower['state']), len(a['state']))


if __name__ == '__main__':
    unittest.main()
