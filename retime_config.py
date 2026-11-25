"""Load validated YAML defaults without changing argparse CLI precedence."""
import argparse
import math
from pathlib import Path
import sys


def parse_config_args(parser, default_config, argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument('--config', metavar='PATH', help='YAML config (default: config/config.yaml beside script)')
    selector.add_argument('--no-config', action='store_true', help='ignore YAML and use built-in defaults')
    parser.add_argument('--dwell-cut', dest='no_dwell_cut', action='store_false',
                        default=argparse.SUPPRESS, help='enable dwell removal, overriding YAML')
    parser.add_argument('--plot', dest='no_plot', action='store_false',
                        default=argparse.SUPPRESS, help='enable plots, overriding YAML')
    parser.add_argument('--no-overwrite', dest='overwrite', action='store_false',
                        default=argparse.SUPPRESS, help='disable overwriting, overriding YAML')
    # Parse only the config selector before installing defaults for required options.
    probe = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    group = probe.add_mutually_exclusive_group()
    group.add_argument('--config')
    group.add_argument('--no-config', action='store_true')
    selected, _ = probe.parse_known_args(argv)
    if '-h' in argv or '--help' in argv:
        return parser.parse_args(argv)
    path = Path(selected.config) if selected.config else Path(default_config)
    if not selected.no_config and (selected.config is not None or path.exists()):
        try:
            import yaml
        except ImportError:
            parser.error('YAML configuration requires PyYAML: python -m pip install "PyYAML>=6,<7"')

        class UniqueLoader(yaml.SafeLoader):
            pass

        def unique_mapping(loader, node):
            result = {}
            for key_node, value_node in node.value:
                key = loader.construct_object(key_node)
                if not isinstance(key, str):
                    raise ValueError('config keys must be strings')
                if key in result:
                    raise ValueError(f'duplicate config key: {key}')
                result[key] = loader.construct_object(value_node)
            return result

        UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)
        try:
            with path.open(encoding='utf-8-sig') as stream:
                values = yaml.load(stream, Loader=UniqueLoader)
            if not isinstance(values, dict):
                raise ValueError('config must be a YAML mapping of parameter names to values')
            actions = {}
            for action in parser._actions:
                actions.setdefault(action.dest, action)
            forbidden = {'help', 'config', 'no_config'}
            unknown = set(values) - (set(actions) - forbidden)
            if unknown:
                raise ValueError('unknown config keys: ' + ', '.join(sorted(unknown)))
            validated = {}
            for key, value in values.items():
                action = actions[key]
                if value is None:
                    if action.default is not None:
                        raise ValueError(f'{key}: null is not allowed')
                    continue
                if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
                    if type(value) is not bool:
                        raise ValueError(f'{key}: expected true or false')
                    validated[key] = value
                    continue
                is_list = action.nargs in ('+', '*') or isinstance(action.nargs, int)
                items = value if is_list else [value]
                if is_list and (not isinstance(value, list) or not value):
                    raise ValueError(f'{key}: expected a non-empty list')
                if isinstance(action.nargs, int) and len(items) != action.nargs:
                    raise ValueError(f'{key}: expected exactly {action.nargs} values')
                converted = []
                for item in items:
                    expected = action.type or str
                    valid = (type(item) in (int, float) if expected is float else type(item) is expected)
                    if not valid:
                        raise ValueError(f'{key}: expected {expected.__name__} values')
                    item = expected(item)
                    if isinstance(item, float) and not math.isfinite(item):
                        raise ValueError(f'{key}: expected finite numbers')
                    if action.choices is not None and item not in action.choices:
                        raise ValueError(f'{key}: expected one of {action.choices}')
                    converted.append(item)
                validated[key] = converted if is_list else converted[0]
            parser.set_defaults(**validated, config=str(path.resolve()))
            for key in validated:
                actions[key].required = False
        except (OSError, ValueError, yaml.YAMLError) as exc:
            parser.error(f'cannot load config {path}: {exc}')
    return parser.parse_args(argv)
