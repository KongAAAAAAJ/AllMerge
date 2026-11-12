"""Expert collection YAML validation and project-relative path resolution."""
from __future__ import annotations

import argparse
from copy import deepcopy
import math
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = 'configs/data/expert_collection.yaml'
SCENARIOS = ('straight', 'curved', 'merge_in', 'merge_out')


def resolve_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def _number(value, name, minimum=0, inclusive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be numeric')
    value = float(value)
    if not math.isfinite(value) or (value < minimum if inclusive else value <= minimum):
        raise ValueError(f'Invalid {name}: {value}')
    return value


def _boolean(value, name):
    if type(value) is not bool:
        raise ValueError(f'{name} must be a YAML boolean')
    return value


def load_config(filename=DEFAULT_CONFIG):
    path = resolve_path(filename)
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(cfg, dict):
        raise ValueError('Expert collection YAML must be a mapping')
    cfg = deepcopy(cfg)
    mode = cfg.setdefault('vehicle_model', 'kinematic')
    if mode not in ('kinematic', 'trucksim'):
        raise ValueError('vehicle_model must be kinematic or trucksim')
    scenarios = cfg.setdefault('scenarios', list(SCENARIOS))
    if (not isinstance(scenarios, list) or not scenarios
            or any(not isinstance(s, str) or s not in SCENARIOS for s in scenarios)
            or len(set(scenarios)) != len(scenarios)):
        raise ValueError('scenarios must contain unique supported scenario names')
    for key in ('collection', 'conditions'):
        if not isinstance(cfg.setdefault(key, {}), dict):
            raise ValueError(f'{key} must be a mapping')
    c = cfg['collection']
    for name, default in [('target_samples', 1000), ('samples_per_shard', 2048),
                          ('max_episodes', 10000), ('worker_retries', 2),
                          ('max_consecutive_empty_episodes', 20)]:
        c[name] = _integer(c.get(name, default), name)
    for name, default in [('start_seed', 1), ('start_episode_index', 0), ('split_seed', 0)]:
        c[name] = _integer(c.get(name, default), name, 0)
    c['group_action'] = _integer(c.get('group_action', 3), 'group_action', 0)
    if c['group_action'] > 3:
        raise ValueError('group_action must be 0, 1, 2 or 3')
    for name, default in [('episode_duration_s', 5.0), ('timeout_s', 300)]:
        c[name] = _number(c.get(name, default), name)
    for name, default in [('train_split_ratio', .8), ('val_split_ratio', .1), ('test_split_ratio', .1)]:
        c[name] = _number(c.get(name, default), name, inclusive=True)
    if not math.isclose(sum(c[k] for k in ('train_split_ratio', 'val_split_ratio', 'test_split_ratio')), 1.0):
        raise ValueError('Train/validation/test split ratios must sum to 1')
    c['resume'] = _boolean(c.get('resume', False), 'resume')
    name = c.setdefault('dataset_name', 'allmerge_expert_trucksim_k2_v1' if mode == 'trucksim' else 'allmerge_expert')
    if not isinstance(name, str) or not name.strip() or '/' in name or '\\' in name or ':' in name or name in ('.', '..'):
        raise ValueError('dataset_name must be a nonempty directory name')
    c['output_root'] = str(resolve_path(c.get('output_root', 'outputs/expert_dataset')))
    cfg['conditions']['copy_instances_beside_source_dll'] = _boolean(
        cfg['conditions'].get('copy_instances_beside_source_dll', False), 'copy_instances_beside_source_dll')
    if mode == 'trucksim':
        cfg['koopman_collection_config'] = str(resolve_path(
            cfg.get('koopman_collection_config', 'configs/koopman/collect.yaml')))
    return cfg


def collection_args(cfg, check_only=False):
    """Adapt validated YAML to the existing collector and batch runner APIs."""
    c = cfg['collection']
    values = dict(c)
    values.update(
        vehicle_model=cfg['vehicle_model'], scenarios=list(cfg['scenarios']), scenario='all',
        output_root=Path(c['output_root']), duration=c['episode_duration_s'],
        retries=c['worker_retries'], check_only=check_only,
        prepare_dll_instances=cfg['conditions']['copy_instances_beside_source_dll'],
        copy_trucksim_dll_instances=cfg['conditions']['copy_instances_beside_source_dll'],
        koopman_config=Path(cfg['koopman_collection_config']) if cfg['vehicle_model'] == 'trucksim' else None,
        trucksim_koopman_config=None, engine_map_cache=None, episode_index=None,
        defer_split=False, allow_contract_mismatch=False, save_episode_videos=False, video_dir=None,
    )
    return argparse.Namespace(**values)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='YAML-controlled expert collection')
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(argv)
    return collection_args(load_config(args.config), args.check_only)
