"""Collection config validation and routing without loading native solvers."""
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
import numpy as np

from data_pipeline.expert_collection.config import ROOT, load_config, collection_args
from scripts.data.collect_expert_dataset import main


def write_config(tmp_path, cfg):
    path = tmp_path / 'collection.yaml'
    current = {'vehicle_model': 'trucksim', 'scenarios': ['straight'], **cfg}
    path.write_text(yaml.safe_dump(current), encoding='utf-8')
    return path


def test_paths_and_no_cwd_side_effect(tmp_path, monkeypatch):
    path = write_config(tmp_path, {'vehicle_model': 'trucksim', 'collection': {
        'output_root': 'outputs/expert_dataset', 'start_seed': 7}})
    monkeypatch.chdir(tmp_path)
    opts = collection_args(load_config(path))
    assert Path.cwd() == tmp_path
    assert opts.output_root == ROOT / 'outputs/expert_dataset'
    assert opts.koopman_config == ROOT / 'configs/koopman/collect.yaml'
    assert opts.start_seed == 7
    assert np.isclose(opts.ego_wheelbase_m, 5.5)


@pytest.mark.parametrize('cfg', [
    {'vehicle_model': 'unsupported'},
    {'scenarios': ['straight', 'straight']},
    {'collection': {'target_samples': 0}},
    {'collection': {'target_samples': True}},
    {'collection': {'resume': 'false'}},
    {'collection': {'timeout_s': float('nan')}},
    {'collection': {'train_split_ratio': .9}},
    {'vehicle_dimensions': {'wheelbase_m': 0.0}},
])
def test_invalid_config_rejected(tmp_path, cfg):
    with pytest.raises(ValueError):
        load_config(write_config(tmp_path, cfg))


def test_trucksim_entry_uses_yaml(tmp_path):
    path = write_config(tmp_path, {'vehicle_model': 'trucksim', 'scenarios': ['merge_in'],
        'collection': {'target_samples': 123, 'resume': True}})
    with patch('data_pipeline.expert_collection.runner.main', return_value=0) as run:
        assert main(['--config', str(path)]) == 0
    opts = run.call_args.args[0]
    assert opts.target_samples == 123
    assert opts.scenarios == ['merge_in'] and opts.resume


def test_retired_kinematic_collector_is_rejected(tmp_path):
    path = write_config(tmp_path, {'vehicle_model': 'kinematic'})
    with pytest.raises(ValueError, match='requires vehicle_model: trucksim'):
        main(['--config', str(path), '--check-only'])
