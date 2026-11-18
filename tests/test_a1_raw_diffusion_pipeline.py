"""Online A1 through raw training, checkpoint reload, inference and evaluators."""
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import yaml

from tests.test_trace_fuel_aware_evaluation import _dataset, _k2_checkpoint, _save_planner, SMALL, CTRL, FUEL
from data_pipeline.raw_expert_dataset import RawExpertDataset
from highway_env.planner.diffusion.training.control import train, load_control_checkpoint, _make_loader
from highway_env.planner.diffusion.inference.control import ControlDiffusionRuntime


def training_data(root):
    _dataset(root)
    path = root / 'episodes/episode_000000.npz'
    with np.load(path) as data:
        values = {k:data[k].copy() for k in data.files}
    values['episode_index'][:] = 1
    np.savez(root/'episodes/episode_000001.npz', **values)
    manifest_path = root/'reports/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['split_ratios'] = [.5,.5,0]
    manifest['episode_files'].append(dict(name='episode_000001.npz',episode_index=1,samples=2))
    manifest_path.write_text(json.dumps(manifest))


def test_raw_training_checkpoint_and_runtime_parity(tmp_path, monkeypatch):
    root = tmp_path/'raw'
    training_data(root)
    a1_yaml = tmp_path/'a1.yaml'
    a1_yaml.write_text(Path('configs/anchors/a1.yaml').read_text())
    cfg = dict(output_space='control', model=SMALL, control=CTRL,
               execution={'model':'kinematic'},
               anchors={'strategy':'a1','a1_config':str(a1_yaml)},
               losses={'trajectory':{'type':'paper_mse','weight':1.}},
               training=dict(batch_size=2,num_workers=0,seed=0,lr=.0001))
    checkpoint = train(cfg,root=root,output_dir=tmp_path/'train',max_epochs=1,max_steps=1,device='cpu')
    model,payload = load_control_checkpoint(checkpoint)
    saved = payload['anchors_config']['resolved_a1']
    assert 'config_path' not in saved['a1']['candidate_generation']
    features,targets,_ = next(iter(_make_loader(root,'val',2,0,0,False)))
    assert 'coarse_controls' not in features and 'target_mode' not in targets
    expected = model._prepare_anchors(features)
    labels_changed = {**features,'future_trajectory_dense':torch.full((2,40,2),999.)}
    actual = model._prepare_anchors(labels_changed)
    torch.testing.assert_close(expected['coarse_controls'],actual['coarse_controls'])

    # Neither A1 YAML nor the candidate YAML is needed after the checkpoint snapshot.
    a1_yaml.unlink()
    from highway_env.planner.adaptive_anchors import yaml as candidate_yaml
    with patch.object(candidate_yaml, 'safe_load', side_effect=AssertionError('unexpected YAML read')):
        loaded,_ = load_control_checkpoint(checkpoint)
    torch.testing.assert_close(expected['coarse_controls'],loaded._prepare_anchors(features)['coarse_controls'])
    runtime = ControlDiffusionRuntime(checkpoint,device='cpu',inference_steps=2)
    numpy_features = {k:v.numpy() for k,v in features.items()}
    roads = {k:numpy_features.pop(k) for k in RawExpertDataset.ROAD_KEYS}
    with patch.object(runtime.model.anchor_manager,'select',wraps=runtime.model.anchor_manager.select) as select:
        prediction = runtime.infer(numpy_features,a1_road_geometry=roads)
        assert select.call_count == 1
    assert prediction['control'].shape == (2,40,2)
    torch.testing.assert_close(expected['coarse_controls'],runtime.model._prepare_anchors(features)['coarse_controls'])

    # Prepared anchors serve both the anchor baseline and diffusion without regeneration.
    from evaluation.trace_planning.open_loop import _anchor_prediction
    with patch.object(model.anchor_manager,'select',wraps=model.anchor_manager.select) as select:
        prepared = model._prepare_anchors(features)
        _anchor_prediction(model,prepared,{})
        model.infer_multimodal(prepared,context={},anchors_prepared=True)
        assert select.call_count == 1

    # Run the public open-loop evaluation with raw input and no A1 library directory.
    config = tmp_path/'diffusion.yaml'
    k2_path = tmp_path/'k2.pt'
    _k2_checkpoint(k2_path)
    k2_checkpoint = tmp_path/'k2_diffusion.pt'
    _save_planner(k2_checkpoint,k2_path,seed=0,weight=0.)
    config.write_text(yaml.safe_dump(dict(output_space='control',control=dict(
        model=SMALL,dynamics=CTRL,training={'dataset_root':str(root)},
        losses={'fuel':{'parameters':FUEL}}))))
    from evaluation.trace_planning.open_loop import evaluate
    with patch('evaluation.trace_planning.open_loop_visualization.plot_open_loop_figures'):
        evaluate(checkpoint=k2_checkpoint,config=config,dataset_root=root,split='val',
                 num_samples=2,batch_size=2,num_workers=0,device='cpu',seeds='0',
                 output_dir=tmp_path/'open_loop')
    assert (tmp_path/'open_loop/samples.csv').is_file()


def test_removed_dynamic_strategy_fails_explicitly():
    from highway_env.planner.diffusion.models.control import ControlDiffusionPlanner
    with pytest.raises(ValueError,match='Control anchor strategy'):
        ControlDiffusionPlanner(SMALL,CTRL,anchors_config={'strategy':'dynamic'})


def test_removed_a1_library_options_fail_explicitly(tmp_path):
    from evaluation.ablation.anchor_ablation.runner import run
    with pytest.raises(SystemExit) as error:
        run(['--stage','anchors','--methods','A1','--a1-dataset-root',str(tmp_path)])
    assert error.value.code == 2
    config=tmp_path/'eval.yaml'
    config.write_text(yaml.safe_dump(dict(dataset_root=str(tmp_path),stage='anchors',
        methods=['A1'],a1_dataset_root=str(tmp_path))))
    with pytest.raises(ValueError,match='a1_dataset_root is removed'):
        run(['--config',str(config),'--stage','anchors','--methods','A1'])
