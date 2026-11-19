"""Full diffusion continuation preserves optimizer, epoch history and early stopping."""
from copy import deepcopy
from unittest.mock import patch

import pytest
import torch

from tests.test_a1_raw_diffusion_pipeline import training_data
from tests.test_trace_fuel_aware_evaluation import SMALL, CTRL
from highway_env.planner.diffusion.models.control import ControlDiffusionPlanner
from highway_env.planner.diffusion.training.control import train, _restore_training_state


def config():
    return dict(output_space='control', model=SMALL, control=CTRL,
                execution={'model':'kinematic'}, anchors={'strategy':'a1'},
                losses={'trajectory':{'type':'paper_mse','weight':1.}},
                training=dict(batch_size=2,num_workers=0,seed=0,lr=.0001))


def test_resume_restores_adam_and_continues_epoch_and_history(tmp_path):
    root, output = tmp_path/'raw', tmp_path/'run'
    training_data(root)
    cfg = config()
    train(cfg,root=root,output_dir=output,max_epochs=1,max_steps=1,device='cpu')
    first = torch.load(output/'last.pt',map_location='cpu',weights_only=True)
    # The user's older epoch-8 checkpoints do not contain these new summary fields.
    first.pop('best_val_loss')
    first.pop('best_epoch')
    torch.save(first,output/'last.pt')
    model = ControlDiffusionPlanner(SMALL, CTRL,execution_config=cfg['execution'],
                                   loss_config=cfg['losses'],anchors_config=cfg['anchors'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.03)
    restored = _restore_training_state(output/'last.pt',model,optimizer,cfg)
    assert restored['epoch'] == 1
    assert optimizer.param_groups[0]['lr'] == .0001
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value,first['model_state'][key])
    assert max(float(state['step']) for state in optimizer.state.values()) == 1
    train(cfg,root=root,output_dir=output,max_epochs=2,max_steps=1,device='cpu',resume=output/'last.pt')
    last = torch.load(output/'last.pt',map_location='cpu',weights_only=True)
    assert last['epoch'] == 2
    assert [row['epoch'] for row in last['history']] == [1,2]
    assert last['history'][0] == first['history'][0]
    assert max(float(state['step']) for state in last['optimizer_state']['state'].values()) == 2
    assert last['best_val_loss'] == min(row['val_loss'] for row in last['history'])
    assert len((output/'loss.csv').read_text().splitlines()) == 3
    with pytest.raises(ValueError,match='total epoch limit'):
        train(cfg,root=root,output_dir=output,max_epochs=2,device='cpu',resume=output/'last.pt')
    changed = deepcopy(cfg)
    changed['losses']['trajectory']['weight'] = 2.
    with pytest.raises(ValueError,match='loss_config differs'):
        train(changed,root=root,output_dir=output,max_epochs=3,device='cpu',resume=output/'last.pt')


def metrics(loss):
    return dict(loss=loss,position_loss=loss,trajectory_loss=loss,
                ltr_loss=0.,fuel_loss=0.,progress_loss=0.)


def test_resume_preserves_best_file_and_early_stopping_in_new_directory(tmp_path):
    root, original, continued = tmp_path/'raw', tmp_path/'original', tmp_path/'continued'
    training_data(root)
    cfg = config()
    cfg['training']['patience'] = 2
    with patch('highway_env.planner.diffusion.training.control._run_pass',
               side_effect=[metrics(2.),metrics(.5),metrics(2.),metrics(1.)]):
        train(cfg,root=root,output_dir=original,max_epochs=2,device='cpu')
    best_bytes = (original/'best.pt').read_bytes()
    with patch('highway_env.planner.diffusion.training.control._run_pass',
               side_effect=[metrics(2.),metrics(1.)]) as run:
        train(cfg,root=root,output_dir=continued,max_epochs=6,device='cpu',resume=original/'last.pt')
    last = torch.load(continued/'last.pt',map_location='cpu',weights_only=True)
    assert run.call_count == 2
    assert last['epoch'] == 3 and last['best_epoch'] == 1
    assert (continued/'best.pt').read_bytes() == best_bytes
