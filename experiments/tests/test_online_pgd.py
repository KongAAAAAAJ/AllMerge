"""Online-PGD scheduler, bank-contract and shared Student-step CPU tests.

Run: python -m experiments.tests.test_online_pgd
(No GPU required; real teacher sampling and training need offpolicyEnv CUDA.)
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from highway_env.planner.diffusion.guide_distill import online_post_guide_distillation as online
from highway_env.planner.diffusion.guide_distill import post_guide_distillation as student


class OnlinePGDContractTest(unittest.TestCase):
    def test_refresh_scheduler(self):
        steps=[t for t in range(2001) if online.refresh_is_due(t,(t//1000)*1000,2000)]
        self.assertEqual(steps,[0,1000])
        self.assertFalse(online.refresh_is_due(2000,2000,2000))
        self.assertFalse(online.refresh_is_due(999,1000,2000))

    def test_bank_provenance_and_fail_closed(self):
        payload={'meta':{'teacher_source':'current_student','source_step':1000,
                'source_model_sha256':'abc','reward_config_sha256':'v6-hash',
                'noise_pairing_schema':'frozen_ddim_v1','paired_timesteps':[8,0],
                'teacher_count':1,'scenario':'all'},'states':[{'teachers':[{'mode':0}]}]}
        online.validate_teacher_bank(payload,source_step=1000,source_sha256='abc',
            reward_config_sha256='v6-hash',scenario='all')
        with self.assertRaisesRegex(RuntimeError,'source_model_sha256'):
            online.validate_teacher_bank(payload,source_step=1000,source_sha256='new-student',
                reward_config_sha256='v6-hash',scenario='all')
        with self.assertRaisesRegex(RuntimeError,'reward_config_sha256'):
            online.validate_teacher_bank(payload,source_step=1000,source_sha256='abc',
                reward_config_sha256='old-reward',scenario='all')
        payload['states']=[]
        with self.assertRaisesRegex(RuntimeError,'no valid'):
            online.validate_teacher_bank(payload,source_step=1000,source_sha256='abc',
                reward_config_sha256='v6-hash',scenario='all')

    def test_parameter_digest_includes_current_weights_and_scalar_buffers(self):
        model=torch.nn.Linear(2,1)
        model.register_buffer('scalar_buffer',torch.tensor(3,dtype=torch.long))
        h0=online.parameter_digest(model)
        self.assertEqual(h0,online.parameter_digest(model))
        with torch.no_grad():model.weight.add_(1)
        self.assertNotEqual(h0,online.parameter_digest(model))

    def test_teacher_refresh_round_path_is_immutable(self):
        path=Path('outputs/policy.pt')
        self.assertNotEqual(online._teacher_path(path,0,0),online._teacher_path(path,1,1000))
        self.assertTrue(str(online._teacher_path(path,1,1000)).endswith('round_0001_step_001000.pt'))

    def test_student_step_reuses_optimizer_and_checks_replay_group(self):
        model=torch.nn.Linear(1,1)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
        entry={'scenario':'straight','features':{'coarse_trajectories':torch.zeros(1,1,1)},
               'teachers':[{'target':torch.ones(1), 'mode':0,'group_id':3,
                            'paired_replay':{'source_group_id':3}}]}
        bank={'states':[entry]}
        rng=np.random.default_rng(0)
        original_loss=student.paired_replay_loss
        def fake_loss(m, features, target, teacher_mode, replay, *, early_stage_weight):
            loss=(m(torch.ones(1,1))-target).square().mean()
            return {'loss':loss,'trajectory_regression_loss':loss,
                    'terminal_regression_loss':loss,'early_regression_loss':loss*0,
                    'trajectory_classification_loss':loss*0,
                    'target_mode':torch.tensor([teacher_mode])}
        try:
            student.paired_replay_loss=fake_loss
            before=online.parameter_digest(model)
            step1=student.student_train_step(model,optimizer,list(model.parameters()),bank,rng,torch.device('cpu'))
            self.assertTrue(np.isfinite(step1['loss']))
            self.assertNotEqual(before,online.parameter_digest(model))
            self.assertTrue(optimizer.state)  # AdamW moments initialized
            step2=student.student_train_step(model,optimizer,list(model.parameters()),bank,rng,torch.device('cpu'))
            self.assertTrue(np.isfinite(step2['loss']))
            self.assertEqual(len(optimizer.state),2)
            entry['teachers'][0]['paired_replay']['source_group_id']=99
            with self.assertRaisesRegex(RuntimeError,'group ID mismatch'):
                student.student_train_step(model,optimizer,list(model.parameters()),bank,rng,torch.device('cpu'))
        finally:
            student.paired_replay_loss=original_loss


if __name__=='__main__':
    unittest.main()
