"""Online-PGD guided-DDIM smoke contracts. No CUDA or model checkpoint required.

Run from the worktree: python -m experiments.tests.test_online_pgd_guidance
"""
from __future__ import annotations
import importlib.util
import re
import ast
import argparse
import contextlib
import io
from types import SimpleNamespace
import sys
import unittest
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[2]


def read(rel):
    return (ROOT/rel).read_text(encoding='utf-8')


class OnlinePGDGuidanceTest(unittest.TestCase):
    def test_random_mean_shift_has_matched_magnitude(self):
        path=ROOT/'highway_env/planner/diffusion/grpo/mean_guidance.py'
        spec=importlib.util.spec_from_file_location('_online_pgd_mean_guide',path)
        module=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=module
        spec.loader.exec_module(module)
        g=torch.randn(4,8,2,generator=torch.Generator().manual_seed(41))
        scale=torch.tensor([1.,1.])
        reward=module.MeanGuideSettings(mode='reward_gradient',step_m=.3)
        random=module.MeanGuideSettings(mode='random_gradient',step_m=.3)
        reward_shift=module._normalized_physical_gradient(g,scale,reward)
        random_shift=module._normalized_physical_gradient(g,scale,random,
            generator=torch.Generator().manual_seed(42))
        self.assertTrue(torch.allclose(reward_shift.abs(),random_shift.abs()))
        self.assertTrue(torch.allclose(reward_shift.square().sum(-1).mean(-1),
                                       random_shift.square().sum(-1).mean(-1)))
        self.assertFalse(torch.equal(reward_shift,random_shift))
        with self.assertRaises(ValueError):
            module.MeanGuideSettings(mode='invalid').validate()

    def test_train_entry_preserves_online_mode(self):
        s=read('train_grpo.py')
        self.assertIn("elif args.method=='post_guide_distillation':",s)
        self.assertIn("args.method == 'online_post_guide_distillation'",s)
        self.assertIn("args.output.parent / args.guide_mode / args.output.name",s)
        self.assertIn('"random_gradient"',s)
        self.assertNotIn("else:\n        args.guide_mode='none'",s)

    def test_current_student_guidance_receives_context_and_velocity(self):
        t=read('highway_env/planner/diffusion/guide_distill/teacher.py')
        self.assertIn('guidance=guidance,guide_model=model',t)
        self.assertIn('sampler.sample(features,generator=gen,context=context)',t)
        self.assertIn("'sampling_guide_mode':sampling_guide_mode",t)
        self.assertIn("'sampling_shift_rms_m':sampling_shift_rms",t)
        self.assertIn('if source_model is not None else',t)
        m=read('highway_env/planner/diffusion/grpo/mean_guidance.py')
        self.assertRegex(m, r"proxy\.bind_initial_velocity\(features\['ego_state'\]\[role\s*,\s*:2\]\)")

    def test_refresh_bank_and_resume_are_mode_aware(self):
        s=read('highway_env/planner/diffusion/guide_distill/online_post_guide_distillation.py')
        self.assertIn('sampling_guide_mode=args.guide_mode',s)
        self.assertIn('guide_mode=args.guide_mode',s)
        self.assertIn("if args.guide_mode != 'none':",s)
        self.assertIn("meta.get('sampling_guide_mode', 'none') != sampling_guide_mode",s)
        self.assertIn("'source_model_sha256'",s)

    def test_bank_validation_rejects_cross_mode_reuse(self):
        path='highway_env/planner/diffusion/guide_distill/online_post_guide_distillation.py'
        mod=ast.parse(read(path))
        fn=next(n for n in mod.body if isinstance(n,ast.FunctionDef) and n.name=='validate_teacher_bank')
        env={}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),path,'exec'),env)
        validate=env['validate_teacher_bank']
        payload={'meta':{'source_step':0,'source_model_sha256':'model',
                         'reward_config_sha256':'reward','scenario':'all',
                         'sampling_guide_mode':'random_gradient',
                         'noise_pairing_schema':'frozen_ddim_v1',
                         'paired_timesteps':[8,0]},'states':[{'teachers':[{}]}]}
        self.assertTrue(validate(payload,source_step=0,source_sha256='model',
                                 reward_config_sha256='reward',scenario='all',
                                 sampling_guide_mode='random_gradient'))
        with self.assertRaisesRegex(RuntimeError,'sampling guide_mode mismatch'):
            validate(payload,source_step=0,source_sha256='model',
                     reward_config_sha256='reward',scenario='all',
                     sampling_guide_mode='reward_gradient')

    def test_fingerprint_compatibility_for_none(self):
        path='highway_env/planner/diffusion/guide_distill/online_post_guide_distillation.py'
        mod=ast.parse(read(path))
        fn=next(n for n in mod.body if isinstance(n,ast.FunctionDef) and n.name=='_fingerprint')
        env={'teacher':SimpleNamespace(digest=lambda p:'fake-sha')}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),path,'exec'),env)
        args=SimpleNamespace(checkpoint='pretrain.pt',scenario='all',seed=0,
            group_size=48,lr=1e-6,pgd_early_stage_weight=0.0,
            pgd_online_refresh_interval=1000,pgd_online_refresh_states=32,
            pgd_guide_count=12,pgd_guide_iters=20,guide_step_m=0.3,
            pgd_strategy='b2_warm_search',guide_mode='none',guide_scope='selected',
            guide_max_x_m=20.,guide_max_y_m=6.)
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            target=root/'highway_env/planner/diffusion/trajectory_mode_reward/task_objective.py'
            target.parent.mkdir(parents=True)
            target.write_text('test')
            none=env['_fingerprint'](args,root)
            self.assertNotIn('guide_mode',none)
            args.guide_mode='reward_gradient'
            guided=env['_fingerprint'](args,root)
            self.assertEqual(guided['guide_mode'],'reward_gradient')
            self.assertNotEqual(none,guided)

    def test_actual_argparse_dispatch_and_output_isolation(self):
        import yaml
        path=ROOT/'train_grpo.py'
        mod=ast.parse(path.read_text(encoding='utf-8'))
        fn=next(n for n in mod.body if isinstance(n,ast.FunctionDef) and n.name=='parse_args')
        env={'argparse':argparse,'Path':Path,'SCENARIOS':{'straight':None,'curved':None,
             'merge_in':None,'merge_out':None},'torch':torch,'sys':sys,'__file__':str(path),
             'GRPOConfig':lambda:None,
             'load_config':lambda p:yaml.safe_load(Path(p).read_text(encoding='utf-8'))}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),env)
        old_argv=sys.argv[:]
        try:
            for requested,expected in [('grpo','none'),('reward_guided_grpo','reward_gradient'),
                ('post_guide_distillation','none'),('online_post_guide_distillation','random_gradient')]:
                sys.argv=['train_grpo.py','--method',requested,'--guide-mode','random_gradient']
                with contextlib.redirect_stdout(io.StringIO()):
                    args=env['parse_args']()
                self.assertEqual(args.guide_mode,expected)
                if requested=='online_post_guide_distillation':
                    self.assertEqual(args.output.parent.name,'random_gradient')
        finally:
            sys.argv=old_argv

    def test_config_has_single_switch(self):
        import yaml
        cfg=yaml.safe_load(read('configs/train/allmerge_posttrain.yaml'))
        self.assertIn(cfg['guide_mode'],('none','random_gradient','reward_gradient'))
        self.assertGreater(cfg['pgd_online_refresh_interval'],0)
        self.assertGreater(cfg['pgd_online_refresh_states'],0)
        self.assertEqual(cfg['task_reward_type'],'unified_v6')

if __name__=='__main__':unittest.main()
