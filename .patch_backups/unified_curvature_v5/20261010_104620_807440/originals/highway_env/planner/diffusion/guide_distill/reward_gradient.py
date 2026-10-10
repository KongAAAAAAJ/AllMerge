"""Unified task-reward guidance. No auxiliary objectives or directional branches.

Existing production `progress_comfort` is the only optimized objective:
    R = task_progress_weight * progress_score
      - task_comfort_weight * comfort_penalty
      + task_road_weight * road_boundary_reward.
All terms use the current worktree's cubic-dense scorer and road boundary model.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np
import torch

from highway_env.planner.diffusion.trajectory_mode_reward.config import TrajectoryModeRewardConfig
from highway_env.planner.diffusion.grpo.task_reward import task_reward_from_w4_result
from highway_env.planner.diffusion.trajectory_mode_reward.spline_dense import evaluate_dense_spline, task_comfort_from_dense

# Kept as 'balanced' for existing probe/test compatibility, but this is the ONLY
# guidance direction and is exactly the full production task objective.
DIRECTIONS = ('balanced',)


@dataclass
class DifferentiableTaskReward:
    config: TrajectoryModeRewardConfig
    device: torch.device
    road_field: object | None = None
    role_pose: object | None = None

    def __post_init__(self):
        cfg = self.config
        required = ('progress_norm_m','task_progress_weight','task_comfort_weight',
                    'task_road_weight','trajectory_dt_s','interpolation_dt_s')
        absent = [k for k in required if not hasattr(cfg,k)]
        if absent:
            raise RuntimeError('V4 unified reward config missing: '+str(absent))
        self.semantics = {
            'reward_type': 'progress_comfort',
            'guidance': 'balanced (unified task reward ONLY)',
            'progress_norm_m': float(cfg.progress_norm_m),
            'weights': {'progress': float(cfg.task_progress_weight),
                        'comfort': float(cfg.task_comfort_weight),
                        'road': float(cfg.task_road_weight)},
            'spline': 'ClampedCubicTrajectorySpline 10 Hz',
        }
        self.gradient_fallback_counts = {}

    def bind_road(self, road_field, role_pose, *, balanced_road_weight=None):
        # Legacy caller may pass the weight; it must never override the task.
        if balanced_road_weight is not None and abs(float(balanced_road_weight)-float(self.config.task_road_weight))>1e-12:
            raise ValueError('Guidance road weight must match current task reward config')
        self.road_field = road_field
        self.role_pose = role_pose

    def components(self, xy: torch.Tensor) -> Dict[str, torch.Tensor]:
        if xy.ndim != 3 or tuple(xy.shape[1:]) != (8,2) or not bool(torch.isfinite(xy).all()):
            raise ValueError('Expected finite XY control points [N,8,2]')
        if self.road_field is None:
            raise RuntimeError('Bind road field before unified reward evaluation')
        cfg = self.config
        dense, _, _ = evaluate_dense_spline(xy, cfg)
        progress = (xy[:,-1,0]/float(cfg.progress_norm_m)).clamp(0.,1.)
        comfort = task_comfort_from_dense(dense, cfg)
        road = self.road_field.components(xy, self.role_pose, cfg)['road']
        native = float(cfg.task_progress_weight)*progress - float(cfg.task_comfort_weight)*comfort
        unified = native + float(cfg.task_road_weight)*road
        return {'balanced':unified, 'native_balanced':native, 'raw_progress':progress,
                'comfort_penalty':comfort, 'road':road}

    def _road_finite_difference(self, xy, *, step_m=1e-3):
        with torch.no_grad():
            out=torch.empty_like(xy)
            for t in range(xy.shape[1]):
                for axis in range(2):
                    xp=xy.detach().clone();xm=xy.detach().clone()
                    xp[:,t,axis]+=step_m;xm[:,t,axis]-=step_m
                    p=self.road_field.components(xp,self.role_pose,self.config)['road']
                    m=self.road_field.components(xm,self.role_pose,self.config)['road']
                    out[:,t,axis]=(p-m)/(2*step_m)
            return out

    def _repair_gradient(self,xy,name,grad):
        if name!='balanced':raise KeyError('Only unified reward guidance is supported')
        bad=~torch.isfinite(grad.flatten(start_dim=1)).all(dim=1)
        if not bool(bad.any()):return grad
        damaged=xy.detach()[bad]
        road_grad=self._road_finite_difference(damaged)
        with torch.enable_grad():
            x=damaged.clone().requires_grad_(True)
            # Native task terms must have valid analytic gradients.
            native=self.components(x)['native_balanced']
            native_grad=torch.autograd.grad(native.sum(),x)[0]
        recovered=native_grad+float(self.config.task_road_weight)*road_grad
        if not bool(torch.isfinite(recovered).all()):
            raise FloatingPointError('Non-finite unified guidance gradient persists after road FD fallback')
        grad=grad.clone();grad[bad]=recovered.detach()
        self.gradient_fallback_counts['balanced']=self.gradient_fallback_counts.get('balanced',0)+int(bad.sum())
        print(f'[road-gradient-fallback] direction=balanced samples={int(bad.sum())}',flush=True)
        return grad

    def gradient_of(self,xy,name='balanced'):
        if name!='balanced':raise KeyError('Only unified reward guidance is supported')
        with torch.enable_grad():
            x=xy.detach().clone().requires_grad_(True)
            score=self.components(x)['balanced']
            if not bool(torch.isfinite(score).all()):raise FloatingPointError('Non-finite unified reward score')
            grad=torch.autograd.grad(score.sum(),x)[0].detach()
            return self._repair_gradient(x,'balanced',grad).detach()

    def gradients(self,xy):
        return {'balanced':self.gradient_of(xy)}

    def validate_against_production(self,xy,result,*,role:int,mode:int,tol:float=.002):
        values=self.components(xy.detach())
        diffs={}
        for key,ours in [('progress_score','raw_progress'),('comfort_penalty','comfort_penalty'),
                         ('road_boundary_reward','road')]:
            if key not in result.components:raise RuntimeError(f'Production scorer missing {key}')
            actual=torch.as_tensor(np.asarray(result.components[key])[role,mode,:],device=xy.device,dtype=xy.dtype)
            diffs[key]=float((actual-values[ours].detach()).abs().max())
        native=task_reward_from_w4_result(result,context={'config':self.config},
                   device=xy.device,dtype=xy.dtype,reward_type='progress_comfort')[role,:,mode]
        diffs['task_reward']=float((native-values['balanced'].detach()).abs().max())
        if not all(np.isfinite(x) for x in diffs.values()) or max(diffs.values())>tol:
            raise RuntimeError('CURRENT REWARD PARITY FAILED '+str(diffs))
        return diffs
