"""Differentiable proxy for the *uploaded guideDistill* progress_comfort task.

Production evaluator uses LINEAR 10Hz interpolation and finite differences,
not cubic-spline motion quality. Physical curvature is a separate auxiliary
search direction and is NEVER included in the reported task reward.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import numpy as np
import torch

from highway_env.planner.diffusion.trajectory_mode_reward.config import TrajectoryModeRewardConfig
from highway_env.planner.diffusion.grpo.task_reward import task_reward_from_w4_result
from highway_env.planner.diffusion.trajectory_spline import ClampedCubicTrajectorySpline

DIRECTIONS = ('comfort', 'balanced', 'curvature_aux', 'road')


def _get_semantics(config):
    if not all(hasattr(config, k) for k in ('progress_norm_m','progress_weight',
                                           'comfort_weight','trajectory_dt_s','interpolation_dt_s')):
        raise RuntimeError('This probe requires the current progress_comfort task config. Refusing to guess.')
    return {'progress_component':'progress_score',
            'progress_norm_m':float(config.progress_norm_m),
            'weights':{'progress':float(config.progress_weight), 'comfort':float(config.comfort_weight)},
            'curvature_aux':'NOT a task reward term; diagnostic curvature minimization'}


@dataclass
class DifferentiableTaskReward:
    config: TrajectoryModeRewardConfig
    device: torch.device
    road_field: object | None = None
    role_pose: object | None = None
    balanced_road_weight: float = 0.25

    def bind_road(self, road_field, role_pose, *, balanced_road_weight=0.25):
        self.road_field = road_field
        self.role_pose = role_pose
        self.balanced_road_weight = float(balanced_road_weight)
        self.semantics['road_boundary_reward']='signed outer road union footprint margin (probe-only)'
        self.semantics['balanced_road_weight']=self.balanced_road_weight

    def __post_init__(self):
        self.semantics = _get_semantics(self.config)
        self.spline = ClampedCubicTrajectorySpline(
            horizon_s=8*float(self.config.trajectory_dt_s),
            sparse_dt=float(self.config.trajectory_dt_s),
            dense_dt=float(self.config.interpolation_dt_s),
        ).to(self.device)
        dt=float(self.config.trajectory_dt_s)
        dense_dt=float(self.config.interpolation_dt_s)
        source=np.arange(9,dtype=np.float64)*dt
        targets=np.arange(0., source[-1]+.5*dense_dt,dense_dt,dtype=np.float64)
        left=(np.searchsorted(source,targets,side='right')-1).clip(0,7)
        self.left=torch.tensor(left,dtype=torch.long,device=self.device)
        self.alpha=torch.tensor((targets-source[left])/(source[left+1]-source[left]),
                                dtype=torch.float64,device=self.device)

    def components(self, xy:torch.Tensor)->Dict[str,torch.Tensor]:
        if xy.ndim!=3 or xy.shape[1:]!=(8,2) or not torch.isfinite(xy).all():
            raise ValueError('Expected finite physical XY control points [N,8,2]')
        cfg=self.config
        origin=torch.zeros((xy.shape[0],1,2),dtype=xy.dtype,device=xy.device)
        knots=torch.cat([origin,xy],dim=1)
        left=self.left.to(xy.device)
        alpha=self.alpha.to(device=xy.device,dtype=xy.dtype)
        dense=knots[:,left,:]+alpha[None,:,None]*(knots[:,left+1,:]-knots[:,left,:])
        deltas=dense[:,1:,:]-dense[:,:-1,:]
        speed=torch.linalg.vector_norm(deltas,dim=-1)/float(cfg.interpolation_dt_s)
        accel=torch.diff(speed,dim=1,prepend=speed[:,:1])/float(cfg.interpolation_dt_s)

        # Match production _heading_from_xy_batch(): forward fill most recent
        # non-degenerate heading; unwrap; then np.diff(...,prepend=first).
        delta_norm=torch.linalg.vector_norm(deltas,dim=-1)
        valid=delta_norm>1e-6
        raw=torch.atan2(torch.where(valid,deltas[...,1],torch.zeros_like(deltas[...,1])),
                        torch.where(valid,deltas[...,0],torch.ones_like(deltas[...,0])))
        idx=torch.arange(raw.shape[1],device=xy.device)[None,:].expand_as(raw)
        last_valid=torch.cummax(torch.where(valid,idx,torch.zeros_like(idx)),dim=1).values
        heading=torch.gather(raw,1,last_valid)
        found=torch.cummax(valid.to(torch.int64),dim=1).values.bool()
        heading=torch.where(found,heading,torch.zeros_like(heading))
        dh=heading[:,1:]-heading[:,:-1]
        wrapped_dh=torch.atan2(torch.sin(dh),torch.cos(dh))
        yaw_rate=torch.cat((torch.zeros_like(heading[:,:1]),wrapped_dh),dim=1)/float(cfg.interpolation_dt_s)
        comfort=(.5*torch.mean(torch.abs(accel),dim=1)/8.
                 +.5*torch.mean(torch.abs(yaw_rate),dim=1)).clamp(0.,1.)
        progress=(xy[:,-1,0]/float(cfg.progress_norm_m)).clamp(0.,1.)
        wp=float(cfg.progress_weight);wc=float(cfg.comfort_weight)
        native_balanced=wp*progress-wc*comfort
        if self.road_field is None:
            # Keep production reward parity independent of exploration additions.
            road=torch.zeros_like(native_balanced)
        else:
            road=self.road_field.components(xy,self.role_pose,cfg)['road']
        balanced=native_balanced+self.balanced_road_weight*road

        # Geometry exploration ONLY, not part of task progress_comfort reward.
        init_v=torch.stack((xy[:,0,0].clamp_min(0.)/float(cfg.trajectory_dt_s),
                            torch.zeros_like(xy[:,0,0])),dim=-1)
        final_v=(xy[:,-1]-xy[:,-2])/float(cfg.trajectory_dt_s)
        _,velocity,acceleration=self.spline.evaluate(xy,origin[:,0,:],init_v,end_velocity_xy=final_v)
        speed_c=torch.linalg.vector_norm(velocity,dim=-1).clamp_min(.5)
        cross=velocity[...,0]*acceleration[...,1]-velocity[...,1]*acceleration[...,0]
        curvature=cross/speed_c.pow(3)
        curvature_aux=-(curvature.square().mean(dim=1)
                        +torch.relu(curvature.abs()-.02).square().mean(dim=1))
        out={'comfort':-wc*comfort,'balanced':balanced,'road':road,'native_balanced':native_balanced,
             'curvature_aux':curvature_aux,'raw_progress':progress,
             'comfort_penalty':comfort,
             'max_abs_curvature':curvature.abs().amax(dim=1)}
        return out

    def _road_finite_difference(self,xy,*,step_m=1e-3):
        """Fallback ONLY for failing samples in the road/combined objective.

        This is a central finite difference of the actual probe road reward;
        no W4 reward semantics or gradient values are silently substituted.
        Expensive, so it is used only after analytic autograd is non-finite.
        """
        with torch.no_grad():
            n=len(xy)
            out=torch.empty_like(xy)
            for t in range(xy.shape[1]):
                for c in range(xy.shape[2]):
                    xp=xy.detach().clone();xm=xy.detach().clone()
                    xp[:,t,c]+=step_m;xm[:,t,c]-=step_m
                    plus=self.road_field.components(xp,self.role_pose,self.config)['road']
                    minus=self.road_field.components(xm,self.role_pose,self.config)['road']
                    out[:,t,c]=(plus-minus)/(2.*step_m)
            return out

    def _repair_gradient(self,xy,name,grad):
        bad=~torch.isfinite(grad.flatten(start_dim=1)).all(dim=1)
        if not bool(bad.any()):return grad
        if name not in ('road','balanced'):
            raise FloatingPointError(f'Non-finite guidance gradient {name}; '
                                     f'bad samples={bad.nonzero().flatten().tolist()}')
        damaged=xy.detach()[bad]
        road_grad=self._road_finite_difference(damaged)
        if name=='balanced':
            # The native task component must remain analytically differentiable.
            with torch.enable_grad():
                x_native=damaged.clone().requires_grad_(True)
                score=self.components(x_native)['native_balanced']
                native_grad=torch.autograd.grad(score.sum(),x_native)[0]
            road_grad=native_grad+self.balanced_road_weight*road_grad
        if not bool(torch.isfinite(road_grad).all()):
            raise FloatingPointError(f'Non-finite guidance gradient {name} '
                                     'persists after finite-difference road fallback')
        grad=grad.clone()
        grad[bad]=road_grad.detach()
        if not hasattr(self,'gradient_fallback_counts'):
            self.gradient_fallback_counts={}
        self.gradient_fallback_counts[name]=self.gradient_fallback_counts.get(name,0)+int(bad.sum())
        print(f'[road-gradient-fallback] direction={name} '
              f'samples={int(bad.sum())} total={self.gradient_fallback_counts[name]} '
              '(finite-difference road term only)',flush=True)
        return grad

    def gradient_of(self,xy,name):
        if self.road_field is None:raise RuntimeError('Road field was not bound for road-guidance gradients')
        if name not in DIRECTIONS:raise KeyError(name)
        with torch.enable_grad():
            x=xy.detach().clone().requires_grad_(True)
            score=self.components(x)[name]
            if not bool(torch.isfinite(score).all()):
                raise FloatingPointError(f'Non-finite guidance score {name}; stop before optimization')
            grad=torch.autograd.grad(score.sum(),x)[0].detach()
            grad=self._repair_gradient(x,name,grad)
            return grad.detach()

    def gradients(self,xy):
        if self.road_field is None:raise RuntimeError('Road field was not bound for road-guidance gradients')
        with torch.enable_grad():
            x=xy.detach().clone().requires_grad_(True)
            values=self.components(x)
            result={}
            for i,name in enumerate(DIRECTIONS):
                if not bool(torch.isfinite(values[name]).all()):
                    raise FloatingPointError(f'Non-finite guidance score {name}')
                grad=torch.autograd.grad(values[name].sum(),x,retain_graph=i<len(DIRECTIONS)-1)[0].detach()
                result[name]=self._repair_gradient(x,name,grad).detach()
            return result

    def validate_against_production(self,xy,result,*,role:int,mode:int,tol:float=.002):
        values=self.components(xy.detach())
        actual_components=result.components
        diffs={}
        for key,ours in [('progress_score','raw_progress'),('comfort_penalty','comfort_penalty')]:
            if key not in actual_components:
                raise RuntimeError(f'Current production scorer missing {key}. STOP.')
            actual=torch.as_tensor(np.asarray(actual_components[key])[role,mode,:],
                                   device=xy.device,dtype=xy.dtype)
            diff=float((actual-values[ours].detach()).abs().max())
            diffs[key]=diff
        # Check the exact current task formula as well, not just its components.
        native=task_reward_from_w4_result(result,context={'config':self.config},
                                         device=xy.device,dtype=xy.dtype,
                                         reward_type='progress_comfort')[role,:,mode]
        diffs['task_reward']=float((native-values['native_balanced'].detach()).abs().max())
        if not all(np.isfinite(x) for x in diffs.values()) or max(diffs.values())>tol:
            raise RuntimeError('CURRENT REWARD PARITY FAILED '+str(diffs)+'; aborting before guidance.')
        return diffs
