"""Priority-grouped LS with metric half-space protection, no legacy solver."""
from __future__ import annotations
import torch
import math


def build_priority_groups(groups, cfg):
    # Fixed per project call; neutral q=1 defaults unless explicitly configured.
    q=cfg['priority_weights']; scores={name:float(q[name])*(1.+(float(v.clamp_min(0).max()) if v.numel() else 0.)) for name,v in groups.items()}
    ordered=sorted(scores, key=lambda name:-scores[name])
    count=int(cfg['group_count'])
    return [ordered[i*len(ordered)//count:(i+1)*len(ordered)//count] for i in range(count)],scores


def group_maxima(metrics, priority_groups):
    return [max((metrics['max_violation_by_group'][name] for name in group), default=0.) for group in priority_groups]


def accept_nonlinear(before, after, cfg):
    if any(not math.isfinite(x) for x in (*before,*after)):
        return False
    tol=float(cfg['nondegradation_tolerance']); improvement=float(cfg['minimum_violation_improvement'])
    return all(b<=a+tol for a,b in zip(before,after)) and any(b<a-improvement for a,b in zip(before,after))


def damped_ls(jac,residual,metric_chol,cfg,damping_multiplier=1.):
    # Original _joint_correction() Cholesky / damped Gram formulation.
    rhs=residual.clamp_min(0)
    weights=(1.+float(cfg['violation_weight_beta'])*rhs/rhs.max().clamp_min(1e-8)).sqrt()
    jw=weights[:,None]*jac; rw=weights*rhs
    inv_jt=torch.cholesky_solve(jw.T,metric_chol)
    gram=jw@inv_jt; gram=.5*(gram+gram.T)
    scale=float(gram.diagonal().mean().abs())
    damping=max(float(cfg['damping'])*max(scale,1e-6)*damping_multiplier,1e-8)
    eye=torch.eye(len(gram),device=gram.device,dtype=gram.dtype)
    for _ in range(int(cfg['damping_retries'])):
        mat=gram+damping*eye; chol,info=torch.linalg.cholesky_ex(mat)
        if int(info)==0:
            solved=torch.cholesky_solve(rw[:,None],chol).squeeze(-1)
            return -(inv_jt@solved),damping,'cholesky'
        damping*=10
    solved=torch.linalg.lstsq(gram+damping*eye,rw[:,None]).solution.squeeze(-1)
    return -(inv_jt@solved),damping,'lstsq'


def protect_direction(d, history, chol, cfg):
    """Cyclic metric half-space projection; reject if still infeasible.

    This is not an exact nearest-point projection onto the intersection.
    Every accepted group increment is checked against all protected rows.
    """
    tol=float(cfg['protection_tolerance'])
    if history.numel()==0:
        return d,0,0.,True
    inv=torch.cholesky_solve(history.T,chol)
    den=(history*inv.T).sum(-1).clamp_min(1e-20)
    for sweep in range(int(cfg['protection_max_sweeps'])+1):
        excess=history@d
        peak=float(excess.clamp_min(0).max())
        if peak<=tol:
            return d,sweep,peak,True
        if sweep==int(cfg['protection_max_sweeps']):
            break
        # Only rows currently violating the protection need processing.
        # Recheck each row after the preceding updates.
        for i in torch.nonzero(excess>tol,as_tuple=False).flatten().tolist():
            d=d-(history[i]@d).clamp_min(0)/den[i]*inv[:,i]
    return torch.zeros_like(d),sweep,peak,False


def solve_groups(controls, groups, selected, rows, metric, order, cfg, scale, trust, damping_multiplier):
    if cfg['solver_method']!='damped_least_squares':
        raise NotImplementedError('constrained_qp is reserved for Stage 2; select damped_least_squares')
    if not torch.isfinite(metric).all():
        return None,{'reason':'nonfinite_metric'}
    chol,info=torch.linalg.cholesky_ex(metric)
    if int(info)!=0:
        return None,{'reason':'invalid_metric'}
    p=torch.zeros_like(controls).flatten(); history=p.new_empty((0,p.numel())); details=[]
    # One shared factorization per outer iteration. Scalar shrink of group
    # increment preserves homogeneous non-degradation inequalities.
    bound=scale.repeat(controls.numel()//2)*min(trust,float(cfg['max_step_norm']))
    for names in order:
        use=[name for name in names if name in rows]
        if not use: continue
        j=torch.cat([rows[name] for name in use]); r=torch.cat([groups[name][selected[name]].detach() for name in use])
        residual=r+j@p
        active=residual>float(cfg['active_tolerance'])
        if bool(active.any()):
            d,damping,backend=damped_ls(j[active],residual[active],chol,cfg,damping_multiplier)
            if not torch.isfinite(d).all(): return None,{'reason':'nonfinite_group_step'}
            d,sweeps,peak,ok=protect_direction(d,history,chol,cfg)
            # Component-wise box for total normalized trust increment. Do not
            # independently clip nodes, which could undo the protection.
            eps=torch.finfo(d.dtype).tiny
            cap=torch.where(d>0,(bound-p)/d.clamp_min(eps),torch.where(d<0,(-bound-p)/d.clamp_max(-eps),torch.full_like(d,float('inf'))))
            alpha=cap.min().clamp(0,1); d=d*alpha
            # A feasible protected direction can still hurt its own group.
            old=float(residual.clamp_min(0).max()); new=float((residual+j@d).clamp_min(0).max())
            if new>old+float(cfg['nondegradation_tolerance']):
                d=torch.zeros_like(d); new=old; ok=False
            actual_peak=float((history@d).clamp_min(0).max()) if history.numel() else 0.
            if actual_peak>float(cfg['protection_tolerance']):
                d=torch.zeros_like(d); new=old; ok=False; actual_peak=0.
            details.append({'families':names,'active_rows':int(active.sum()),'backend':backend,'damping':damping,'protection_sweeps':sweeps,'protection_trial_peak':peak,'protection_peak':actual_peak,'protection_ok':ok,'trust_scale':float(alpha),'linear_max_before':old,'linear_max_after':new})
            p=p+d
        # All selected earlier rows remain protected, even if not repaired.
        history=torch.cat((history,j),0)
    return p.reshape_as(controls),{'groups':details,'jacobian_shape':[sum(len(j) for j in rows.values()),p.numel()],'metric_factorizations':1,'solver_method':cfg['solver_method']}
