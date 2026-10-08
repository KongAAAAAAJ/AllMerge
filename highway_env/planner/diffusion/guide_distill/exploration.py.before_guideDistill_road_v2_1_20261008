"""Independent multi-direction gradient ascent on sparse physical trajectories."""
from __future__ import annotations
import torch
from .reward_gradient import DIRECTIONS


def gradient_cosine(gradients):
    """Return pairwise cosines; NaN for a zero-norm gradient (not 0 degrees)."""
    flat=torch.stack([gradients[k].flatten(start_dim=1) for k in DIRECTIONS],dim=1)
    norm=torch.linalg.vector_norm(flat,dim=-1,keepdim=True)
    unit=flat/norm.clamp_min(1e-9)
    cosine=torch.matmul(unit,unit.transpose(-1,-2)).clamp(-1.,1.)
    pair_valid=(norm.squeeze(-1)>1e-8)
    mask=pair_valid[:,:,None] & pair_valid[:,None,:]
    return cosine.masked_fill(~mask,float('nan'))


def guide_all(xy, scorer, *, iters=8, step_m=0.12, trust_rms_m=0.75,
              max_point_move_m=1.25, backtracking_trials=6):
    """Four independent objectives; per-sample monotone proxy ascent.

    Samples without a good directional step remain unchanged. This guarantees
    monotonicity of the differentiable objective, not of the official W4 reward.
    """
    if iters<1 or step_m<=0 or trust_rms_m<=0 or max_point_move_m<=0:
        raise ValueError('Invalid guidance hyperparameters')
    x0=xy.detach().clone()
    out={};histories={}
    for name in DIRECTIONS:
        current=x0.clone()
        hist=[float(scorer.components(current)[name].mean().detach().cpu())]
        for _ in range(iters):
            x=current.detach().requires_grad_(True)
            score=scorer.components(x)[name]
            grad=torch.autograd.grad(score.sum(),x)[0].detach()
            norm=torch.linalg.vector_norm(grad.flatten(start_dim=1),dim=-1)
            unit=grad/norm.clamp_min(1e-9)[:,None,None]
            best=current.clone()
            best_score=score.detach().clone()
            accepted=torch.zeros(len(current),dtype=torch.bool,device=current.device)
            # Candidate selection and all projection operations are stop-gradient.
            with torch.no_grad():
                for k in range(backtracking_trials):
                    candidate=current+(step_m/(2.**k))*unit
                    change=candidate-x0
                    point_norm=torch.linalg.vector_norm(change,dim=-1).clamp_min(1e-9)
                    change=change*(max_point_move_m/point_norm).clamp(max=1.)[...,None]
                    rms=torch.sqrt(change.square().sum(dim=-1).mean(dim=-1)).clamp_min(1e-9)
                    change=change*(trust_rms_m/rms).clamp(max=1.)[:,None,None]
                    candidate=x0+change
                    candidate_score=scorer.components(candidate)[name]
                    update=(~accepted)&(candidate_score>=best_score-1e-8)&(norm>1e-8)
                    best=torch.where(update[:,None,None],candidate,best)
                    best_score=torch.where(update,candidate_score,best_score)
                    accepted=accepted|update
                    if bool(accepted.all()):break
            current=best.detach()
            hist.append(float(scorer.components(current)[name].mean().detach().cpu()))
        out[name]=current
        histories[name]=hist
    return out,histories
