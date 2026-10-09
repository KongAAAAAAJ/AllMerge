"""Single unified reward guidance with per-sample backtracking and trust limits."""
from __future__ import annotations
import torch


def guide_all(xy, scorer, *, iters=8, step_m=0.12, trust_rms_m=0.75,
              max_point_move_m=1.25, backtracking_trials=6):
    """Return one guided result under key 'balanced' for API compatibility.

    The accepted sequence is monotone for the DIFFERENTIABLE current task reward,
    but not automatically safe or monotone after MATP projection.
    """
    if iters<1 or step_m<=0 or trust_rms_m<=0 or max_point_move_m<=0:
        raise ValueError('Invalid unified guidance hyperparameters')
    anchor=xy.detach().clone();current=anchor.clone()
    hist=[float(scorer.components(current)['balanced'].mean().detach().cpu())]
    for _ in range(iters):
        with torch.enable_grad():
            x=current.detach().clone().requires_grad_(True)
            values=scorer.components(x)['balanced']
            grad=torch.autograd.grad(values.sum(),x)[0].detach()
            grad=scorer._repair_gradient(x,'balanced',grad)
        norm=torch.linalg.vector_norm(grad.flatten(start_dim=1),dim=-1)
        unit=grad/norm.clamp_min(1e-9)[:,None,None]
        best=current.clone(); best_score=values.detach().clone()
        accepted=torch.zeros(len(xy),dtype=torch.bool,device=xy.device)
        with torch.no_grad():
            for k in range(backtracking_trials):
                candidate=current+(step_m/(2.**k))*unit
                displacement=candidate-anchor
                per_point=torch.linalg.vector_norm(displacement,dim=-1).clamp_min(1e-9)
                displacement=displacement*(max_point_move_m/per_point).clamp(max=1.)[...,None]
                rms=torch.sqrt(displacement.square().sum(dim=-1).mean(dim=-1)).clamp_min(1e-9)
                displacement=displacement*(trust_rms_m/rms).clamp(max=1.)[:,None,None]
                candidate=anchor+displacement
                score=scorer.components(candidate)['balanced']
                update=(~accepted)&(score>=best_score-1e-8)&(norm>1e-8)
                best=torch.where(update[:,None,None],candidate,best)
                best_score=torch.where(update,score,best_score)
                accepted|=update
                if bool(accepted.all()):break
        current=best.detach()
        hist.append(float(scorer.components(current)['balanced'].mean().detach().cpu()))
    return {'balanced':current},{'balanced':hist}
