"""A2 / B1 / B2 teacher-only optimization; never changes DDIM mean GRPO RGG."""
from __future__ import annotations
import torch


def guide_teacher(xy, reward, *, strategy='b2_warm_search', iterations=20,
                  step_m=.4,max_x_m=10.,max_y_m=3.):
    if strategy not in ('a2_v2','b1_adaptive','b2_warm_search'):
        raise ValueError('Unknown teacher strategy '+str(strategy))
    base=xy.detach().clone()
    weight=torch.linspace(0,1,base.shape[1],device=base.device,dtype=base.dtype).pow(1.7)
    seeds=[base]
    if strategy=='b2_warm_search':
        for longitudinal in (-8.,-4.,4.,8.):
            z=base.clone();z[...,0]+=longitudinal*weight
            seeds.append(z)
        for lateral in (-2.,2.):
            z=base.clone();z[...,1]+=lateral*weight
            seeds.append(z)
    outputs=[]
    for initial in seeds:
        current=initial.detach().clone()
        for j in range(iterations):
            grad=reward.gradient_of(current)
            x=grad[...,0];y=grad[...,1]
            if strategy=='a2_v2':
                norm=torch.linalg.vector_norm(grad.flatten(start_dim=1),dim=-1).clamp_min(1e-9)
                change=step_m*grad/norm[:,None,None]
            else:
                nx=torch.linalg.vector_norm(x,dim=-1).clamp_min(1e-8)
                ny=torch.linalg.vector_norm(y,dim=-1).clamp_min(1e-8)
                change=torch.stack((step_m*x/nx[:,None],(.35 if strategy=='b2_warm_search' else .25)*y/ny[:,None]),dim=-1)
            current_score=reward.components(current)['balanced'].detach()
            accepted=current
            for fraction in (1.,.5,.25,.125):
                candidate=current+change*fraction
                delta=candidate-base
                delta=torch.stack((delta[...,0].clamp(-max_x_m,max_x_m),
                                   delta[...,1].clamp(-max_y_m,max_y_m)),dim=-1)
                candidate=base+delta
                score=reward.components(candidate)['balanced'].detach()
                improvement=score>current_score+1e-7
                if improvement.any():
                    accepted=torch.where(improvement[:,None,None],candidate,accepted)
                    current_score=torch.where(improvement,score,current_score)
            current=accepted.detach()
        outputs.append(current)
    scores=torch.stack([reward.components(o)['balanced'].detach() for o in outputs])
    selection=scores.argmax(dim=0)
    optimized=torch.stack(outputs)
    return optimized[selection,torch.arange(len(base),device=base.device)],{'candidate_count':len(seeds), 'score':scores.max(dim=0).values}
