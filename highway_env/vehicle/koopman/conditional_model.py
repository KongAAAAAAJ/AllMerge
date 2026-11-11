"""Payload-/friction-conditioned neural Koopman K1-P, K1-F and K2.

Shared lifting Phi and chassis decoder Psi are inherited from K0. Only latent
A(theta), B(theta) and, where payload is permitted, the auxiliary LTR decoder
are conditioned on the operating condition. No condition leakage in K1-F.
"""
from __future__ import annotations
import torch
from torch import nn
from .base_model import NeuralKoopmanK0, rollout_pose

VARIANTS={'K1-P':(0,), 'K1-F':(1,), 'K2':(0,1)}


class ConditionalKoopman(NeuralKoopmanK0):
    def __init__(self, variant='K2', condition_mean=(0.0,0.5), condition_std=(1000.0,0.1),
                 condition_hidden=64, **kwargs):
        if variant not in VARIANTS:raise ValueError(f'Unknown conditional variant: {variant}')
        super().__init__(**kwargs)
        self.variant=variant
        indices=VARIANTS[variant]
        self.register_buffer('condition_mean',torch.as_tensor(condition_mean,dtype=torch.float32).reshape(2))
        self.register_buffer('condition_std',torch.as_tensor(condition_std,dtype=torch.float32).reshape(2).clamp_min(1e-5))
        self.condition_hidden=int(condition_hidden)
        d=self.latent_dim
        self.condition_net=nn.Sequential(nn.Linear(len(indices),self.condition_hidden),nn.SiLU(),
                                         nn.Linear(self.condition_hidden,d*(d+2)))
        nn.init.zeros_(self.condition_net[-1].weight)
        nn.init.zeros_(self.condition_net[-1].bias)
        if 0 in indices:
            self.ltr_head=nn.Sequential(nn.Linear(d+1,self.hidden_dim),nn.SiLU(),
                                        nn.Linear(self.hidden_dim,1))

    def _conditions(self, conditions, batch):
        if conditions is None or conditions.shape != (batch,2):
            raise ValueError(f'{self.variant} needs conditions [B,2] = [payload_kg, road_mu]')
        return (conditions-self.condition_mean)/self.condition_std

    def dynamics_matrices(self, conditions):
        c=self._conditions(conditions,len(conditions))
        delta=self.condition_net(c[:,VARIANTS[self.variant]])
        n=self.latent_dim
        delta_A=delta[:,:n*n].reshape(-1,n,n)
        delta_B=delta[:,n*n:].reshape(-1,n,2)
        return self.A[None]+delta_A, self.B[None]+delta_B

    def forward(self, state_hist, control_hist, future_controls, conditions=None):
        if future_controls.ndim!=3 or future_controls.shape[0]!=len(state_hist) or future_controls.shape[-1]!=2:
            raise ValueError('future_controls must be [B,H,2]')
        A,B=self.dynamics_matrices(conditions)
        z=self.lift(state_hist,control_hist)
        zs=[z]
        for j in range(future_controls.shape[1]):
            u=self.normalize_control(future_controls[:,j])
            z=torch.bmm(A,z.unsqueeze(-1)).squeeze(-1)+torch.bmm(B,u.unsqueeze(-1)).squeeze(-1)
            zs.append(z)
        latent=torch.stack(zs,dim=1)
        return self.decode(latent),latent

    def decode_ltr(self, latent, conditions=None):
        if 0 in VARIANTS[self.variant]:
            c=self._conditions(conditions,len(latent))
            payload=c[:,0:1,None].transpose(1,2).expand(-1,latent.shape[1],-1)
            x=torch.cat((latent,payload),dim=-1)
        else:x=latent
        return torch.tanh(self.ltr_head(x).squeeze(-1))

    def predict_all(self,state_hist,control_hist,future_controls,initial_pose=None,dt=0.1,conditions=None):
        states,latent=self.forward(state_hist,control_hist,future_controls,conditions)
        result={'states':states,'ltr':self.decode_ltr(latent,conditions),'latent':latent}
        if initial_pose is not None:result['pose']=rollout_pose(states,initial_pose,dt)
        return result

    def settings(self):
        return {**super().settings(),'variant':self.variant,'condition_hidden':self.condition_hidden}
