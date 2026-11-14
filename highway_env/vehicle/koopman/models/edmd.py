"""Polynomial EDMDc / condition-parametric EDMDc baselines, 10 Hz.

Fixed dictionary of bias, normalized history state/control, and degree-two
monomials of the most recent physical state. Ridge regression identifies
linear controlled lifted dynamics. A second ridge regression learns the
TruckSim-supervised LTR decoder. Only training episodes are used for fitting.
"""
from __future__ import annotations
import numpy as np
import torch
import warnings
from torch import nn
from highway_env.vehicle.koopman.models.base_model import rollout_pose

NAMES=('EDMDc','Parametric EDMDc')


class PolynomialEDMDc(nn.Module):
    def __init__(self, variant, history_steps=5, normalization=None, condition_normalization=None,
                 ridge=1e-3, spectral_cap=1.02):
        super().__init__()
        if variant not in NAMES:raise ValueError(variant)
        self.variant=variant;self.history_steps=int(history_steps)
        self.ridge=float(ridge);self.spectral_cap=float(spectral_cap)
        d=1+5*self.history_steps+2*(self.history_steps-1)+15
        self.latent_dim=d
        normalization=normalization or {}
        condition_normalization=condition_normalization or {}
        for key,n,zero in [('state_mean',5,0.),('state_std',5,1.),('control_mean',2,0.),('control_std',2,1.),
                          ('condition_mean',2,0.),('condition_std',2,1.)]:
            value=normalization.get(key,condition_normalization.get(key,[zero]*n))
            self.register_buffer(key,torch.as_tensor(value,dtype=torch.float32).reshape(n).clone())
        for key in ('state_std','control_std','condition_std'):
            getattr(self,key).clamp_(min=1e-5)
        self.register_buffer('A',torch.eye(d))
        self.register_buffer('B',torch.zeros(d,2))
        self.register_buffer('A_payload',torch.zeros(d,d))
        self.register_buffer('A_friction',torch.zeros(d,d))
        self.register_buffer('B_payload',torch.zeros(d,2))
        self.register_buffer('B_friction',torch.zeros(d,2))
        self.register_buffer('ltr_C',torch.zeros(d))
        self.register_buffer('ltr_C_payload',torch.zeros(d))
        self.register_buffer('condition_min',torch.full((2,),float('-inf')))
        self.register_buffer('condition_max',torch.full((2,),float('inf')))

    def lift(self,state_hist,control_hist):
        b=len(state_hist)
        if state_hist.shape!=(b,self.history_steps,5) or control_hist.shape!=(b,self.history_steps-1,2):
            raise ValueError('EDMD history shape mismatch')
        s=(state_hist-self.state_mean)/self.state_std
        u=(control_hist-self.control_mean)/self.control_std
        recent=s[:,-1]
        polynomials=torch.stack([recent[:,i]*recent[:,j] for i in range(5) for j in range(i,5)],dim=-1)
        return torch.cat((torch.ones(b,1,device=s.device,dtype=s.dtype),s.flatten(1),u.flatten(1),polynomials),dim=-1)

    def matrices(self,conditions):
        A=self.A.unsqueeze(0).expand(len(conditions),-1,-1)
        B=self.B.unsqueeze(0).expand(len(conditions),-1,-1)
        if self.variant=='Parametric EDMDc':
            c=(conditions.clamp(min=self.condition_min,max=self.condition_max)-self.condition_mean)/self.condition_std
            A=A+c[:,0,None,None]*self.A_payload+c[:,1,None,None]*self.A_friction
            B=B+c[:,0,None,None]*self.B_payload+c[:,1,None,None]*self.B_friction
        return A,B

    def predict_all(self,state_hist,control_hist,future_controls,initial_pose=None,dt=0.1,conditions=None):
        n=len(state_hist)
        if conditions is None:conditions=state_hist.new_zeros(n,2)
        z=self.lift(state_hist,control_hist)
        A,B=self.matrices(conditions)
        all_z=[z]
        for j in range(future_controls.shape[1]):
            u=(future_controls[:,j]-self.control_mean)/self.control_std
            z=torch.bmm(A,z.unsqueeze(-1)).squeeze(-1)+torch.bmm(B,u.unsqueeze(-1)).squeeze(-1)
            all_z.append(z)
        z=torch.stack(all_z,dim=1)
        idx=1+(self.history_steps-1)*5
        states=z[:,:,idx:idx+5]*self.state_std+self.state_mean
        ltr=(z*self.ltr_C).sum(-1)
        if self.variant=='Parametric EDMDc':
            c=(conditions[:,0].clamp(min=self.condition_min[0],max=self.condition_max[0])-self.condition_mean[0])/self.condition_std[0]
            ltr=ltr+c[:,None]*(z*self.ltr_C_payload).sum(-1)
        result={'states':states,'ltr':ltr.clamp(-1.0,1.0),'latent':z}
        if initial_pose is not None:result['pose']=rollout_pose(states,initial_pose,dt)
        return result


def _ridge_fit_stats(gram, rhs, count, ridge):
    """Ridge solution from streaming sufficient statistics, no giant design matrix.

    Feature scaling is calculated from the accumulated column sums of squares.
    Equivalent to normal equations of the standardized full design matrix.
    """
    if count < 1:raise ValueError('Empty EDMD training regression')
    gram=np.asarray(gram,dtype=np.float64)
    rhs=np.asarray(rhs,dtype=np.float64)
    scale=np.maximum(np.sqrt(np.maximum(np.diag(gram),0.)/count),1e-6)
    gram_s=gram/scale[:,None]/scale[None,:]
    rhs_s=rhs/scale[:,None]
    penalty=float(ridge)*count
    reg=np.eye(gram.shape[0],dtype=np.float64)*penalty
    reg[0,0]*=.1
    try:beta=np.linalg.solve(gram_s+reg,rhs_s)
    except np.linalg.LinAlgError:beta=np.linalg.lstsq(gram_s+reg,rhs_s,rcond=None)[0]
    return beta/scale[:,None]


def fit_edmd(model, episodes):
    """One-step EDMDc identification using all TRAIN episodes (no val/test)."""
    gram=rhs=ltr_gram=ltr_rhs=None
    transitions=labels=0
    history=model.history_steps
    with torch.no_grad():
        for ep in episodes:
            if len(ep.states)<=history:continue
            # Work from full state/control recordings, NOT overlapping 40step windows.
            hist_s=np.stack([ep.states[k-history+1:k+1] for k in range(history-1,len(ep.states)-1)])
            hist_u=np.stack([ep.controls[k-history+1:k] for k in range(history-1,len(ep.states)-1)])
            next_s=np.stack([ep.states[k-history+2:k+2] for k in range(history-1,len(ep.states)-1)])
            next_u=np.stack([ep.controls[k-history+2:k+1] for k in range(history-1,len(ep.states)-1)])
            z=model.lift(torch.from_numpy(hist_s),torch.from_numpy(hist_u)).numpy().astype(np.float64)
            zn=model.lift(torch.from_numpy(next_s),torch.from_numpy(next_u)).numpy().astype(np.float64)
            u=np.stack([ep.controls[k] for k in range(history-1,len(ep.states)-1)])
            u=(u-model.control_mean.numpy())/model.control_std.numpy()
            condition=np.asarray([float(ep.info['payload_kg']),float(ep.info['road_mu'])],dtype=np.float64)
            c=(condition-model.condition_mean.numpy())/model.condition_std.numpy()
            base=np.concatenate((z,u),axis=1)
            if model.variant=='Parametric EDMDc':
                features=np.concatenate((base,base*c[0],base*c[1]),axis=1)
            else:features=base
            if gram is None:
                gram=np.zeros((features.shape[1],features.shape[1]),np.float64)
                rhs=np.zeros((features.shape[1],model.latent_dim),np.float64)
            gram+=features.T@features
            rhs+=features.T@zn
            transitions+=len(features)
            mask=np.isfinite(ep.ltr[history-1:len(ep.states)-1])
            if mask.any():
                zz=z[mask];yl=ep.ltr[history-1:len(ep.states)-1][mask]
                if model.variant=='Parametric EDMDc':zz=np.concatenate((zz,zz*c[0]),axis=1)
                if ltr_gram is None:
                    ltr_gram=np.zeros((zz.shape[1],zz.shape[1]),np.float64)
                    ltr_rhs=np.zeros((zz.shape[1],1),np.float64)
                ltr_gram+=zz.T@zz
                ltr_rhs+=zz.T@yl[:,None]
                labels+=len(zz)
    if transitions==0 or labels==0:raise ValueError('EDMD needs training transitions and valid LTR supervision')
    observed_conditions=np.asarray([[float(e.info['payload_kg']),float(e.info['road_mu'])] for e in episodes],dtype=np.float32)
    model.condition_min.copy_(torch.from_numpy(observed_conditions.min(axis=0)))
    model.condition_max.copy_(torch.from_numpy(observed_conditions.max(axis=0)))
    beta=_ridge_fit_stats(gram,rhs,transitions,model.ridge)
    d=model.latent_dim
    A=beta[:d].T.copy();B=beta[d:d+2].T.copy()
    # Constant dictionary entry must stay invariant under every input/condition.
    A[0,:]=0.;A[0,0]=1.;B[0,:]=0.
    model.A.copy_(torch.as_tensor(A,dtype=torch.float32))
    model.B.copy_(torch.as_tensor(B,dtype=torch.float32))
    if model.variant=='Parametric EDMDc':
        for order,field in enumerate(('payload','friction')):
            off=(order+1)*(d+2)
            aa=beta[off:off+d].T.copy();bb=beta[off+d:off+d+2].T.copy()
            aa[0,:]=0.;bb[0,:]=0.
            getattr(model,'A_'+field).copy_(torch.tensor(aa,dtype=torch.float32))
            getattr(model,'B_'+field).copy_(torch.tensor(bb,dtype=torch.float32))
    # Preserve the deterministic delay-stack transitions and stabilize only
    # the learned lifted-dynamics rows for Parametric EDMDc.
    payload_values=sorted({float(ep.info['payload_kg']) for ep in episodes})
    friction_values=sorted({float(ep.info['road_mu']) for ep in episodes})

    if model.variant=='EDMDc':
        # Keep the original EDMDc baseline exactly on its legacy path.
        max_radius=0.0
        aa=model.A[1:,1:].numpy().astype(np.float64).copy()
        max_radius=float(max(abs(np.linalg.eigvals(aa))))
        ridge_pre_shift_radius=max_radius
        pre_stabilization_radius=max_radius
        multiplier=min(1.0,model.spectral_cap/max(max_radius,1e-12))
        if multiplier<1.0:
            model.A[1:,1:]*=multiplier
        stabilization_policy='legacy_uniform_shrink'
        stabilization_applied=multiplier<1.0
        post_stabilization_radius=max_radius*multiplier
        shift_structure_enforced=False
    else:
        # Snapshot the unconstrained Ridge matrices for diagnostics.
        ridge_A0=model.A.numpy().astype(np.float64).copy()
        ridge_Ap=model.A_payload.numpy().astype(np.float64).copy()
        ridge_Am=model.A_friction.numpy().astype(np.float64).copy()
        # z = [bias, state history, control history, quadratic(current state)].
        # The delay-stack shifts are deterministic identities, not quantities
        # that should be estimated by Ridge from highly correlated samples.
        h=model.history_steps
        d=model.latent_dim
        state_start=1
        control_start=1+5*h
        poly_start=control_start+2*(h-1)
        learned_rows=np.concatenate((
            np.arange(state_start+(h-1)*5,state_start+h*5,dtype=np.int64),
            np.arange(poly_start,d,dtype=np.int64),
        ))

        with torch.no_grad():
            # Constant dictionary entry.
            model.A[0,:].zero_()
            model.A[0,0]=1.0
            model.B[0,:].zero_()
            for key in ('A_payload','A_friction','B_payload','B_friction'):
                getattr(model,key)[0,:].zero_()

            # State-history shift: x[t+1] in the next stack equals x[t+2]
            # in the current stack for the first h-1 history slots.
            for t in range(h-1):
                for j in range(5):
                    row=state_start+t*5+j
                    col=state_start+(t+1)*5+j
                    for key in ('A','A_payload','A_friction'):
                        getattr(model,key)[row,:].zero_()
                    model.A[row,col]=1.0
                    for key in ('B','B_payload','B_friction'):
                        getattr(model,key)[row,:].zero_()

            # Control-history shift for the first h-2 slots.
            for t in range(max(h-2,0)):
                for j in range(2):
                    row=control_start+t*2+j
                    col=control_start+(t+1)*2+j
                    for key in ('A','A_payload','A_friction'):
                        getattr(model,key)[row,:].zero_()
                    model.A[row,col]=1.0
                    for key in ('B','B_payload','B_friction'):
                        getattr(model,key)[row,:].zero_()

            # The newest control-history slot is exactly the current input u_k.
            if h>1:
                for j in range(2):
                    row=control_start+(h-2)*2+j
                    for key in ('A','A_payload','A_friction'):
                        getattr(model,key)[row,:].zero_()
                    for key in ('B','B_payload','B_friction'):
                        getattr(model,key)[row,:].zero_()
                    model.B[row,j]=1.0

        # Record the radius before imposing exact shift rows. This is useful
        # for auditing the Ridge solution and comparing old checkpoints.
        condition_mean=model.condition_mean.numpy().astype(np.float64)
        condition_std=model.condition_std.numpy().astype(np.float64)
        A0=model.A.numpy().astype(np.float64).copy()
        Ap=model.A_payload.numpy().astype(np.float64).copy()
        Am=model.A_friction.numpy().astype(np.float64).copy()

        def _condition_radius(a0,ap,am):
            radius=0.0
            for payload in payload_values:
                for friction in friction_values:
                    c=(np.asarray([payload,friction],dtype=np.float64)-condition_mean)/condition_std
                    aa=a0+c[0]*ap+c[1]*am
                    eig=np.linalg.eigvals(aa[1:,1:])
                    radius=max(radius,float(np.max(np.abs(eig))))
            return radius

        ridge_pre_shift_radius=_condition_radius(ridge_A0,ridge_Ap,ridge_Am)
        pre_stabilization_radius=_condition_radius(A0,Ap,Am)

        def _radius_for_factor(factor):
            a0=A0.copy()
            ap=Ap.copy()
            am=Am.copy()
            a0[learned_rows,:]*=factor
            ap[learned_rows,:]*=factor
            am[learned_rows,:]*=factor
            return _condition_radius(a0,ap,am)

        if pre_stabilization_radius<=model.spectral_cap:
            multiplier=1.0
        else:
            # Find the largest shared factor that satisfies the spectral cap
            # over the complete observed payload x friction condition grid.
            # A coarse scan avoids assuming global monotonicity of rho(A(alpha));
            # bisection then refines the upper edge of the largest feasible cell.
            alpha_grid=np.linspace(0.0,1.0,65)
            radii=[_radius_for_factor(float(a)) for a in alpha_grid]
            feasible=[i for i,r in enumerate(radii) if r<=model.spectral_cap]
            if not feasible:
                raise RuntimeError(
                    'Parametric EDMDc shift-preserving stabilization could not '
                    'find a feasible learned-dynamics factor.'
                )
            i=max(feasible)
            if i==len(alpha_grid)-1:
                multiplier=1.0
            else:
                lo=float(alpha_grid[i])
                hi=float(alpha_grid[i+1])
                for _ in range(28):
                    mid=0.5*(lo+hi)
                    if _radius_for_factor(mid)<=model.spectral_cap:
                        lo=mid
                    else:
                        hi=mid
                multiplier=lo

        with torch.no_grad():
            model.A[learned_rows,:]=model.A[learned_rows,:]*multiplier
            model.A_payload[learned_rows,:]=model.A_payload[learned_rows,:]*multiplier
            model.A_friction[learned_rows,:]=model.A_friction[learned_rows,:]*multiplier

        post_stabilization_radius=_condition_radius(
            model.A.numpy().astype(np.float64).copy(),
            model.A_payload.numpy().astype(np.float64).copy(),
            model.A_friction.numpy().astype(np.float64).copy(),
        )
        stabilization_policy='shift_preserving_learned_rows'
        stabilization_applied=multiplier<0.999999
        shift_structure_enforced=True
        if post_stabilization_radius>model.spectral_cap+1e-6:
            raise RuntimeError(
                'Parametric EDMDc stabilization failed to meet spectral cap: '
                f'{post_stabilization_radius:.8f} > {model.spectral_cap:.8f}'
            )
        if stabilization_applied:
            warnings.warn(
                'Parametric EDMDc applied shift-preserving stabilization to '
                f'learned dynamics rows only (factor={multiplier:.6f}, '
                f'radius={pre_stabilization_radius:.6f}->{post_stabilization_radius:.6f}).',
                RuntimeWarning,
            )
        max_radius=pre_stabilization_radius

    ltr_beta=_ridge_fit_stats(ltr_gram,ltr_rhs,labels,model.ridge).reshape(-1)
    model.ltr_C.copy_(torch.as_tensor(ltr_beta[:d],dtype=torch.float32))
    if model.variant=='Parametric EDMDc':
        model.ltr_C_payload.copy_(torch.as_tensor(ltr_beta[d:],dtype=torch.float32))
    return {'transitions':transitions, 'ltr_labels':labels,
            'dictionary_dim':d,'ridge':model.ridge,
            'ridge_pre_shift_radius':float(ridge_pre_shift_radius),
            'pre_stabilization_radius':float(pre_stabilization_radius),
            'post_stabilization_radius':float(post_stabilization_radius),
            'stabilization_factor':float(multiplier),
            'stabilization_applied':bool(stabilization_applied),
            'stabilization_policy':stabilization_policy,
            'shift_structure_enforced':bool(shift_structure_enforced)}
