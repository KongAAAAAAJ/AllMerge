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



def _edmd_condition_grid(episodes):
    return sorted({(float(ep.info['payload_kg']), float(ep.info['road_mu']))
                   for ep in episodes})


def _edmd_condition_radius(a0, ap, am, model, conditions):
    mean = model.condition_mean.detach().cpu().numpy().astype(np.float64)
    std = model.condition_std.detach().cpu().numpy().astype(np.float64)
    radius = 0.0
    for payload, friction in conditions:
        raw_c = np.asarray([payload, friction], dtype=np.float64)
        c = (np.clip(raw_c, model.condition_min.detach().cpu().numpy(),
                     model.condition_max.detach().cpu().numpy()) - mean) / std
        aa = a0 + c[0] * ap + c[1] * am
        eig = np.linalg.eigvals(aa[1:, 1:])
        if not np.isfinite(eig).all():
            return float('inf')
        radius = max(radius, float(np.max(np.abs(eig))))
    return radius


def _edmd_apply_factor(model, raw_a, raw_ap, raw_am, learned_rows, factor):
    with torch.no_grad():
        model.A.copy_(torch.as_tensor(raw_a, dtype=model.A.dtype, device=model.A.device))
        model.A_payload.copy_(torch.as_tensor(raw_ap, dtype=model.A_payload.dtype, device=model.A_payload.device))
        model.A_friction.copy_(torch.as_tensor(raw_am, dtype=model.A_friction.dtype, device=model.A_friction.device))
        model.A[learned_rows, :] *= float(factor)
        model.A_payload[learned_rows, :] *= float(factor)
        model.A_friction[learned_rows, :] *= float(factor)



def _edmd_validation_metrics(model, episodes, config):
    """Evaluate candidate with the exact frozen-window protocol used by evaluate.py."""
    import math
    h = int(config.get('horizon_steps', 40))
    history = int(model.history_steps)
    dt = float(config.get('dt_s', 0.1))
    warmup_s = float(config.get('warmup_s', 1.0))
    stride = int(config.get('eval_stride', 5))
    batch_size = max(1, int(config.get('batch_size', 256)))
    if h < 1 or stride < 1 or dt <= 0:
        raise ValueError('Invalid Parametric EDMDc validation configuration')
    # Materialize the same episode-window starts as KoopmanWindowDataset.
    arrays = {k: [] for k in ('sh','uh','fu','ip','ts','tp','tl','cond','ep_idx')}
    ep_meta = []
    for ei, ep in enumerate(episodes):
        start = max(history - 1, int(math.ceil(warmup_s / dt - 1e-8)))
        last = len(ep.states) - 1 - h
        starts = list(range(start, last + 1, stride))
        if not starts:
            ep_meta.append((str(getattr(ep, 'id', ei)), float(ep.info['payload_kg']),
                            float(ep.info['road_mu']), 0))
            continue
        arrays['sh'].append(np.stack([ep.states[k-history+1:k+1] for k in starts]).astype(np.float32))
        arrays['uh'].append(np.stack([ep.controls[k-history+1:k] for k in starts]).astype(np.float32))
        arrays['fu'].append(np.stack([ep.controls[k:k+h] for k in starts]).astype(np.float32))
        arrays['ip'].append(np.stack([ep.poses[k] for k in starts]).astype(np.float32))
        arrays['ts'].append(np.stack([ep.states[k:k+h+1] for k in starts]).astype(np.float32))
        arrays['tp'].append(np.stack([ep.poses[k:k+h+1] for k in starts]).astype(np.float32))
        arrays['tl'].append(np.stack([ep.ltr[k:k+h+1] for k in starts]).astype(np.float32))
        arrays['cond'].append(np.repeat(np.asarray([[float(ep.info['payload_kg']),
                                                      float(ep.info['road_mu'])]], dtype=np.float32),
                                        len(starts), axis=0))
        arrays['ep_idx'].append(np.full(len(starts), ei, dtype=np.int32))
        ep_meta.append((str(getattr(ep, 'id', ei)), float(ep.info['payload_kg']),
                        float(ep.info['road_mu']), len(starts)))
    if not arrays['sh']:
        return {'valid': False, 'reason': 'no valid validation windows'}
    for key in arrays:
        arrays[key] = np.concatenate(arrays[key], axis=0)
    n_ep = len(episodes)
    # Accumulate exact episode-level sufficient statistics, matching evaluate.py.
    pe_sum = np.zeros((n_ep, h + 1), dtype=np.float64)
    fde_sum = np.zeros(n_ep, dtype=np.float64)
    yaw_sum = np.zeros(n_ep, dtype=np.float64)
    roll_sum = np.zeros(n_ep, dtype=np.float64)
    ltr_sum = np.zeros(n_ep, dtype=np.float64)
    ltr_count = np.zeros(n_ep, dtype=np.int64)
    window_count = np.zeros(n_ep, dtype=np.int64)
    device = model.A.device
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(arrays['sh']), batch_size):
            sl = slice(offset, offset + batch_size)
            ep_ids = arrays['ep_idx'][sl]
            def tensor(key):
                return torch.from_numpy(arrays[key][sl]).to(device)
            pred = model.predict_all(
                tensor('sh'), tensor('uh'), tensor('fu'),
                initial_pose=tensor('ip'), dt=dt, conditions=tensor('cond'))
            ps = pred['states'].detach().cpu().numpy()
            pp = pred['pose'].detach().cpu().numpy()
            pl = pred['ltr'].detach().cpu().numpy()
            ts = arrays['ts'][sl]
            tp = arrays['tp'][sl]
            tl = arrays['tl'][sl]
            if not (np.isfinite(ps).all() and np.isfinite(pp).all() and np.isfinite(pl).all()):
                return {'valid': False, 'reason': 'nonfinite validation rollout'}
            pos = np.linalg.norm(pp[..., :2] - tp[..., :2], axis=-1)
            yaw_sq = np.square(ps[:, 1:, 2] - ts[:, 1:, 2]).sum(axis=1)
            roll_sq = np.square(np.rad2deg(ps[:, 1:, 3] - ts[:, 1:, 3])).sum(axis=1)
            ltr_mask = np.isfinite(tl[:, 1:]) & np.isfinite(pl[:, 1:])
            ltr_sq = np.square(pl[:, 1:] - tl[:, 1:])
            np.add.at(pe_sum, ep_ids, pos)
            np.add.at(fde_sum, ep_ids, pos[:, -1])
            np.add.at(yaw_sum, ep_ids, yaw_sq)
            np.add.at(roll_sum, ep_ids, roll_sq)
            np.add.at(ltr_sum, ep_ids, np.where(ltr_mask, ltr_sq, 0.0).sum(axis=1))
            np.add.at(ltr_count, ep_ids, ltr_mask.sum(axis=1))
            window_count += np.bincount(ep_ids, minlength=n_ep)
    rows = []
    for ei, (ep_id, payload, friction, _) in enumerate(ep_meta):
        nw = int(window_count[ei])
        if nw == 0 or ltr_count[ei] == 0:
            continue
        curve = pe_sum[ei] / nw
        rows.append({
            'episode_id': ep_id, 'windows': nw,
            'ADE (m)': float(curve[1:].mean()),
            'FDE (m)': float(fde_sum[ei] / nw),
            'Yaw Rate RMSE (rad/s)': float(np.sqrt(yaw_sum[ei] / (nw * h))),
            'Roll RMSE (deg)': float(np.sqrt(roll_sum[ei] / (nw * h))),
            'LTR RMSE': float(np.sqrt(ltr_sum[ei] / ltr_count[ei])),
            'curve': curve.tolist(), 'payload_kg': payload, 'road_mu': friction,
        })
    if not rows:
        return {'valid': False, 'reason': 'no valid validation episodes'}
    keys = ('ADE (m)', 'FDE (m)', 'Yaw Rate RMSE (rad/s)', 'Roll RMSE (deg)', 'LTR RMSE')
    summary = {key: float(np.mean([r[key] for r in rows])) for key in keys}
    fdes = np.asarray([r['FDE (m)'] for r in rows], dtype=np.float64)
    group_fde = {}
    for row in rows:
        key = (row['payload_kg'], row['road_mu'])
        group_fde.setdefault(key, []).append(row['FDE (m)'])
    group_means = {f'{k[0]:g}|{k[1]:g}': float(np.mean(v)) for k, v in group_fde.items()}
    summary['P95 FDE (m)'] = float(np.quantile(fdes, 0.95))
    summary['Max condition mean FDE (m)'] = float(max(group_means.values()))
    return {'valid': True, 'summary': summary, 'episodes': rows,
            'condition_fde': group_means}


def _edmd_objective(metrics, scales):
    weights = {
        'ADE (m)': 0.30,
        'FDE (m)': 0.25,
        'P95 FDE (m)': 0.15,
        'Max condition mean FDE (m)': 0.10,
        'LTR RMSE': 0.10,
        'Yaw Rate RMSE (rad/s)': 0.05,
        'Roll RMSE (deg)': 0.05,
    }
    return float(sum(w * metrics[k] / max(scales[k], 1e-8) for k, w in weights.items()))


def _edmd_fit_rollout_ltr_decoder(model, episodes, ridge):
    """Fit LTR decoder on model-generated latent rollouts using TRAIN episodes only."""
    gram = rhs = None
    count = 0
    h = model.history_steps
    device = model.A.device
    with torch.inference_mode():
        for ep in episodes:
            if len(ep.states) <= h or not np.isfinite(ep.ltr[h-1:]).any():
                continue
            sh = torch.from_numpy(ep.states[:h].astype(np.float32)).unsqueeze(0).to(device)
            uh = torch.from_numpy(ep.controls[:h-1].astype(np.float32)).unsqueeze(0).to(device)
            cond = torch.tensor([[float(ep.info['payload_kg']), float(ep.info['road_mu'])]],
                                dtype=torch.float32, device=device)
            z = model.lift(sh, uh)
            aa, bb = model.matrices(cond)
            c = ((cond[:, 0].clamp(min=model.condition_min[0], max=model.condition_max[0])
                  - model.condition_mean[0]) / model.condition_std[0]).item()
            for k in range(h-1, len(ep.states)):
                label = float(ep.ltr[k])
                if np.isfinite(label):
                    zv = z[0].detach().cpu().numpy().astype(np.float64)
                    feat = np.concatenate((zv, zv * c)) if model.variant == 'Parametric EDMDc' else zv
                    if gram is None:
                        gram = np.zeros((len(feat), len(feat)), dtype=np.float64)
                        rhs = np.zeros((len(feat), 1), dtype=np.float64)
                    gram += np.outer(feat, feat)
                    rhs[:, 0] += feat * label
                    count += 1
                if k < len(ep.states) - 1:
                    u = (torch.from_numpy(ep.controls[k].astype(np.float32)).to(device)
                         - model.control_mean) / model.control_std
                    z = torch.bmm(aa, z.unsqueeze(-1)).squeeze(-1) + torch.bmm(
                        bb, u.reshape(1, 2, 1)).squeeze(-1)
    if count == 0:
        raise ValueError('No valid rollout-aligned LTR training labels')
    beta = _ridge_fit_stats(gram, rhs, count, ridge).reshape(-1)
    return beta[:model.latent_dim], (beta[model.latent_dim:] if model.variant == 'Parametric EDMDc'
                                     else np.zeros(model.latent_dim, dtype=np.float64))

def fit_edmd(model, episodes, validation_episodes=None, validation_config=None):
    """Fit EDMDc; Parametric EDMDc uses validation-selected shift-preserving stabilization."""
    gram=rhs=ltr_gram=ltr_rhs=None
    transitions=labels=0
    history=model.history_steps
    with torch.no_grad():
        for ep in episodes:
            if len(ep.states)<=history:continue
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
            features=np.concatenate((base,base*c[0],base*c[1]),axis=1) if model.variant=='Parametric EDMDc' else base
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

    # Preserve the original EDMDc baseline and its legacy stabilization behavior.
    if model.variant=='EDMDc':
        payload_values=sorted({float(ep.info['payload_kg']) for ep in episodes})
        friction_values=sorted({float(ep.info['road_mu']) for ep in episodes})
        max_radius=0.0
        aa=model.A[1:,1:].numpy().astype(np.float64).copy()
        max_radius=float(max(abs(np.linalg.eigvals(aa))))
        multiplier=min(1.0,model.spectral_cap/max(max_radius,1e-12))
        if multiplier<1.0:model.A[1:,1:]*=multiplier
        ltr_beta=_ridge_fit_stats(ltr_gram,ltr_rhs,labels,model.ridge).reshape(-1)
        model.ltr_C.copy_(torch.as_tensor(ltr_beta[:d],dtype=torch.float32))
        return {'transitions':transitions,'ltr_labels':labels,'dictionary_dim':d,'ridge':model.ridge,
                'pre_stabilization_radius':max_radius,'post_stabilization_radius':max_radius*multiplier,
                'stabilization_factor':multiplier,'stabilization_policy':'legacy_uniform_shrink',
                'shift_structure_enforced':False,'validation_selection':None}

    # Enforce exact deterministic history shifts before considering stabilization.
    h=model.history_steps
    state_start=1
    control_start=1+5*h
    poly_start=control_start+2*(h-1)
    learned_rows=np.concatenate((
        np.arange(state_start+(h-1)*5,state_start+h*5,dtype=np.int64),
        np.arange(poly_start,d,dtype=np.int64),
    ))
    with torch.no_grad():
        model.A[0,:].zero_();model.A[0,0]=1.0;model.B[0,:].zero_()
        for key in ('A_payload','A_friction','B_payload','B_friction'):
            getattr(model,key)[0,:].zero_()
        for t in range(h-1):
            for j in range(5):
                row=state_start+t*5+j;col=state_start+(t+1)*5+j
                for key in ('A','A_payload','A_friction'):getattr(model,key)[row,:].zero_()
                model.A[row,col]=1.0
                for key in ('B','B_payload','B_friction'):getattr(model,key)[row,:].zero_()
        for t in range(max(h-2,0)):
            for j in range(2):
                row=control_start+t*2+j;col=control_start+(t+1)*2+j
                for key in ('A','A_payload','A_friction'):getattr(model,key)[row,:].zero_()
                model.A[row,col]=1.0
                for key in ('B','B_payload','B_friction'):getattr(model,key)[row,:].zero_()
        if h>1:
            for j in range(2):
                row=control_start+(h-2)*2+j
                for key in ('A','A_payload','A_friction'):getattr(model,key)[row,:].zero_()
                for key in ('B','B_payload','B_friction'):getattr(model,key)[row,:].zero_()
                model.B[row,j]=1.0

    raw_a=model.A.detach().cpu().numpy().astype(np.float64).copy()
    raw_ap=model.A_payload.detach().cpu().numpy().astype(np.float64).copy()
    raw_am=model.A_friction.detach().cpu().numpy().astype(np.float64).copy()
    condition_grid=_edmd_condition_grid(list(episodes)+(list(validation_episodes) if validation_episodes else []))
    ridge_pre_shift_radius=_edmd_condition_radius(
        model.A.detach().cpu().numpy().astype(np.float64),
        model.A_payload.detach().cpu().numpy().astype(np.float64),
        model.A_friction.detach().cpu().numpy().astype(np.float64),model,condition_grid)
    pre_stabilization_radius=_edmd_condition_radius(raw_a,raw_ap,raw_am,model,condition_grid)

    # Fit the original teacher-forced LTR decoder first. It is retained as an
    # explicit comparator against the rollout-aligned decoder below.
    ltr_beta=_ridge_fit_stats(ltr_gram,ltr_rhs,labels,model.ridge).reshape(-1)
    teacher_ltr_c=ltr_beta[:d].copy()
    teacher_ltr_cp=(ltr_beta[d:].copy() if model.variant=='Parametric EDMDc'
                    else np.zeros(d,dtype=np.float64))

    def radius_at(alpha):
        aa=raw_a.copy();ap=raw_ap.copy();am=raw_am.copy()
        aa[learned_rows,:]*=float(alpha)
        ap[learned_rows,:]*=float(alpha)
        am[learned_rows,:]*=float(alpha)
        return _edmd_condition_radius(aa,ap,am,model,condition_grid)

    # Coarse scan then refine the largest feasible alpha. The scan does not
    # assume monotonicity of spectral radius with respect to alpha.
    alpha_scan=np.linspace(0.0,1.0,101)
    radii=[radius_at(float(a)) for a in alpha_scan]
    feasible=[i for i,r in enumerate(radii) if np.isfinite(r) and r<=model.spectral_cap]
    if not feasible:
        raise RuntimeError('No shift-preserving Parametric EDMDc factor satisfies spectral_cap.')
    imax=max(feasible)
    if imax==len(alpha_scan)-1:
        max_feasible=1.0
    else:
        lo=float(alpha_scan[imax]);hi=float(alpha_scan[imax+1])
        for _ in range(32):
            mid=0.5*(lo+hi)
            if radius_at(mid)<=model.spectral_cap:lo=mid
            else:hi=mid
        max_feasible=lo

    # Validation-driven search over all feasible factors. Coarse candidates
    # span [0,max_feasible]; a local refinement is added around the best one.
    coarse=np.linspace(max(0.0,max_feasible*0.55),max_feasible,13)
    candidates=set(float(x) for x in coarse)
    validation_episodes=list(validation_episodes or [])
    validation_config=dict(validation_config or {})
    candidate_records=[]
    if validation_episodes:
        for alpha in sorted(candidates):
            _edmd_apply_factor(model,raw_a,raw_ap,raw_am,learned_rows,alpha)
            radius=radius_at(alpha)
            if radius>model.spectral_cap+1e-6:continue
            result=_edmd_validation_metrics(model,validation_episodes,validation_config)
            if result.get('valid'):
                candidate_records.append({'alpha':alpha,'radius':radius,'metrics':result['summary']})
        if not candidate_records:
            raise RuntimeError('All validation-scored Parametric EDMDc candidates were invalid.')
        metric_keys=('ADE (m)','FDE (m)','P95 FDE (m)','Max condition mean FDE (m)',
                     'LTR RMSE','Yaw Rate RMSE (rad/s)','Roll RMSE (deg)')
        scales={k:min(r['metrics'][k] for r in candidate_records) for k in metric_keys}
        scored=[(_edmd_objective(r['metrics'],scales),r['alpha']) for r in candidate_records]
        best_score,best_alpha=min(scored,key=lambda x:(x[0],-x[1]))
        step=max_feasible/12.0
        refine=np.linspace(max(0.0,best_alpha-step),min(max_feasible,best_alpha+step),11)
        known={round(r['alpha'],10) for r in candidate_records}
        for alpha in refine:
            alpha=float(alpha)
            if round(alpha,10) in known:continue
            radius=radius_at(alpha)
            if radius>model.spectral_cap+1e-6:continue
            _edmd_apply_factor(model,raw_a,raw_ap,raw_am,learned_rows,alpha)
            result=_edmd_validation_metrics(model,validation_episodes,validation_config)
            if result.get('valid'):
                candidate_records.append({'alpha':alpha,'radius':radius,'metrics':result['summary']})
        scales={k:min(r['metrics'][k] for r in candidate_records) for k in metric_keys}
        scored=[(_edmd_objective(r['metrics'],scales),r['alpha']) for r in candidate_records]
        best_score,best_alpha=min(scored,key=lambda x:(x[0],-x[1]))
        for record in candidate_records:
            record['objective']=_edmd_objective(record['metrics'],scales)
        selection_enabled=True
    else:
        best_alpha=max_feasible
        best_score=None
        selection_enabled=False
    _edmd_apply_factor(model,raw_a,raw_ap,raw_am,learned_rows,best_alpha)
    post_stabilization_radius=radius_at(best_alpha)
    if post_stabilization_radius>model.spectral_cap+1e-6:
        raise RuntimeError('Selected Parametric EDMDc candidate violates spectral_cap.')

    # Compare teacher-forced LTR decoding against a decoder fitted on model-
    # generated TRAIN rollouts. The validation split selects the decoder only.
    model.ltr_C.copy_(torch.as_tensor(teacher_ltr_c,dtype=torch.float32))
    model.ltr_C_payload.copy_(torch.as_tensor(teacher_ltr_cp,dtype=torch.float32))
    teacher_ltr_metrics=None
    rollout_ltr_metrics=None
    if validation_episodes:
        teacher_eval=_edmd_validation_metrics(model,validation_episodes,validation_config)
        teacher_ltr_metrics=(teacher_eval['summary']['LTR RMSE'] if teacher_eval.get('valid') else None)
    rollout_c,rollout_cp=_edmd_fit_rollout_ltr_decoder(model,episodes,model.ridge)
    model.ltr_C.copy_(torch.as_tensor(rollout_c,dtype=torch.float32))
    model.ltr_C_payload.copy_(torch.as_tensor(rollout_cp,dtype=torch.float32))
    if validation_episodes:
        rollout_eval=_edmd_validation_metrics(model,validation_episodes,validation_config)
        rollout_ltr_metrics=(rollout_eval['summary']['LTR RMSE'] if rollout_eval.get('valid') else None)
        if (teacher_ltr_metrics is not None and
                (rollout_ltr_metrics is None or rollout_ltr_metrics >= teacher_ltr_metrics)):
            model.ltr_C.copy_(torch.as_tensor(teacher_ltr_c,dtype=torch.float32))
            model.ltr_C_payload.copy_(torch.as_tensor(teacher_ltr_cp,dtype=torch.float32))
            ltr_decoder_policy='teacher_forced_ridge'
        else:
            ltr_decoder_policy='rollout_aligned_ridge'
    else:
        ltr_decoder_policy='rollout_aligned_ridge'
    return {
        'transitions':transitions,'ltr_labels':labels,'dictionary_dim':d,'ridge':model.ridge,
        'ridge_pre_shift_radius':float(ridge_pre_shift_radius),
        'pre_stabilization_radius':float(pre_stabilization_radius),
        'post_stabilization_radius':float(post_stabilization_radius),
        'stabilization_factor':float(best_alpha),
        'stabilization_applied':bool(best_alpha<0.999999),
        'stabilization_policy':'validation_selected_shift_preserving_learned_rows',
        'shift_structure_enforced':True,
        'validation_selection':{
            'enabled':bool(selection_enabled),
            'objective':'0.30*ADE + 0.25*FDE + 0.15*P95FDE + 0.10*max_condition_mean_FDE + 0.10*LTR + 0.05*Yaw + 0.05*Roll; each metric normalized by the minimum among feasible candidates',
            'best_objective':None if best_score is None else float(best_score),
            'candidates':candidate_records,
        },
        'ltr_decoder_policy':ltr_decoder_policy,
        'ltr_teacher_forced_val_rmse':teacher_ltr_metrics,
        'ltr_rollout_aligned_val_rmse':rollout_ltr_metrics,
    }
