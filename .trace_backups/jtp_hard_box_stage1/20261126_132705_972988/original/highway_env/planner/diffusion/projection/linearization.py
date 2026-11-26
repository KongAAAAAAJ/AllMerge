"""Shared K2 sensitivities. Cache lifetime is one project() invocation."""
from __future__ import annotations
import torch
from .execution_adapter import KoopmanExecutionAdapter
from highway_env.vehicle.koopman.models.base_model import rollout_pose


def jacobian(fn, x):
    try:
        return torch.autograd.functional.jacobian(fn, x, vectorize=True)
    except (RuntimeError, NotImplementedError):
        return torch.autograd.functional.jacobian(fn, x, vectorize=False)


def _block_rows(j):
    # [N,T,D,2K] -> [N,T,D,2KN]. No cross-vehicle execution coupling.
    n, t, d, k = j.shape
    return torch.einsum('ntdk,nm->ntdmk', j, torch.eye(n, device=j.device, dtype=j.dtype)).reshape(n,t,d,n*k)


class K2StaticCache:
    def __init__(self, adapter, context, interpolation):
        self.model = adapter.vehicle_execution.koopman
        self.dt = adapter.vehicle_execution.dt
        self.pose0 = context['koopman_pose0_si']
        self.conditions = torch.stack((context['koopman_payload_kg'].reshape(-1), context['koopman_road_mu'].reshape(-1)), -1)
        self.P = interpolation.to(self.pose0)
        with torch.no_grad():
            self.A, self.B = self.model.dynamics_matrices(self.conditions)
            self.z0 = self.model.lift(context['koopman_state_hist_si'], context['koopman_control_hist_si'])
            self.Bsi = self.B / self.model.control_std
            n, d = self.z0.shape
            k = self.P.shape[1]
            s = self.z0.new_zeros(n,d,2*k)
            rows = [s]
            for w in self.P:
                injection = (self.Bsi[:,:,None,:] * w[None,None,:,None]).reshape(n,d,2*k)
                s = self.A @ s + injection
                rows.append(s)
            self.sensitivity = torch.stack(rows,1)

    def latent(self, controls):
        with torch.no_grad():
            u = torch.einsum('hk,nkc->nhc', self.P, controls)
            z = self.z0
            rows = [z]
            for step in u.unbind(1):
                z = (self.A @ z.unsqueeze(-1)).squeeze(-1) + (self.B @ self.model.normalize_control(step).unsqueeze(-1)).squeeze(-1)
                rows.append(z)
            return torch.stack(rows,1)

    def output_jacobians(self, controls):
        # Decoder outputs are pointwise in vehicle/time. Six batched VJPs
        # give the exact local derivatives without differentiating a full rollout.
        with torch.enable_grad():
            z = self.latent(controls).requires_grad_(True)
            states = self.model.decode(z)
            ltr = self.model.decode_ltr(z, self.conditions)
            y = torch.cat((states, ltr.unsqueeze(-1)), -1)
            local = torch.stack([torch.autograd.grad(y[...,i].sum(), z, retain_graph=i<5)[0] for i in range(6)], -2)
        with torch.no_grad():
            outj = local @ self.sensitivity
            sx = outj[:,:,:5]
            states = states.detach()
            pose = rollout_pose(states, self.pose0, self.dt)
            # Exact derivative of rollout_pose's trapezoidal integration.
            def integrate(v):
                return torch.cat((torch.zeros_like(v[:,:1]), torch.cumsum((v[:,:-1]+v[:,1:])*(.5*self.dt), 1)), 1)
            jyaw = integrate(sx[:,:,2])
            yaw = pose[:,:,2:3]
            c,s = yaw.cos(), yaw.sin()
            vx,vy = states[:,:,0:1], states[:,:,1:2]
            jvx,jvy = sx[:,:,0], sx[:,:,1]
            jwx = c*jvx-s*jvy+(-vx*s-vy*c)*jyaw
            jwy = s*jvx+c*jvy+(vx*c-vy*s)*jyaw
            jp = torch.stack((integrate(jwx),integrate(jwy),jyaw),2)
            return {'trajectory':_block_rows(jp), 'states':_block_rows(sx), 'ltr':_block_rows(outj[:,:,5:6]).squeeze(2)}


def shared_linearization(projector, controls, context, roads, current, selected, actors, cache):
    """Use one output sensitivity and one frozen association per outer iterate."""
    flat = controls.detach().flatten().requires_grad_(True)
    if cache is not None:
        outj = cache.output_jacobians(controls)
        method = 'structured_k2'
    else:
        # Existing public callable adapter contract, used for synthetic tests.
        # Not a legacy solver; both paths use identical grouped LS.
        keys = ('trajectory','states','ltr')
        sizes = [current.rollout[key].numel() for key in keys]
        def f(v):
            out = projector._rollout(v.reshape_as(controls), context)
            return torch.cat([out[key].reshape(-1) for key in keys])
        j = jacobian(f, flat)
        outj = {key: part.reshape(*current.rollout[key].shape,flat.numel()) for key,part in zip(keys,j.split(sizes,0))}
        method = 'callable_autograd'
    active_names = [name for name, ids in selected.items() if ids.numel()]
    def residual(v):
        dv = v - flat.detach()
        out = {key: current.rollout[key].detach() + (j.reshape(-1,flat.numel()) @ dv).reshape_as(current.rollout[key]) for key,j in outj.items()}
        _, groups, _ = projector._constraint_groups(v.reshape_as(controls),context,roads,current.associations,actors,rollout=out)
        return torch.cat([groups[name][selected[name]] for name in active_names])
    j = jacobian(residual, flat)
    jtau = outj['trajectory'].reshape(-1,flat.numel())
    weights = controls.new_tensor(projector.cfg['optimization']['trajectory_weights']).repeat(controls.shape[0]*(projector.horizon+1))
    opt=projector.cfg['optimization']; cfg=projector.cfg['projection']
    wu=controls.new_tensor(opt['control_weights']).repeat(controls.numel()//2)
    scales=controls.new_tensor(projector.cfg['control']['trust_scale']).repeat(controls.numel()//2)
    metric=jtau.T @ (weights[:,None]*jtau)+torch.diag(float(opt['lambda_u'])*wu+float(cfg['metric_regularization'])/scales.square())
    metric=(metric+metric.T)*.5
    sizes=[selected[name].numel() for name in active_names]
    rows=dict(zip(active_names,j.split(sizes,0)))
    return rows,metric,method
