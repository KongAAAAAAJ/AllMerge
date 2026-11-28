"""Shared K2 sensitivities. Cache lifetime is one project() invocation."""
from __future__ import annotations
import torch


def jacobian(fn, x):
    return torch.autograd.functional.jacobian(fn, x, vectorize=True)


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
            return {'states': outj[:, :, :5], 'ltr': outj[:, :, 5]}


def shared_linearization(projector, controls, context, current, selected, cache):
    """Differentiate only stability rows; no pose, geometry or control-rate Jacobians."""
    flat = controls.detach().flatten().requires_grad_(True)
    if cache is not None:
        outj = cache.output_jacobians(controls)
    else:
        def output(v):
            rollout = projector._rollout(v.reshape_as(controls), context)
            return torch.cat((rollout['states'].reshape(-1), rollout['ltr'].reshape(-1)))
        j = jacobian(output, flat)
        size = current.rollout['states'].numel()
        outj = {'states': j[:size].reshape(*current.rollout['states'].shape, flat.numel()),
                'ltr': j[size:].reshape(*current.rollout['ltr'].shape, flat.numel())}
    def residual(v):
        dv = v - flat.detach()
        out = {key: current.rollout[key].detach() +
               (j.reshape(-1, flat.numel()) @ dv).reshape_as(current.rollout[key])
               for key, j in outj.items()}
        return projector._stability(out, context)[selected]
    return jacobian(residual, flat)
