from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence
import time

import numpy as np
import torch


@dataclass
class SafeMPOTargetConfig:
    """Configuration for the finite-particle SafeMPO E-step.

    The implementation follows the linear-safety-function SafeMPO dual, using
    the G sampled trajectories as an empirical approximation of the old policy.
    A single set of dual variables is shared across all valid vehicle/mode
    groups in the batch, while q*(g|state) is normalized independently over G.
    """

    kl_epsilon: float = 0.10
    kappa: float = 10.0
    constraint_beta: float = 1.0
    lambda_init: float = 1.0
    lambda_min: float = 1e-6
    lambda_max: float = 10000.0
    nu_init: float = 1.0
    nu_min: float = 1e-5
    nu_max: float = 1000.0
    active_range_eps: float = 1e-6
    maxiter: int = 128
    ftol: float = 1e-9

    def validate(self) -> None:
        if self.kl_epsilon <= 0.0:
            raise ValueError("SafeMPO kl_epsilon must be > 0")
        if self.kappa <= 0.0:
            raise ValueError("SafeMPO kappa must be > 0")
        if self.constraint_beta <= 0.0:
            raise ValueError("SafeMPO constraint_beta must be > 0")
        if not 0.0 < self.lambda_min < self.lambda_max:
            raise ValueError("invalid SafeMPO lambda bounds")
        if not self.lambda_min <= self.lambda_init <= self.lambda_max:
            raise ValueError("SafeMPO lambda_init must lie inside lambda bounds")
        if not 0.0 < self.nu_min < self.nu_max:
            raise ValueError("invalid SafeMPO nu bounds")
        if not self.nu_min <= self.nu_init <= self.nu_max:
            raise ValueError("SafeMPO nu_init must lie inside nu bounds")
        if self.active_range_eps < 0.0:
            raise ValueError("SafeMPO active_range_eps must be >= 0")
        if self.maxiter < 1:
            raise ValueError("SafeMPO maxiter must be >= 1")
        if self.ftol <= 0.0:
            raise ValueError("SafeMPO ftol must be > 0")


@dataclass(frozen=True)
class SafeMPOTargetResult:
    target_q: torch.Tensor  # [B,G,M], normalized over G
    metrics: Dict[str, float]


class SafeMPOTargetBuilder:
    """Solve the low-dimensional multi-constraint SafeMPO dual on G particles."""

    def __init__(
        self,
        constraint_names: Sequence[str],
        *,
        config: SafeMPOTargetConfig | None = None,
    ) -> None:
        self.constraint_names = tuple(str(x) for x in constraint_names)
        if not self.constraint_names:
            raise ValueError("SafeMPO requires at least one constraint")
        self.config = config or SafeMPOTargetConfig()
        self.config.validate()
        self._last_lambda = np.full(
            (len(self.constraint_names),),
            float(self.config.lambda_init),
            dtype=np.float64,
        )
        self._last_nu = float(self.config.nu_init)

    @staticmethod
    def _softmax_np(logits: np.ndarray, axis: int = 1) -> np.ndarray:
        shifted = logits - np.max(logits, axis=axis, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=axis, keepdims=True)

    @staticmethod
    def _logmeanexp_np(value: np.ndarray, axis: int = 1) -> np.ndarray:
        vmax = np.max(value, axis=axis, keepdims=True)
        return (
            np.squeeze(vmax, axis=axis)
            + np.log(np.mean(np.exp(value - vmax), axis=axis))
        )

    def _prepare(
        self,
        task_rewards: torch.Tensor,
        violations: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if task_rewards.ndim != 3:
            raise ValueError("task_rewards must be [B,G,M]")
        if violations.ndim != 4:
            raise ValueError("violations must be [B,G,M,C]")
        if violations.shape[:3] != task_rewards.shape:
            raise ValueError("SafeMPO task/constraint shapes do not match")
        if violations.shape[-1] != len(self.constraint_names):
            raise ValueError("SafeMPO constraint count does not match names")

        batch, group, modes = task_rewards.shape
        if valid_mask is None:
            vm = torch.ones(
                (batch, modes), dtype=torch.bool, device=task_rewards.device
            )
        else:
            vm = valid_mask.to(device=task_rewards.device, dtype=torch.bool)
            if tuple(vm.shape) != (batch, modes):
                raise ValueError("valid_mask must be [B,M]")
        if not bool(vm.any()):
            raise ValueError("SafeMPO batch has no valid vehicle/mode groups")

        # [B,G,M] -> [B,M,G] -> [Nvalid,G]
        q_values = task_rewards.detach().float().permute(0, 2, 1)[vm]
        # SafeMPO Eq. (7): G = -max((C-B)/beta, 0).  The AllMerge
        # constraint pipeline already returns normalized nonnegative violation.
        safety = -violations.detach().float() / float(self.config.constraint_beta)
        safety_values = safety.permute(0, 2, 1, 3)[vm]
        return (
            q_values.cpu().double().numpy(),
            safety_values.cpu().double().numpy(),
            vm.detach().cpu().numpy(),
        )

    def solve(
        self,
        task_rewards: torch.Tensor,
        violations: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None,
    ) -> SafeMPOTargetResult:
        try:
            from scipy.optimize import minimize
        except Exception as exc:  # pragma: no cover - environment check
            raise RuntimeError(
                "SafeMPO-Diff V1 requires scipy.optimize.minimize (SLSQP)"
            ) from exc

        reward_np, safety_np, vm_np = self._prepare(
            task_rewards, violations, valid_mask
        )
        n_state, group_size = reward_np.shape
        n_constraint = safety_np.shape[-1]
        log_group = float(np.log(float(group_size)))

        # A constraint can only be improved by reweighting G if it varies over
        # the sampled group for at least one valid state.  Constant channels
        # (commonly collision=0 for every candidate) are excluded from the
        # barrier dual; forcing x_j>0 there would make the finite-sample primal
        # infeasible even though the channel is already safe.
        ranges = np.ptp(safety_np, axis=1)  # [Nstate,C]
        active = np.max(ranges, axis=0) > float(self.config.active_range_eps)
        active_idx = np.flatnonzero(active)
        base_g = np.mean(safety_np, axis=(0, 1))
        kappa = float(self.config.kappa)
        eps_kl = float(self.config.kl_epsilon)

        def q_from(lam_active: np.ndarray, nu: float) -> tuple[np.ndarray, np.ndarray]:
            if active_idx.size:
                score = reward_np + np.einsum(
                    "sgc,c->sg", safety_np[:, :, active_idx], lam_active
                )
            else:
                score = reward_np
            logits = score / float(nu)
            q = self._softmax_np(logits, axis=1)
            return q, logits

        def objective_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
            if active_idx.size:
                lam = x[:-1]
                nu = float(x[-1])
            else:
                lam = np.zeros((0,), dtype=np.float64)
                nu = float(x[0])
            q, logits = q_from(lam, nu)
            logz = self._logmeanexp_np(logits, axis=1)
            value = nu * eps_kl + nu * float(np.mean(logz))

            if active_idx.size:
                base_active = base_g[active_idx]
                value -= float(np.dot(lam, base_active))
                value -= kappa * float(np.log(lam).sum())

                eq_g = np.mean(
                    np.sum(
                        q[:, :, None] * safety_np[:, :, active_idx],
                        axis=1,
                    ),
                    axis=0,
                )
                grad_lambda = eq_g - base_active - kappa / lam
            else:
                grad_lambda = np.zeros((0,), dtype=np.float64)

            log_q = np.log(np.clip(q, 1e-300, None))
            kl_state = np.sum(q * (log_q + log_group), axis=1)
            grad_nu = eps_kl - float(np.mean(kl_state))
            grad = np.concatenate(
                [grad_lambda, np.asarray([grad_nu], dtype=np.float64)]
            )
            return float(value), grad

        if active_idx.size:
            x0 = np.concatenate(
                [
                    np.clip(
                        self._last_lambda[active_idx],
                        self.config.lambda_min,
                        self.config.lambda_max,
                    ),
                    np.asarray(
                        [
                            np.clip(
                                self._last_nu,
                                self.config.nu_min,
                                self.config.nu_max,
                            )
                        ],
                        dtype=np.float64,
                    ),
                ]
            )
            bounds = [
                (float(self.config.lambda_min), float(self.config.lambda_max))
                for _ in active_idx
            ] + [(float(self.config.nu_min), float(self.config.nu_max))]
        else:
            x0 = np.asarray(
                [
                    np.clip(
                        self._last_nu,
                        self.config.nu_min,
                        self.config.nu_max,
                    )
                ],
                dtype=np.float64,
            )
            bounds = [(float(self.config.nu_min), float(self.config.nu_max))]

        def fun(x: np.ndarray) -> float:
            return objective_grad(x)[0]

        def jac(x: np.ndarray) -> np.ndarray:
            return objective_grad(x)[1]

        solve_start = time.perf_counter()
        result = minimize(
            fun,
            x0,
            jac=jac,
            method="SLSQP",
            bounds=bounds,
            options={
                "maxiter": int(self.config.maxiter),
                "ftol": float(self.config.ftol),
                "disp": False,
            },
        )
        solve_ms = (time.perf_counter() - solve_start) * 1000.0

        x = np.asarray(result.x, dtype=np.float64)
        if not np.all(np.isfinite(x)):
            raise FloatingPointError("SafeMPO dual solver returned NaN/Inf")
        if active_idx.size:
            lam_active = x[:-1]
            nu = float(x[-1])
        else:
            lam_active = np.zeros((0,), dtype=np.float64)
            nu = float(x[0])
        if not np.isfinite(nu) or nu <= 0.0:
            raise FloatingPointError("SafeMPO dual solver returned invalid nu")

        lambdas = np.zeros((n_constraint,), dtype=np.float64)
        if active_idx.size:
            lambdas[active_idx] = lam_active
            self._last_lambda[active_idx] = lam_active
        self._last_nu = nu

        q_np, _ = q_from(lam_active, nu)
        if not np.all(np.isfinite(q_np)):
            raise FloatingPointError("SafeMPO q* contains NaN/Inf")

        log_q = np.log(np.clip(q_np, 1e-300, None))
        kl_state = np.sum(q_np * (log_q + log_group), axis=1)
        teacher_kl = float(np.mean(kl_state))
        entropy_state = -np.sum(q_np * log_q, axis=1)
        entropy_norm = float(np.mean(entropy_state) / max(log_group, 1e-12))
        top1_mass = float(np.mean(np.max(q_np, axis=1)))

        eq_g_all = np.mean(
            np.sum(q_np[:, :, None] * safety_np, axis=1), axis=0
        )
        improvement = eq_g_all - base_g

        # Restore [B,G,M]. Invalid modes get uniform q and are excluded by the
        # M-step mask, so diagnostics remain finite without creating gradients.
        q_t = torch.full_like(task_rewards, 1.0 / float(group_size))
        vm_t = (
            torch.ones(
                (task_rewards.shape[0], task_rewards.shape[2]),
                dtype=torch.bool,
                device=task_rewards.device,
            )
            if valid_mask is None
            else valid_mask.to(device=task_rewards.device, dtype=torch.bool)
        )
        packed = torch.as_tensor(
            q_np,
            device=task_rewards.device,
            dtype=task_rewards.dtype,
        )
        q_bmg = q_t.permute(0, 2, 1).contiguous()
        q_bmg[vm_t] = packed
        q_t = q_bmg.permute(0, 2, 1).contiguous().detach()

        active_residuals = [
            float(improvement[j] - (kappa / max(lambdas[j], self.config.lambda_min)))
            for j in active_idx
        ]
        max_negative_improvement_residual = (
            max([max(-x, 0.0) for x in active_residuals], default=0.0)
        )
        numerical_feasible = (
            teacher_kl <= eps_kl + 5e-4
            and max_negative_improvement_residual <= 1e-4
        )

        metrics: Dict[str, float] = {
            "safempo/dual_success": float(bool(result.success) or numerical_feasible),
            "safempo/solver_success": float(bool(result.success)),
            "safempo/dual_status": float(int(result.status)),
            "safempo/max_negative_improvement_residual": float(
                max_negative_improvement_residual
            ),
            "safempo/dual_iterations": float(getattr(result, "nit", 0)),
            "safempo/dual_solve_ms": float(solve_ms),
            "safempo/dual_objective": float(result.fun),
            "safempo/active_constraint_count": float(active_idx.size),
            "safempo/nu": float(nu),
            "safempo/teacher_kl": teacher_kl,
            "safempo/teacher_kl_budget": eps_kl,
            "safempo/teacher_entropy_normalized": entropy_norm,
            "safempo/teacher_top1_mass": top1_mass,
        }
        for j, name in enumerate(self.constraint_names):
            required = (
                kappa / max(lambdas[j], self.config.lambda_min)
                if active[j]
                else 0.0
            )
            metrics[f"safempo/active_{name}"] = float(bool(active[j]))
            metrics[f"safempo/lambda_{name}"] = float(lambdas[j])
            metrics[f"safempo/base_safety_{name}"] = float(base_g[j])
            metrics[f"safempo/improvement_{name}"] = float(improvement[j])
            metrics[f"safempo/required_improvement_{name}"] = float(required)
            metrics[f"safempo/improvement_residual_{name}"] = float(
                improvement[j] - required
            )
            metrics[f"safempo/group_range_{name}"] = float(np.max(ranges[:, j]))

        return SafeMPOTargetResult(target_q=q_t, metrics=metrics)
