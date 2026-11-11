"""Differentiable VT-CPFM power-based fuel objective (prepared for Koopman D2).

Coefficients are REQUIRED from calibrated source; this module deliberately
has no invented truck-specific coefficient defaults. Flat road only.
"""
from __future__ import annotations
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FuelParameters:
    empty_mass_kg: float
    rolling_resistance: float
    drag_coefficient: float
    frontal_area_m2: float
    drivetrain_efficiency: float
    alpha0: float
    alpha1: float
    alpha2: float
    air_density_kg_m3: float = 1.225
    gravity_m_s2: float = 9.81

    def validate(self):
        if min(self.empty_mass_kg, self.rolling_resistance,
               self.drag_coefficient, self.frontal_area_m2,
               self.drivetrain_efficiency, self.air_density_kg_m3) <= 0:
            raise ValueError('Physical coefficients must be positive')
        if not 0 < self.drivetrain_efficiency <= 1:
            raise ValueError('Drivetrain efficiency must be in (0,1]')
        if min(self.alpha0, self.alpha1, self.alpha2) < 0:
            raise ValueError('Fuel coefficients must be nonnegative')


def vt_cpfm_from_states(vx_mps: torch.Tensor, payload_kg: torch.Tensor,
                        params: FuelParameters, dt: float = 0.1):
    """Compute differentiable fuel rate [B,H] and cumulative fuel [B].

    Args:
        vx_mps: predicted (not commanded) longitudinal speeds [B,H+1].
        payload_kg: physical payload [B], held constant across horizon.
        params: calibrated VT-CPFM vehicle coefficients.
    Note:
        Units of fuel rate/integral follow the alpha coefficient calibration.
    """
    params.validate()
    if vx_mps.ndim != 2 or vx_mps.shape[1] < 2:
        raise ValueError('vx_mps must be [B,H+1]')
    if payload_kg.reshape(-1).numel() != vx_mps.shape[0] or dt <= 0:
        raise ValueError('payload batch and timestep mismatch')
    total_mass = params.empty_mass_kg + payload_kg.reshape(-1, 1).to(vx_mps)
    vx = vx_mps[:, :-1].clamp_min(0)
    ax = (vx_mps[:, 1:] - vx_mps[:, :-1]) / dt
    force = (total_mass * ax + total_mass * params.gravity_m_s2 * params.rolling_resistance
             + .5 * params.air_density_kg_m3 * params.drag_coefficient
             * params.frontal_area_m2 * vx.square())
    power_kw = vx * force / (1000.0 * params.drivetrain_efficiency)
    positive_power_kw = power_kw.clamp_min(0)
    fuel_rate = (params.alpha0 + params.alpha1 * positive_power_kw
                 + params.alpha2 * positive_power_kw.square())
    return {'power_kw': power_kw, 'fuel_rate': fuel_rate,
            'fuel_total': fuel_rate.sum(dim=1) * dt}


def ltr_soft_objective(pred_ltr: torch.Tensor):
    """Soft response penalty only; no dynamic stability limit/projection here."""
    if pred_ltr.ndim != 2 or pred_ltr.shape[1] < 2:
        raise ValueError('LTR prediction must be [B,H+1]')
    return pred_ltr[:, 1:].square().mean()
