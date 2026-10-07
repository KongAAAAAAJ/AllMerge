from __future__ import annotations

# GRPO_PROJECTION_V3_1_ONE_SIDED_KL_20261007

from dataclasses import dataclass
from typing import Dict

from highway_env.planner.diffusion.projection import (
    ProjectionConfig as _BaseProjectionConfig,
    ProjectionTrainer as _BaseProjectionTrainer,
)


@dataclass
class ProjectionConfig(_BaseProjectionConfig):
    one_sided_kl_low_threshold: float = 0.10
    one_sided_kl_high_threshold: float = 0.20
    one_sided_kl_hard_guard: float = 0.25
    one_sided_kl_up_factor: float = 1.5
    one_sided_kl_down_factor: float = 1.25

    def validate_projection(self) -> None:
        super().validate_projection()
        if self.projection_reference_kl_mode != "adaptive":
            raise ValueError(
                "Projection V3.1 requires projection_reference_kl_mode='adaptive'"
            )
        if float(self.projection_reference_kl_beta_min) < float(self.kl_coef):
            raise ValueError(
                "Projection V3.1 requires beta_min >= fixed V2 kl_coef"
            )
        if self.one_sided_kl_low_threshold < 0.0:
            raise ValueError("one_sided_kl_low_threshold must be >= 0")
        if self.one_sided_kl_high_threshold <= self.one_sided_kl_low_threshold:
            raise ValueError(
                "one_sided_kl_high_threshold must exceed low threshold"
            )
        if self.one_sided_kl_hard_guard <= self.one_sided_kl_high_threshold:
            raise ValueError(
                "one_sided_kl_hard_guard must exceed high threshold"
            )
        if self.one_sided_kl_up_factor <= 1.0:
            raise ValueError("one_sided_kl_up_factor must be > 1")
        if self.one_sided_kl_down_factor <= 1.0:
            raise ValueError("one_sided_kl_down_factor must be > 1")


class ProjectionTrainer(_BaseProjectionTrainer):
    """Projection V3.1: V2/V2.1 projection + one-sided adaptive reference KL.

    The fixed Projection V2 KL coefficient is a hard lower bound.  KL pressure
    can only become stronger than V2 while drift is high, and can only relax
    slowly back toward the V2 baseline.
    """

    def __init__(
        self,
        *args,
        config: ProjectionConfig | None = None,
        **kwargs,
    ) -> None:
        cfg = config or ProjectionConfig()
        cfg.validate_projection()
        super().__init__(*args, config=cfg, **kwargs)
        self.config: ProjectionConfig
        self._one_sided_kl_trigger = 0

    def _projection_reference_kl_control(
        self,
        reference_kl: float,
    ) -> tuple[float, int]:
        value = float(reference_kl)
        self._projection_reference_kl_history.append(value)
        window = int(self.config.projection_reference_kl_window)
        if len(self._projection_reference_kl_history) > window:
            self._projection_reference_kl_history = (
                self._projection_reference_kl_history[-window:]
            )

        rolling = float(
            sum(self._projection_reference_kl_history)
            / len(self._projection_reference_kl_history)
        )

        beta_min = max(
            float(self.config.kl_coef),
            float(self.config.projection_reference_kl_beta_min),
        )
        beta_max = float(self.config.projection_reference_kl_beta_max)
        old_beta = float(self._projection_reference_kl_beta)
        new_beta = old_beta
        self._one_sided_kl_trigger = 0

        # Priority 1: instantaneous hard guard. It does not wait for the
        # rolling window to fill.
        if value > float(self.config.one_sided_kl_hard_guard):
            new_beta = min(
                old_beta * float(self.config.one_sided_kl_up_factor),
                beta_max,
            )
            self._one_sided_kl_trigger = 2

        # Priority 2: persistent high-KL drift.
        elif (
            len(self._projection_reference_kl_history) >= window
            and rolling > float(self.config.one_sided_kl_high_threshold)
        ):
            new_beta = min(
                old_beta * float(self.config.one_sided_kl_up_factor),
                beta_max,
            )
            self._one_sided_kl_trigger = 1

        # Priority 3: slow release only toward the fixed V2 baseline.
        elif (
            len(self._projection_reference_kl_history) >= window
            and rolling < float(self.config.one_sided_kl_low_threshold)
        ):
            new_beta = max(
                old_beta / float(self.config.one_sided_kl_down_factor),
                beta_min,
            )
            self._one_sided_kl_trigger = -1

        new_beta = min(max(new_beta, beta_min), beta_max)
        self._projection_reference_kl_beta = float(new_beta)

        if new_beta > old_beta:
            return rolling, 1
        if new_beta < old_beta:
            return rolling, -1
        return rolling, 0

    def train_step(self, *args, **kwargs) -> Dict[str, float]:
        metrics = super().train_step(*args, **kwargs)
        metrics.update(
            {
                "diagnostics/one_sided_kl_low_threshold": float(
                    self.config.one_sided_kl_low_threshold
                ),
                "diagnostics/one_sided_kl_high_threshold": float(
                    self.config.one_sided_kl_high_threshold
                ),
                "diagnostics/one_sided_kl_hard_guard": float(
                    self.config.one_sided_kl_hard_guard
                ),
                "diagnostics/one_sided_kl_trigger": float(
                    self._one_sided_kl_trigger
                ),
            }
        )
        return metrics

    def constraint_state_dict(self) -> dict:
        state = super().constraint_state_dict()
        state["strategy"] = "projection_v3_1_one_sided_kl"
        state["one_sided_kl"] = {
            "beta_floor": max(
                float(self.config.kl_coef),
                float(self.config.projection_reference_kl_beta_min),
            ),
            "low_threshold": float(self.config.one_sided_kl_low_threshold),
            "high_threshold": float(self.config.one_sided_kl_high_threshold),
            "hard_guard": float(self.config.one_sided_kl_hard_guard),
            "up_factor": float(self.config.one_sided_kl_up_factor),
            "down_factor": float(self.config.one_sided_kl_down_factor),
            "trigger": int(self._one_sided_kl_trigger),
        }
        return state
