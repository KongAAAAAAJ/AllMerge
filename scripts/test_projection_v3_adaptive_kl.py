from __future__ import annotations

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from highway_env.planner.diffusion.projection.trainer import (
    ProjectionConfig,
    ProjectionTrainer,
)


def _make() -> ProjectionTrainer:
    trainer = object.__new__(ProjectionTrainer)
    trainer.config = ProjectionConfig(
        kl_coef=0.05,
        projection_reference_kl_mode="adaptive",
        projection_reference_kl_target=0.15,
        projection_reference_kl_beta_min=0.005,
        projection_reference_kl_beta_max=0.5,
        projection_reference_kl_adapt_factor=1.5,
        projection_reference_kl_window=10,
    )
    trainer.config.validate_projection()
    trainer._projection_reference_kl_beta = 0.05
    trainer._projection_reference_kl_history = []
    return trainer


def main() -> int:
    t = _make()

    for _ in range(9):
        rolling, event = t._projection_reference_kl_control(0.30)
        assert event == 0
        assert abs(t._projection_reference_kl_beta - 0.05) < 1e-12

    rolling, event = t._projection_reference_kl_control(0.30)
    assert event == 1
    assert abs(t._projection_reference_kl_beta - 0.075) < 1e-12
    print(
        "[PASS] high rolling KL increases beta "
        f"rolling={rolling:.3f} beta={t._projection_reference_kl_beta:.6f}"
    )

    down_seen = False
    beta_before_down = None
    beta_after_down = None
    for _ in range(30):
        old_beta = float(t._projection_reference_kl_beta)
        rolling, event = t._projection_reference_kl_control(0.05)
        if event < 0:
            down_seen = True
            beta_before_down = old_beta
            beta_after_down = float(t._projection_reference_kl_beta)
            break

    assert down_seen
    assert beta_before_down is not None
    assert beta_after_down is not None
    assert beta_after_down < beta_before_down
    assert 0.005 <= t._projection_reference_kl_beta <= 0.5
    print(
        "[PASS] sustained low rolling KL eventually decreases beta "
        f"rolling={rolling:.3f} "
        f"beta={beta_before_down:.6f}->{beta_after_down:.6f}"
    )
    print("[OK] Projection V3 adaptive-KL controller self-test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
