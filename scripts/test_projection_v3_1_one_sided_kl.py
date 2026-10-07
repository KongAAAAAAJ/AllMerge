from __future__ import annotations

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from highway_env.planner.diffusion.projection_v3_1 import (
    ProjectionConfig,
    ProjectionTrainer,
)


def _make() -> ProjectionTrainer:
    t = object.__new__(ProjectionTrainer)
    t.config = ProjectionConfig(
        kl_coef=0.05,
        projection_reference_kl_mode="adaptive",
        projection_reference_kl_target=0.15,
        projection_reference_kl_beta_min=0.05,
        projection_reference_kl_beta_max=0.5,
        projection_reference_kl_adapt_factor=1.5,
        projection_reference_kl_window=10,
        one_sided_kl_low_threshold=0.10,
        one_sided_kl_high_threshold=0.20,
        one_sided_kl_hard_guard=0.25,
        one_sided_kl_up_factor=1.5,
        one_sided_kl_down_factor=1.25,
    )
    t.config.validate_projection()
    t._projection_reference_kl_beta = 0.05
    t._projection_reference_kl_history = []
    t._one_sided_kl_trigger = 0
    return t


def main() -> int:
    t = _make()

    # Low KL cannot weaken below the fixed V2 baseline.
    for _ in range(15):
        rolling, event = t._projection_reference_kl_control(0.06)
        assert event == 0
        assert abs(t._projection_reference_kl_beta - 0.05) < 1e-12
    print("[PASS] beta floor preserves fixed V2 trust penalty at 0.05")

    # Hard guard reacts immediately.
    rolling, event = t._projection_reference_kl_control(0.30)
    assert event == 1
    assert t._one_sided_kl_trigger == 2
    assert abs(t._projection_reference_kl_beta - 0.075) < 1e-12
    print("[PASS] instantaneous KL>0.25 tightens beta 0.05->0.075")

    # Persistent high rolling KL continues tightening.
    before = float(t._projection_reference_kl_beta)
    for _ in range(10):
        rolling, event = t._projection_reference_kl_control(0.22)
    assert t._projection_reference_kl_beta > before
    assert t._projection_reference_kl_beta <= 0.5
    print(
        "[PASS] persistent high rolling KL tightens beta further "
        f"beta={t._projection_reference_kl_beta:.6f}"
    )

    # Sustained low KL releases slowly, never below 0.05.
    start = float(t._projection_reference_kl_beta)
    down_seen = False
    for _ in range(30):
        rolling, event = t._projection_reference_kl_control(0.05)
        down_seen = down_seen or (event < 0)
    assert down_seen
    assert t._projection_reference_kl_beta < start
    assert t._projection_reference_kl_beta >= 0.05 - 1e-12
    print(
        "[PASS] sustained low KL releases beta slowly but respects floor "
        f"{start:.6f}->{t._projection_reference_kl_beta:.6f}"
    )

    print("[OK] Projection V3.1 One-Sided Adaptive-KL self-test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
