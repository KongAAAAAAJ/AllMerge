from __future__ import annotations

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from highway_env.planner.diffusion.projection.trainer import (
    ProjectionConfig,
    ProjectionTrainer,
)


def _trainer() -> ProjectionTrainer:
    trainer = object.__new__(ProjectionTrainer)
    trainer.config = ProjectionConfig(
        max_grad_norm=5.0,
        projection_passes=1,
        projection_tolerance=1e-7,
        projection_restoration_strength=0.25,
        projection_margin_backoff_factor=0.5,
        projection_margin_backoff_steps=1,
    )
    return trainer


def main() -> int:
    trainer = _trainer()

    preferred = torch.tensor([0.90927435, 1.58725555], dtype=torch.float32)
    normals = [
        torch.tensor([0.99039376, 0.13827583], dtype=torch.float32),
        torch.tensor([-0.25968553, -0.96569324], dtype=torch.float32),
        torch.tensor([-0.79324645, 0.60890071], dtype=torch.float32),
    ]
    normals = [n / n.norm() for n in normals]

    _, fast_diag = trainer._dykstra_shifted_halfspaces_ball_once(
        preferred,
        normals,
        [0.0, 0.0, 0.0],
        radius=5.0,
        max_passes=1,
    )
    print(
        "[test] one-pass zero-margin converged="
        f"{int(fast_diag['converged'])} "
        f"min_residual={fast_diag['min_margin_residual_after']:.6g}"
    )

    rescued, rescue_diag = trainer._dykstra_shifted_halfspaces_ball_once(
        preferred,
        normals,
        [0.0, 0.0, 0.0],
        radius=5.0,
        max_passes=128,
    )
    residuals = [float(torch.dot(n, rescued)) for n in normals]
    assert min(residuals) >= -5e-6
    assert float(rescued.norm()) <= 5.00001
    print(
        "[PASS] extended Dykstra rescue reached feasible zero-margin cone "
        f"passes={rescue_diag['passes_used']:.0f}"
    )

    projected, diag, used_normals, margins = trainer._restorative_projection(
        preferred,
        [n.clone() for n in normals],
        [1.0, 1.0, 1.0],
    )
    assert diag["violated_after_count"] == 0.0
    assert diag["hard_feasible"] == 1.0
    assert float(projected.norm()) <= 5.00001
    for n, m in zip(used_normals, margins):
        assert float(torch.dot(n, projected)) + 5e-6 >= float(m)

    print(
        "[PASS] full restorative projection survived narrow geometry "
        f"fallback_zero_margin={diag['fallback_zero_margin']:.0f} "
        f"exact_zero_fallback={diag['exact_zero_fallback']:.0f} "
        f"solver_rescue_passes={diag['solver_rescue_passes']:.0f}"
    )
    print("[OK] Projection V2.1 Dykstra rescue self-test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
