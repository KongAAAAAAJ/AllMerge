from __future__ import annotations

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if not (_REPO_ROOT / "highway_env").is_dir():
    raise SystemExit(f"[FAIL] repo root not found from helper script: {_REPO_ROOT}")
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from highway_env.planner.diffusion.projection.trainer import ProjectionConfig, ProjectionTrainer


def _trainer() -> ProjectionTrainer:
    trainer = object.__new__(ProjectionTrainer)
    trainer.config = ProjectionConfig(
        max_grad_norm=5.0,
        projection_passes=32,
        projection_tolerance=1e-7,
        projection_restoration_strength=0.25,
        projection_margin_backoff_factor=0.5,
        projection_margin_backoff_steps=4,
    )
    return trainer


def main() -> int:
    trainer = _trainer()

    preferred = torch.tensor([5.0, 0.0])
    constraint = [torch.tensor([0.0, 2.0])]
    projected, diag, normals, margins = trainer._restorative_projection(
        preferred, constraint, [1.0]
    )
    assert diag["violated_before_count"] == 1.0
    assert diag["violated_after_count"] == 0.0
    assert diag["correction_norm"] > 0.0
    assert float(projected.norm()) <= 5.00001
    assert float(torch.dot(normals[0], projected)) + 1e-5 >= margins[0]
    print(
        "[PASS] shifted-halfspace correction "
        f"correction_ratio={diag['correction_ratio']:.6f} "
        f"margin={margins[0]:.6f}"
    )

    preferred_aligned = torch.tensor([0.0, 5.0])
    projected2, diag2, _, _ = trainer._restorative_projection(
        preferred_aligned, constraint, [0.5]
    )
    assert diag2["violated_before_count"] == 0.0
    assert diag2["correction_norm"] < 1e-5
    torch.testing.assert_close(projected2, preferred_aligned, atol=1e-5, rtol=0.0)
    print("[PASS] already-feasible preferred gradient is unchanged")

    contradictory = [torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0])]
    projected3, diag3, _, margins3 = trainer._restorative_projection(
        torch.tensor([0.0, 1.0]), contradictory, [1.0, 1.0]
    )
    assert diag3["violated_after_count"] == 0.0
    assert diag3["fallback_zero_margin"] == 1.0
    assert all(abs(value) <= 1e-12 for value in margins3)
    assert float(projected3.norm()) <= 5.00001
    print("[PASS] contradictory positive margins safely fall back to zero-margin cone")
    print("[OK] Projection V2 mathematical self-test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
