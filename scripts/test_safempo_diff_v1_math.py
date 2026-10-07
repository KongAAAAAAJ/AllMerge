from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_PATH = REPO_ROOT / "highway_env" / "planner" / "diffusion" / "safempo_diff" / "target.py"
_spec = importlib.util.spec_from_file_location("safempo_diff_target_standalone", TARGET_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"cannot load SafeMPO target module: {TARGET_PATH}")
_target = importlib.util.module_from_spec(_spec)
import sys
sys.modules[_spec.name] = _target
_spec.loader.exec_module(_target)
SafeMPOTargetBuilder = _target.SafeMPOTargetBuilder
SafeMPOTargetConfig = _target.SafeMPOTargetConfig


def main() -> int:
    # One vehicle, G=8, one valid mode.  collision is intentionally constant
    # and must be excluded from the finite-particle barrier dual.
    reward = torch.tensor(
        [[[1.00], [0.92], [0.84], [0.76], [0.68], [0.60], [0.52], [0.44]]],
        dtype=torch.float64,
    )
    road = torch.tensor(
        [[[3.0], [2.4], [1.8], [1.2], [0.8], [0.4], [0.1], [0.0]]],
        dtype=torch.float64,
    )
    ttc = torch.tensor(
        [[[0.0], [0.0], [0.1], [0.1], [0.2], [0.2], [0.1], [0.0]]],
        dtype=torch.float64,
    )
    collision = torch.zeros_like(road)
    violations = torch.stack([road, ttc, collision], dim=-1)

    builder = SafeMPOTargetBuilder(
        ("road", "ttc", "collision"),
        config=SafeMPOTargetConfig(
            kl_epsilon=0.10,
            kappa=0.05,
            lambda_init=1.0,
            maxiter=128,
        ),
    )
    out = builder.solve(
        reward,
        violations,
        valid_mask=torch.tensor([[True]]),
    )
    q = out.target_q[0, :, 0]
    assert torch.isfinite(q).all()
    assert abs(float(q.sum()) - 1.0) < 1e-8
    assert float(out.metrics["safempo/teacher_kl"]) <= 0.1005
    assert out.metrics["safempo/active_road"] == 1.0
    assert out.metrics["safempo/active_collision"] == 0.0
    print("[PASS] SafeMPO-Diff V1 finite-particle dual")
    print("q*=", [round(float(x), 6) for x in q])
    for key in (
        "safempo/dual_success",
        "safempo/dual_iterations",
        "safempo/nu",
        "safempo/teacher_kl",
        "safempo/teacher_entropy_normalized",
        "safempo/lambda_road",
        "safempo/lambda_ttc",
        "safempo/active_collision",
    ):
        print(f"{key}={out.metrics[key]:.9g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
