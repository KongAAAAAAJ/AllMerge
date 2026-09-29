from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

module_path = (
    Path(__file__).resolve().parent
    / "highway_env/planner/diffusion/guidance.py"
)
spec = importlib.util.spec_from_file_location(
    "allmerge_sparse_curvature_guidance", module_path
)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load guidance module: {module_path}")

guidance_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = guidance_module
spec.loader.exec_module(guidance_module)

SparseCurvatureGuidanceConfig = guidance_module.SparseCurvatureGuidanceConfig
apply_sparse_curvature_guidance = guidance_module.apply_sparse_curvature_guidance
sparse_curvature_statistics = guidance_module.sparse_curvature_statistics


def main() -> int:
    physical = torch.tensor(
        [[[[5.0, 0.0], [10.0, 2.0], [15.0, -2.0], [20.0, 2.0],
           [25.0, -2.0], [30.0, 2.0], [35.0, -2.0], [40.0, 0.0]]]],
        dtype=torch.float32,
    )
    scale = torch.tensor([120.0, 24.0], dtype=torch.float32)
    x0_norm = physical / scale
    mode_valid = torch.ones((1, 1), dtype=torch.bool)

    disabled = SparseCurvatureGuidanceConfig(enabled=False)
    same, diag = apply_sparse_curvature_guidance(
        x0_norm,
        trajectory_scale=scale,
        mode_valid_mask=mode_valid,
        config=disabled,
        timestep=8,
    )
    assert diag is None
    assert torch.equal(same, x0_norm), "disabled guidance changed x0"

    cfg = SparseCurvatureGuidanceConfig(
        enabled=True,
        scale=0.10,
        curvature_limit=0.02,
        grad_clip_norm=1.0,
        preserve_endpoint=True,
    )
    before = sparse_curvature_statistics(
        physical,
        curvature_limit=cfg.curvature_limit,
    )
    guided_norm, diag = apply_sparse_curvature_guidance(
        x0_norm,
        trajectory_scale=scale,
        mode_valid_mask=mode_valid,
        config=cfg,
        timestep=8,
    )
    assert diag is not None

    guided = guided_norm * scale
    after = sparse_curvature_statistics(
        guided,
        curvature_limit=cfg.curvature_limit,
    )

    assert torch.allclose(
        guided[..., -1, :],
        physical[..., -1, :],
    ), "endpoint moved although preserve_endpoint=True"

    assert float(diag["loss_after"].item()) <= (
        float(diag["loss_before"].item()) + 1e-6
    )

    assert torch.isfinite(guided).all()
    assert torch.isfinite(after["max_abs_curvature"]).all()

    print("[PASS] disabled path is bit-identical")
    print("[PASS] enabled guidance keeps endpoint fixed and reduces curvature loss")
    print(
        "max|kappa| before/after:",
        float(before["max_abs_curvature"].item()),
        float(after["max_abs_curvature"].item()),
    )
    print("update_l2_m:", float(diag["update_l2_m"].item()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
