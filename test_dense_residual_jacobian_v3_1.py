from __future__ import annotations

import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import importlib.util

module_path = ROOT / "highway_env/planner/diffusion/dense_supervision.py"
spec = importlib.util.spec_from_file_location("allmerge_dense_supervision", module_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load dense supervision module: {module_path}")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
DenseSparseResidualHead = module.DenseSparseResidualHead


def main() -> int:
    torch.manual_seed(7)
    head = DenseSparseResidualHead(
        feature_dim=4,
        horizon_steps=8,
        hidden_dim=32,
        max_residual_x_m=2.0,
        max_residual_y_m=0.75,
    )

    # Constructor intentionally zero-initializes the last layer. Make the test
    # input-sensitive without changing the semantics being tested.
    with torch.no_grad():
        final = head.net[-1]
        final.weight.fill_(0.02)
        final.bias.zero_()

    feature = torch.randn(2, 4)
    sparse = torch.randn(2, 8, 2, requires_grad=True)

    legacy = head(feature, sparse)
    guided = head.forward_guidance(feature, sparse)

    # Forward numerics must stay identical.
    if not torch.allclose(legacy, guided, atol=0.0, rtol=0.0):
        raise AssertionError("legacy and guidance forward values differ")

    # Historical forward must still cut the sparse-input Jacobian.
    legacy_grad = torch.autograd.grad(
        legacy.sum(),
        sparse,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if legacy_grad is not None and float(legacy_grad.abs().max().item()) != 0.0:
        raise AssertionError("legacy forward unexpectedly exposes sparse Jacobian")

    # Guidance-only forward must expose a non-zero sparse-input Jacobian.
    guided_grad = torch.autograd.grad(
        guided.sum(),
        sparse,
        retain_graph=False,
        allow_unused=False,
    )[0]
    jac_norm = float(torch.linalg.vector_norm(guided_grad).item())
    if not torch.isfinite(guided_grad).all() or jac_norm <= 0.0:
        raise AssertionError("guidance residual Jacobian is not active")

    # autograd.grad(..., sparse) does not accumulate parameter gradients.
    if any(p.grad is not None for p in head.parameters()):
        raise AssertionError("residual-head parameter .grad was unexpectedly populated")

    print("[PASS] legacy forward still hard-detaches sparse input")
    print("[PASS] forward_guidance exposes non-zero d(residual)/d(sparse)")
    print("[PASS] legacy/guidance forward values are exactly identical")
    print("[PASS] residual-head parameters receive no accumulated gradient")
    print(f"residual_jacobian_norm={jac_norm:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
