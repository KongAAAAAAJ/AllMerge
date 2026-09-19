from __future__ import annotations

import sys
from pathlib import Path

# Direct execution sets sys.path[0] to scripts/. Add the repo root explicitly so
# the local highway_env package is importable without relying on PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from highway_env.planner.diffusion.grpo.objective import (
    group_relative_advantage,
    rank_group_advantage,
)
from highway_env.planner.diffusion.grpo.trainer import GRPOConfig


def _close(a, b, atol=1e-6):
    if not torch.allclose(a, b, atol=atol, rtol=0.0):
        raise AssertionError(f"actual={a} expected={b}")


def test_default_is_standard():
    assert GRPOConfig().advantage_transform == "standard"


def test_zero_mean_and_order():
    r = torch.tensor([[[1.0], [4.0], [2.0], [3.0]]])
    a = rank_group_advantage(r)
    _close(a.mean(dim=1), torch.zeros((1, 1)))
    assert int(a[:, :, 0].argmax(dim=1).item()) == int(r[:, :, 0].argmax(dim=1).item())


def test_positive_affine_invariance():
    r = torch.tensor([
        [[-100.0, 1.0], [-5.0, 4.0], [0.0, 2.0], [20.0, 3.0]],
        [[2.0, 8.0], [1.0, 8.0], [9.0, -1.0], [3.0, 2.0]],
    ])
    base = rank_group_advantage(r)
    _close(base, rank_group_advantage(r + 1234.5))
    _close(base, rank_group_advantage(r * 7.25))


def test_ties_are_average_rank_and_deterministic():
    r = torch.tensor([[[0.0], [0.0], [2.0], [4.0]]])
    first = rank_group_advantage(r)
    second = rank_group_advantage(r.clone())
    _close(first, second)
    expected = torch.tensor([[[-2.0 / 3.0], [-2.0 / 3.0], [1.0 / 3.0], [1.0]]])
    _close(first, expected)


def test_group_size_one_and_mask_are_finite():
    one = torch.tensor([[[3.0, -7.0]]])
    a = rank_group_advantage(one)
    assert bool(torch.isfinite(a).all())
    _close(a, torch.zeros_like(a))

    r = torch.tensor([[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]])
    mask = torch.tensor([[True, False]])
    a = rank_group_advantage(r, valid_mask=mask)
    assert bool(torch.isfinite(a).all())
    _close(a[:, :, 1], torch.zeros_like(a[:, :, 1]))
    _close(a[:, :, 0].mean(dim=1), torch.zeros((1,)))


def test_standard_math_is_unchanged():
    r = torch.tensor([[[1.0], [2.0], [5.0], [8.0]]])
    got = group_relative_advantage(r, eps=1e-6)
    expected = (r - r.mean(dim=1, keepdim=True)) / (r.std(dim=1, unbiased=False, keepdim=True) + 1e-6)
    _close(got, expected)


def main():
    tests = [
        test_default_is_standard,
        test_zero_mean_and_order,
        test_positive_affine_invariance,
        test_ties_are_average_rank_and_deterministic,
        test_group_size_one_and_mask_are_finite,
        test_standard_math_is_unchanged,
    ]
    for fn in tests:
        fn()
        print(f"[PASS] {fn.__name__}")
    print("PASS: GRPO D rank-based robust advantage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
