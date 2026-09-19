from __future__ import annotations

import torch

from highway_env.planner.diffusion.grpo.objective import (
    group_relative_advantage,
    grpo_clipped_objective,
)
from highway_env.planner.diffusion.grpo.trainer import GRPOConfig, GRPOTrainer


def stub(mode: str) -> GRPOTrainer:
    trainer = GRPOTrainer.__new__(GRPOTrainer)
    trainer.config = GRPOConfig(advantage_mode=mode)
    return trainer


def main() -> int:
    assert GRPOConfig().advantage_mode == "joint"

    rewards = torch.tensor(
        [
            [[0.0, 1.0, 2.0], [1.0, 3.0, 4.0], [2.0, 2.0, 8.0], [5.0, 7.0, 9.0]],
            [[2.0, 1.0, 0.0], [3.0, 5.0, 2.0], [4.0, 4.0, 7.0], [9.0, 8.0, 6.0]],
            [[1.0, 2.0, 3.0], [4.0, 2.0, 7.0], [6.0, 5.0, 8.0], [8.0, 9.0, 10.0]],
        ],
        dtype=torch.float32,
    )
    valid_mask = torch.tensor(
        [
            [True, True, True],
            [True, True, False],
            [True, False, False],
        ]
    )

    joint = stub("joint")
    baseline = group_relative_advantage(
        rewards,
        eps=joint.config.advantage_eps,
        valid_mask=valid_mask,
    )
    assert torch.equal(joint._compute_advantages(rewards, valid_mask), baseline)

    rolewise = stub("rolewise")
    role_adv = rolewise._compute_advantages(rewards, valid_mask)
    assert torch.equal(role_adv, baseline)

    for role in range(3):
        for mode in range(3):
            if bool(valid_mask[role, mode]):
                assert abs(float(role_adv[role, :, mode].mean())) < 2e-5

    one = torch.tensor(
        [[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]],
        dtype=torch.float32,
    )
    one_mask = torch.ones((3, 2), dtype=torch.bool)
    one_adv = rolewise._compute_advantages(one, one_mask)
    assert torch.isfinite(one_adv).all()
    assert torch.equal(one_adv, torch.zeros_like(one_adv))

    torch.manual_seed(7)
    old_lp = torch.randn(3, 4, 2, 3) * 0.01
    new_lp = old_lp + torch.randn(3, 4, 2, 3) * 0.02

    _, role_objs, role_loss = rolewise._policy_terms(
        new_lp, old_lp, role_adv, valid_mask
    )
    expected = torch.stack([obj.policy_loss for obj in role_objs.values()]).mean()
    assert torch.allclose(role_loss, expected, atol=1e-7, rtol=0.0)

    original = grpo_clipped_objective(
        new_lp,
        old_lp,
        baseline,
        clip_eps=joint.config.clip_eps,
        valid_mask=valid_mask,
    )
    global_obj, _, joint_loss = joint._policy_terms(
        new_lp, old_lp, baseline, valid_mask
    )
    assert torch.equal(joint_loss, original.policy_loss)
    assert torch.equal(global_obj.approx_kl, original.approx_kl)
    assert torch.equal(global_obj.clip_fraction, original.clip_fraction)
    assert torch.equal(global_obj.ratio_mean, original.ratio_mean)

    diag = rolewise._role_advantage_diagnostics(role_adv, valid_mask)
    for role in range(3):
        assert f"diagnostics/vehicle_{role}_advantage_mean" in diag
        assert f"diagnostics/vehicle_{role}_advantage_std" in diag

    print("PASS: GRPO A role-wise credit smoke tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
