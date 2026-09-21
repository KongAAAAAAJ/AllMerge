# STAGED_DENSE_SUPERVISION_V1
from __future__ import annotations

import torch

from highway_env.planner.diffusion.config import build_structured_diffusion_config
from highway_env.planner.diffusion.structured_model import StructuredDiffusionPlanner
from highway_env.planner.diffusion.tensor_adapter import PlannerTensorAdapter


def _synthetic_features(cfg, device):
    b = 2
    features = {
        "ego_state": torch.zeros((b, cfg.ego_dim), dtype=torch.float32, device=device),
        "agent_states": torch.zeros((b, cfg.max_agents, cfg.agent_dim), dtype=torch.float32, device=device),
        "agent_valid_mask": torch.zeros((b, cfg.max_agents), dtype=torch.bool, device=device),
        "map_polylines": torch.zeros((b, cfg.max_map_polylines, cfg.map_points, cfg.map_dim), dtype=torch.float32, device=device),
        "map_valid_mask": torch.ones((b, cfg.max_map_polylines), dtype=torch.bool, device=device),
        "target_point": torch.zeros((b, 2), dtype=torch.float32, device=device),
        "target_lane_polyline": torch.zeros((b, cfg.map_points, cfg.map_dim), dtype=torch.float32, device=device),
        "coarse_trajectories": torch.zeros((b, cfg.num_modes, cfg.horizon_steps, 2), dtype=torch.float32, device=device),
        "mode_valid_mask": torch.ones((b, cfg.num_modes), dtype=torch.bool, device=device),
    }
    features["ego_state"][:, 0] = 6.0
    t_sparse = torch.arange(1, cfg.horizon_steps + 1, device=device, dtype=torch.float32) * cfg.trajectory_dt
    for m in range(cfg.num_modes):
        features["coarse_trajectories"][:, m, :, 0] = 6.0 * t_sparse
        features["coarse_trajectories"][:, m, :, 1] = 0.1 * float(m)
    return features



def _dense_target(cfg, device, lateral_offset=0.35):
    t_dense = torch.arange(1, 41, dtype=torch.float32, device=device) * 0.1
    expert_dense = torch.zeros((2, 40, 2), dtype=torch.float32, device=device)
    expert_dense[..., 0] = 6.0 * t_dense
    # Deliberately not identical to the baseline spline so residual gradients
    # are guaranteed non-zero in this unit test.
    expert_dense[..., 1] = lateral_offset * torch.sin(t_dense * 1.2)
    return expert_dense


def test_dense_terminal_auxiliary_backpropagates_only_to_residual_head():
    device = torch.device("cpu")
    cfg = build_structured_diffusion_config(
        d_model=32,
        d_ffn=64,
        num_heads=4,
        num_scene_layers=1,
        num_denoiser_layers=1,
        dropout=0.1,
        dense_residual_hidden_dim=32,
    )
    adapter = PlannerTensorAdapter(cfg, device)
    planner = StructuredDiffusionPlanner(cfg, adapter).to(device)
    features = _synthetic_features(cfg, device)
    scene = planner.scene_encoder(features)
    clean_anchor_norm = adapter.normalize_trajectory(features["coarse_trajectories"])
    noise = torch.zeros_like(clean_anchor_norm)
    target_mode = torch.zeros((2,), dtype=torch.long, device=device)
    expert_dense = _dense_target(cfg, device)

    out = planner._dense_terminal_auxiliary(
        scene=scene,
        clean_anchor_norm=clean_anchor_norm,
        noise=noise,
        features=features,
        target_mode=target_mode,
        target_trajectory_dense=expert_dense,
        dense_loss_type="smooth_l1",
        dense_loss_lambda_p=1.0,
        dense_loss_terminal_timestep=0,
        dense_loss_weight_mode="terminal_constant",
        dense_loss_terminal_weight=1.0,
    )
    assert torch.isfinite(out["dense_loss_raw"])
    out["dense_loss_weighted"].backward()

    residual_grads = [
        p.grad for p in planner.dense_residual_head.parameters()
        if p.grad is not None
    ]
    assert residual_grads
    assert all(torch.isfinite(g).all() for g in residual_grads)
    assert sum(float(g.abs().sum()) for g in residual_grads) > 0.0

    # Hard requirement: L_dense must not touch the Diffusion planner.
    for name, parameter in planner.named_parameters():
        if name.startswith("dense_residual_head."):
            continue
        assert parameter.grad is None, f"dense gradient leaked into {name}"


def test_residual_head_is_exact_identity_at_initialization():
    device = torch.device("cpu")
    cfg = build_structured_diffusion_config(
        d_model=32,
        d_ffn=64,
        num_heads=4,
        num_scene_layers=1,
        num_denoiser_layers=1,
        dropout=0.0,
        dense_residual_hidden_dim=32,
    )
    adapter = PlannerTensorAdapter(cfg, device)
    planner = StructuredDiffusionPlanner(cfg, adapter).to(device)
    feature = torch.randn(3, cfg.d_model)
    sparse_norm = torch.randn(3, cfg.horizon_steps, 2)
    delta = planner.dense_residual_head(feature, sparse_norm)
    assert torch.equal(delta, torch.zeros_like(delta))


def test_dense_branch_preserves_rng_stream():
    device = torch.device("cpu")
    cfg = build_structured_diffusion_config(
        d_model=32,
        d_ffn=64,
        num_heads=4,
        num_scene_layers=1,
        num_denoiser_layers=1,
        dropout=0.2,
        dense_residual_hidden_dim=32,
    )
    adapter = PlannerTensorAdapter(cfg, device)
    planner = StructuredDiffusionPlanner(cfg, adapter).to(device).train()
    features = _synthetic_features(cfg, device)
    target_sparse = features["coarse_trajectories"][:, 0].clone()
    target_semantic = torch.zeros((2,), dtype=torch.long, device=device)
    expert_dense = _dense_target(cfg, device)

    torch.manual_seed(12345)
    planner.forward_train(
        features,
        target_sparse,
        target_semantic,
        dense_loss_enabled=False,
        dense_loss_lambda_p=0.0,
    )
    after_off = torch.rand(8)

    torch.manual_seed(12345)
    planner.forward_train(
        features,
        target_sparse,
        target_semantic,
        target_trajectory_dense=expert_dense,
        dense_loss_enabled=True,
        dense_loss_lambda_p=0.1,
    )
    after_on = torch.rand(8)
    assert torch.equal(after_off, after_on)


def test_dense_disabled_or_zero_lambda_preserves_baseline_path():
    device = torch.device("cpu")
    cfg = build_structured_diffusion_config(
        d_model=32,
        d_ffn=64,
        num_heads=4,
        num_scene_layers=1,
        num_denoiser_layers=1,
        dropout=0.0,
        dense_residual_hidden_dim=32,
    )
    adapter = PlannerTensorAdapter(cfg, device)
    planner = StructuredDiffusionPlanner(cfg, adapter).to(device)
    features = _synthetic_features(cfg, device)
    target_sparse = features["coarse_trajectories"][:, 0].clone()
    target_semantic = torch.zeros((2,), dtype=torch.long, device=device)

    torch.manual_seed(123)
    out_off = planner.forward_train(
        features,
        target_sparse,
        target_semantic,
        dense_loss_enabled=False,
        dense_loss_lambda_p=0.0,
    )
    torch.manual_seed(123)
    out_zero = planner.forward_train(
        features,
        target_sparse,
        target_semantic,
        target_trajectory_dense=None,
        dense_loss_enabled=True,
        dense_loss_lambda_p=0.0,
    )
    assert torch.equal(out_off["loss"], out_zero["loss"])
    assert torch.equal(
        out_off["trajectory_regression_loss"],
        out_zero["trajectory_regression_loss"],
    )
    assert float(out_zero["dense_loss_weighted"]) == 0.0


def test_dense_enabled_requires_real_dense_target():
    device = torch.device("cpu")
    cfg = build_structured_diffusion_config(
        d_model=32,
        d_ffn=64,
        num_heads=4,
        num_scene_layers=1,
        num_denoiser_layers=1,
        dropout=0.0,
        dense_residual_hidden_dim=32,
    )
    adapter = PlannerTensorAdapter(cfg, device)
    planner = StructuredDiffusionPlanner(cfg, adapter).to(device)
    features = _synthetic_features(cfg, device)
    target_sparse = features["coarse_trajectories"][:, 0].clone()
    target_semantic = torch.zeros((2,), dtype=torch.long, device=device)

    try:
        planner.forward_train(
            features,
            target_sparse,
            target_semantic,
            target_trajectory_dense=None,
            dense_loss_enabled=True,
            dense_loss_lambda_p=0.1,
        )
    except ValueError as exc:
        assert "real 10 Hz dense target" in str(exc)
    else:
        raise AssertionError("dense supervision accepted a missing dense target")
