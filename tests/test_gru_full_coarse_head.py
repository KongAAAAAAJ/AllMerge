from __future__ import annotations

import torch

from highway_env.planner.diffusion.config import build_structured_diffusion_config
from highway_env.planner.diffusion.structured_model import StructuredDiffusionPlanner
from highway_env.planner.diffusion.tensor_adapter import PlannerTensorAdapter


def _features(cfg, device):
    b = 2
    coarse = torch.randn(b, cfg.num_modes, cfg.horizon_steps, 2, device=device)
    return {
        "ego_state": torch.zeros(b, cfg.ego_dim, device=device),
        "agent_states": torch.zeros(b, cfg.max_agents, cfg.agent_dim, device=device),
        "agent_valid_mask": torch.ones(b, cfg.max_agents, dtype=torch.bool, device=device),
        "map_polylines": torch.zeros(b, cfg.max_map_polylines, cfg.map_points, cfg.map_dim, device=device),
        "map_valid_mask": torch.ones(b, cfg.max_map_polylines, dtype=torch.bool, device=device),
        "target_point": torch.zeros(b, 2, device=device),
        "target_lane_polyline": torch.zeros(b, cfg.map_points, cfg.map_dim, device=device),
        "coarse_trajectories": coarse,
        "mode_valid_mask": torch.ones(b, cfg.num_modes, dtype=torch.bool, device=device),
    }


def run(kind: str):
    device = torch.device("cpu")
    cfg = build_structured_diffusion_config(reg_head_type=kind, dropout=0.0)
    adapter = PlannerTensorAdapter(cfg, device)
    model = StructuredDiffusionPlanner(cfg, adapter).to(device)
    features = _features(cfg, device)
    target = features["coarse_trajectories"][:, 0].clone()
    semantic = torch.zeros(target.shape[0], dtype=torch.long, device=device)
    out = model.forward_train(features, target, target_semantic=semantic)
    assert torch.isfinite(out["loss"])
    with torch.no_grad():
        pred = model.infer_multimodal(features)
    assert pred["trajectory_candidates"].shape == (
        target.shape[0], cfg.num_modes, cfg.horizon_steps, 2
    )
    return float(out["loss"])


if __name__ == "__main__":
    mlp = run("mlp")
    gru = run("gru")
    print(f"PASS full-coarse reg-head smoke | mlp_loss={mlp:.6f} gru_loss={gru:.6f}")
