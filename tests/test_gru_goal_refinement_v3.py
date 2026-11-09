from __future__ import annotations

import torch

from highway_env.planner.diffusion.config import build_structured_diffusion_config
from highway_env.planner.diffusion.structured_model import StructuredDiffusionPlanner
from highway_env.planner.diffusion.tensor_adapter import PlannerTensorAdapter


def features(cfg):
    b = 2
    return {
        "ego_state": torch.zeros(b, cfg.ego_dim),
        "agent_states": torch.zeros(b, cfg.max_agents, cfg.agent_dim),
        "agent_valid_mask": torch.ones(b, cfg.max_agents, dtype=torch.bool),
        "map_polylines": torch.zeros(b, cfg.max_map_polylines, cfg.map_points, cfg.map_dim),
        "map_valid_mask": torch.ones(b, cfg.max_map_polylines, dtype=torch.bool),
        "target_point": torch.zeros(b, 2),
        "target_lane_polyline": torch.zeros(b, cfg.map_points, cfg.map_dim),
        "coarse_trajectories": torch.randn(b, cfg.num_modes, cfg.horizon_steps, 2),
        "mode_valid_mask": torch.ones(b, cfg.num_modes, dtype=torch.bool),
    }


def run(kind):
    torch.manual_seed(7)
    cfg = build_structured_diffusion_config(reg_head_type=kind, dropout=0.0)
    adapter = PlannerTensorAdapter(cfg, torch.device("cpu"))
    model = StructuredDiffusionPlanner(cfg, adapter)
    f = features(cfg)
    target = f["coarse_trajectories"][:, 0].clone()
    semantic = torch.zeros(target.shape[0], dtype=torch.long)
    out = model.forward_train(f, target, target_semantic=semantic)
    assert torch.isfinite(out["loss"])
    with torch.no_grad():
        pred = model.infer_multimodal(f)
    assert pred["trajectory_candidates"].shape == (
        target.shape[0], cfg.num_modes, cfg.horizon_steps, 2
    )
    if kind == "gru":
        assert model.denoiser.gru_refine is not None
        assert model.denoiser.gru_refine_out is not None
        # New refinement starts from identity, avoiding the unstable random
        # trajectory perturbation seen in the previous direct-GRU head.
        assert torch.count_nonzero(model.denoiser.gru_refine_out.weight) == 0
    return float(out["loss"])


if __name__ == "__main__":
    mlp = run("mlp")
    gru = run("gru")
    print(f"PASS goal-refinement smoke | mlp_loss={mlp:.6f} gru_loss={gru:.6f}")
