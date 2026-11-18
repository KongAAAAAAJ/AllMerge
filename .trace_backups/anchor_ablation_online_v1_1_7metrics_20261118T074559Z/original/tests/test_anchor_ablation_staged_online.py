from pathlib import Path

import torch

from evaluation.ablation.anchor_ablation.paper_metrics import (
    compute_anchor_metrics,
    geometry_valid_mask,
)


def test_a1_geometry_mask_uses_online_trajectory_geometry_not_traffic_mask():
    xy = torch.zeros((1, 3, 8, 2), dtype=torch.float32)
    xy[:, :, :, 0] = torch.arange(8, dtype=torch.float32)[None, None, :] * 1.0
    features = {"mode_valid_mask": torch.zeros((1, 3), dtype=torch.bool)}
    valid = geometry_valid_mask("a1", features, {}, xy)
    assert valid.shape == (1, 3)
    assert valid.all()


def test_seven_metric_contract_and_pdf_outputs_are_preserved():
    project = Path(__file__).resolve().parents[1]
    runner = (project / "evaluation/ablation/anchor_ablation/runner.py").read_text(encoding="utf-8")
    reporting = (project / "evaluation/ablation/anchor_ablation/reporting.py").read_text(encoding="utf-8")
    visual = (project / "evaluation/ablation/anchor_ablation/visualization.py").read_text(encoding="utf-8")
    expected = (
        "anchor_minade_m", "safe_ratio_pct", "trajectory_diversity_m",
        "control_diversity_normalized", "planning_minade_m",
        "preprojection_feasibility_pct", "projection_correction_m",
    )
    for key in expected:
        assert key in reporting
        assert key in runner
    assert 'choices=("anchors", "feasibility", "all")' in runner
    assert '"anchor_ablation.pdf"' in runner
    assert '"anchor_planning_ablation.pdf"' in runner
    assert "Times New Roman" in visual
    assert "pdf.fonttype" in visual


def test_anchor_metric_calculation_returns_four_paper_metrics():
    b, k = 2, 3
    xy = torch.zeros((b, k, 8, 2), dtype=torch.float32)
    for m in range(k):
        xy[:, m, :, 0] = torch.arange(8, dtype=torch.float32) + m * 0.2
    controls = torch.zeros((b, k, 40, 2), dtype=torch.float32)
    controls[:, :, :, 0] = torch.arange(k, dtype=torch.float32)[None, :, None] * 0.1
    expert = torch.zeros((b, 8, 2), dtype=torch.float32)
    valid = torch.ones((b, k), dtype=torch.bool)
    safety = {"hard_safe": torch.tensor([[True, False, True], [True, True, False]])}
    values = compute_anchor_metrics(
        xy=xy, controls=controls, expert_xy=expert, valid=valid,
        safety=safety, expected_anchor_count=k,
    )
    for key in ("anchor_minade_m", "safe_ratio", "trajectory_diversity_m",
                "control_diversity_normalized"):
        assert key in values
        assert values[key].shape == (b,)
