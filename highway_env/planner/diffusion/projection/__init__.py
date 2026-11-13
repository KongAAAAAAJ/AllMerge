"""Projection utilities migrated from the AllMerge GuideDistill worktree."""
from .matp_adapter import load_matp, project_w1
from .matp_w1 import (
    MatpW1Config,
    curvature_series,
    evaluate_w1_curvature_feasibility,
    project_w1_local,
)

__all__ = [
    "MatpW1Config",
    "curvature_series",
    "evaluate_w1_curvature_feasibility",
    "load_matp",
    "project_w1",
    "project_w1_local",
]
