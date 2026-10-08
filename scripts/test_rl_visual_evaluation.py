"""Offline smoke for the in-repository RL visualization module.

No simulator/checkpoints needed; does not test the paired checkpoint forward pass.
"""
from __future__ import annotations
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.rl_training_report import parse_train_log, parse_fixed_validation, plot_training_curves


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="allmerge_rl_visual_") as directory:
        root = Path(directory)
        log = root / "train.log"
        lines = []
        for step in range(1, 31):
            lines.append(
                f"[step {step:03d}/30] scenario=curved reward/task_reward_mean={0.4+step/1000:.5f} "
                f"loss={0.5-step/1000:.5f} policy_loss={0.4-step/1000:.5f} "
                f"reference_kl={step/300:.5f} constraint/feasible_fraction=0.1 "
                f"constraint/road_violation_mean={3-step/50:.5f} "
                f"constraint/road_violation_fraction=0.9 "
                f"constraint/max_violation_mean={3-step/50:.5f}"
            )
        log.write_text("\n".join(lines), encoding="utf-8")
        fixed = root / "fixed.csv"
        fixed.write_text(
            "step,fixed_validation/current_reward_mean," \
            "fixed_validation/current_constraint_road_violation_mean_mean," \
            "fixed_validation/constraint_road_violation_change_mean\n" \
            "0,0.4,3.2,0\n10,0.41,3.19,-0.01\n20,0.42,3.18,-0.02\n30,0.43,3.17,-0.03\n",
            encoding="utf-8",
        )
        train_rows = parse_train_log(log)
        fixed_rows = parse_fixed_validation(fixed)
        assert len(train_rows) == 30 and len(fixed_rows) == 4
        names = [Path(p).name for p in plot_training_curves(train_rows, fixed_rows, root / "figures")]
        assert "03_constraint_violation_magnitude.png" in names
        assert "04_fixed_validation_violation_magnitude.png" in names
        assert "01_reward_curve.png" in names
        assert "02_loss_curve.png" in names
        assert "05_constraint_violation_rates_optional.png" in names
        assert all((root / "figures" / name).stat().st_size > 10000 for name in names)
        print(f"[PASS] curve renderer: {len(names)} figures; parsed 30 train + 4 fixed-val rows")


if __name__ == "__main__":
    main()
