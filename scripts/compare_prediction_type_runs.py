from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(rows: Iterable[dict], key: str) -> float:
    values = []
    for row in rows:
        value = row.get(key, "")
        if value in (None, ""):
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return float(sum(values) / len(values)) if values else float("nan")


def _rate(rows: Iterable[dict], key: str) -> float:
    values = []
    for row in rows:
        value = str(row.get(key, "")).strip().lower()
        if value in {"true", "1", "1.0"}:
            values.append(1.0)
        elif value in {"false", "0", "0.0"}:
            values.append(0.0)
    return float(sum(values) / len(values)) if values else float("nan")


def _read_eval_csv(path: Path, miss_threshold_m: float) -> Dict[str, float]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        rows = list(csv.DictReader(fp))
    if not rows:
        raise RuntimeError(f"No evaluation rows found in {path}")
    min_fde = [float(row["minFDE_at_M_dense_all"]) for row in rows]
    return {
        "eval_samples": float(len(rows)),
        "selected_ADE": _mean(rows, "selected_ADE"),
        "selected_FDE": _mean(rows, "selected_FDE"),
        "minADE_dense": _mean(rows, "minADE_at_M_dense_all"),
        "minFDE_dense": _mean(rows, "minFDE_at_M_dense_all"),
        "miss_rate_dense": float(
            sum(value > miss_threshold_m for value in min_fde) / len(min_fde)
        ),
        "selected_mode_accuracy": _rate(rows, "selected_mode_correct"),
        "offroad_rate_dense": _rate(rows, "execution_offroad_proxy"),
    }


def _read_training(run_dir: Path) -> Dict[str, float | str]:
    result_path = run_dir / "train_result.json"
    result = _read_json(result_path)
    metrics = result.get("metrics") or {}
    cfg_path = run_dir / "resolved_config.json"
    resolved = _read_json(cfg_path) if cfg_path.is_file() else {}
    model_cfg = resolved.get("model") or {}
    return {
        "prediction_type": str(model_cfg.get("prediction_type", "sample")),
        "global_step": float(result.get("global_step", 0)),
        "train_loss": float(metrics.get("train/loss", float("nan"))),
        "val_loss": float(metrics.get("val/loss", float("nan"))),
        "val_prediction_loss": float(
            metrics.get("val/prediction_loss", metrics.get("val/trajectory_regression_loss", float("nan")))
        ),
        "val_x0_reconstruction_loss": float(
            metrics.get("val/x0_reconstruction_loss", metrics.get("val/trajectory_regression_loss", float("nan")))
        ),
        "val_noise_prediction_loss": float(
            metrics.get("val/noise_prediction_loss", float("nan"))
        ),
        "val_classification_loss": float(
            metrics.get("val/trajectory_classification_loss", float("nan"))
        ),
        "val_mode_match": float(metrics.get("val/w1_target_mode_match", float("nan"))),
    }


def _fmt(value) -> str:
    if isinstance(value, str):
        return value
    value = float(value)
    return "nan" if not math.isfinite(value) else f"{value:.6f}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare matched x0/sample and epsilon/noise diffusion runs."
    )
    parser.add_argument("--x0-run-dir", type=Path, required=True)
    parser.add_argument("--epsilon-run-dir", type=Path, required=True)
    parser.add_argument("--x0-eval-csv", type=Path, default=None)
    parser.add_argument("--epsilon-eval-csv", type=Path, default=None)
    parser.add_argument("--miss-threshold-m", type=float, default=2.0)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    x0 = _read_training(args.x0_run_dir)
    eps = _read_training(args.epsilon_run_dir)
    if args.x0_eval_csv is not None:
        x0.update(_read_eval_csv(args.x0_eval_csv, args.miss_threshold_m))
    if args.epsilon_eval_csv is not None:
        eps.update(_read_eval_csv(args.epsilon_eval_csv, args.miss_threshold_m))

    keys = [
        "prediction_type",
        "global_step",
        "train_loss",
        "val_loss",
        "val_prediction_loss",
        "val_x0_reconstruction_loss",
        "val_noise_prediction_loss",
        "val_classification_loss",
        "val_mode_match",
        "eval_samples",
        "selected_ADE",
        "selected_FDE",
        "minADE_dense",
        "minFDE_dense",
        "miss_rate_dense",
        "selected_mode_accuracy",
        "offroad_rate_dense",
    ]

    print("=" * 92)
    print("ALLMERGE DIFFUSION PARAMETERIZATION A/B")
    print("=" * 92)
    print(f"{'metric':34s} {'x0/sample':>18s} {'epsilon/noise':>18s}")
    print("-" * 92)
    for key in keys:
        if key not in x0 and key not in eps:
            continue
        print(f"{key:34s} {_fmt(x0.get(key, float('nan'))):>18s} {_fmt(eps.get(key, float('nan'))):>18s}")

    output = {"x0": x0, "epsilon": eps}
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"[compare] wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
