"""Raw-episode TRACE A2 safety scorer train/eval with reusable candidate cache.

The scorer architecture, safety teacher, soft-label objective and Top-R metrics
are unchanged. Candidate controls/trajectories and deterministic teacher labels
are precomputed once, then memory-mapped during training.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from highway_env.planner.diffusion.config.experiment import resolve_repo_file
from highway_env.planner.adaptive_anchors import AdaptiveCandidateBuilder
from .model import AnchorSafetyScorer, select_anchors
from .training_data import build_cached_loader, prepare_split_cache



def _resolve_num_selected(cfg):
    """Resolve the A2 Top-R metric K from the canonical control diffusion config."""
    diffusion_config = resolve_repo_file(
        cfg.get("diffusion_config", "configs/train/diffusion.yaml")
    )
    from learning.config import load_yaml_config
    diffusion_cfg = load_yaml_config(diffusion_config)
    num_selected = int(
        (((diffusion_cfg.get("control") or {}).get("model") or {}).get("num_modes", 10))
    )
    if num_selected <= 0:
        raise ValueError(f"control.model.num_modes must be positive, got {num_selected}")
    return num_selected

def run_epoch(model, loader, cfg, device, *, optimizer=None, max_batches=None, epoch=0):
    training = optimizer is not None
    model.train(training)
    if hasattr(loader.batch_sampler, "set_epoch"):
        loader.batch_sampler.set_epoch(epoch)

    metric_names = (
        "loss", "safe_ratio", "oracle_safe_ratio", "top_r_safe_recall",
        "label_entropy", "score_gap",
    )
    metric_sums = torch.zeros(len(metric_names), dtype=torch.float64, device=device)
    count = 0
    top_r, ka = int(cfg["top_r"]), int(cfg["num_selected"])
    epoch_started = time.perf_counter()
    data_wait_s = 0.0
    step_s = 0.0
    wait_started = epoch_started

    for batch_index, (features, cached) in enumerate(loader):
        batch_started = time.perf_counter()
        data_wait_s += batch_started - wait_started
        if max_batches is not None and batch_index >= max_batches:
            break

        features = {name: value.to(device, non_blocking=True)
                    for name, value in features.items()}
        control = cached["coarse_controls"].to(device, non_blocking=True)
        xy = cached["coarse_trajectories"].to(device, non_blocking=True)
        valid = cached["mode_valid_mask"].to(device, non_blocking=True)
        target = cached["teacher_target"].to(device, non_blocking=True)
        hard_safe = cached["safety_hard_safe"].to(device, non_blocking=True)

        step_started = time.perf_counter()
        with torch.set_grad_enabled(training):
            scores = model(features, control, xy)
            log_prob = F.log_softmax(scores.masked_fill(~valid, -torch.inf), dim=-1)
            loss = -(target * log_prob.masked_fill(~valid, 0.0)).sum(-1).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite anchor-scoring loss")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(cfg.get("grad_clip", 1.0))
                )
                optimizer.step()

        with torch.no_grad():
            idx = select_anchors(
                scores.detach(), control, xy, valid,
                top_r=top_r,
                num_selected=ka,
                trajectory_weight=float(cfg.get("trajectory_weight", 1.0)),
                control_weight=float(cfg.get("control_weight", 1.0)),
            )
            selected_safe = hard_safe.gather(1, idx)
            sorted_ids = scores.detach().masked_fill(~valid, -torch.inf).topk(
                top_r, dim=1
            ).indices
            total_safe = hard_safe.sum(1).clamp_min(1)
            recall = hard_safe.gather(1, sorted_ids).sum(1) / total_safe
            optimal_safe = torch.minimum(
                hard_safe.sum(1), torch.full_like(total_safe, ka)
            ).float() / ka
            values = torch.stack((
                loss.detach(),
                selected_safe.float().mean(),
                optimal_safe.mean(),
                recall.mean(),
                (-(target * target.clamp_min(1e-12).log()).sum(1)).mean(),
                scores.detach().masked_fill(~valid, 0).std(dim=1).mean(),
            )).to(dtype=torch.float64)
            batch_n = int(xy.shape[0])
            count += batch_n
            metric_sums += values * batch_n

        step_s += time.perf_counter() - step_started
        wait_started = time.perf_counter()

    if count == 0:
        raise RuntimeError("No samples processed")
    # One device-to-host synchronization per epoch, rather than several
    # scalar synchronizations on every training batch.
    metric_values = (metric_sums / count).detach().cpu().tolist()
    elapsed = max(time.perf_counter() - epoch_started, 1e-9)
    result = dict(zip(metric_names, metric_values))
    result.update({
        "samples": count,
        "elapsed_sec": elapsed,
        "data_wait_sec": data_wait_s,
        "model_step_dispatch_sec": step_s,
        "samples_per_sec": count / elapsed,
    })
    return result


def _save_history(path: Path, history):
    if not history:
        return
    fields = list(history[0])
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train/evaluate TRACE adaptive safety scorer V3 (shared 108-candidate pool)")
    parser.add_argument("--config", default="configs/train/adaptive_anchors.yaml")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--precompute-batch-size", type=int, default=None)
    parser.add_argument("--precompute-device", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--prepare-cache-only", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--checkpoint", default=None,
                        help="Warm-start model weights only; optimizer state is reset.")
    parser.add_argument("--resume", default=None,
                        help="Resume a V3 checkpoint including optimizer/history state.")
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(resolve_repo_file(args.config).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Scorer config must be a mapping")
    candidate_builder = AdaptiveCandidateBuilder(cfg.get("candidate_generation"))
    cfg['num_selected'] = _resolve_num_selected(cfg)
    cfg['top_r'] = max(cfg['num_selected'], int(round(
        candidate_builder.num_candidates * float(cfg.get('top_r_fraction', 0.5555556)))))
    if not 1 <= cfg['num_selected'] <= cfg['top_r'] <= candidate_builder.num_candidates:
        raise ValueError('A2 requires 1 <= num_modes <= Top-R <= candidate pool size')
    candidate_generation_resolved = dict(getattr(
        candidate_builder, "resolved_config",
        getattr(getattr(candidate_builder, "base", None), "config",
                cfg.get("candidate_generation") or {}),
    ))
    if args.dataset_root:
        cfg["dataset_root"] = args.dataset_root
    if args.max_samples is not None:
        cfg["max_samples"] = args.max_samples
    if args.precompute_batch_size is not None:
        cfg["precompute_batch_size"] = args.precompute_batch_size
    if args.precompute_device:
        cfg["precompute_device"] = args.precompute_device

    seed = int(cfg.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    dataset_root = resolve_repo_file(cfg["dataset_root"])
    output_dir = resolve_repo_file(
        args.output_dir or cfg.get("output_dir", "outputs/anchor_ablation/ckpt")
    )
    cache_root = resolve_repo_file(
        args.cache_dir or cfg.get("cache_dir", "outputs/anchor_ablation/cache_v3")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    needed_splits = [cfg.get("val_split", "val")]
    if not args.evaluate_only:
        needed_splits.insert(0, cfg.get("train_split", "train"))
    cache_dirs = {}
    for split in needed_splits:
        cache_dirs[split], _ = prepare_split_cache(
            dataset_root, split, cfg, cache_root, rebuild=args.rebuild_cache
        )
    if args.prepare_cache_only:
        print("[A2-cache] preprocessing completed; training was not started.")
        return

    train_loader = None
    train_sampler = None
    if not args.evaluate_only:
        train_loader, train_sampler = build_cached_loader(
            dataset_root, cfg.get("train_split", "train"), cfg,
            cache_dirs[cfg.get("train_split", "train")], shuffle=True,
        )
    val_loader, _ = build_cached_loader(
        dataset_root, cfg.get("val_split", "val"), cfg,
        cache_dirs[cfg.get("val_split", "val")], shuffle=False,
    )

    model_cfg = cfg.get("model", {})
    model = AnchorSafetyScorer(model_cfg).to(device)
    if args.checkpoint and args.resume:
        raise ValueError("Use either --checkpoint (warm start) or --resume, not both")

    saved_resume = None
    if args.resume:
        saved_resume = torch.load(
            resolve_repo_file(args.resume), map_location="cpu", weights_only=True
        )
        if saved_resume.get("architecture") != "TRACE_Adaptive_Anchor_Safety_Scorer_V1":
            raise ValueError("Incompatible resume checkpoint")
        model.load_state_dict(saved_resume["model_state"], strict=True)
    elif args.checkpoint:
        saved = torch.load(
            resolve_repo_file(args.checkpoint), map_location="cpu", weights_only=True
        )
        if saved.get("architecture") != "TRACE_Adaptive_Anchor_Safety_Scorer_V1":
            raise ValueError("Incompatible checkpoint")
        model.load_state_dict(saved["model_state"], strict=True)

    if args.evaluate_only:
        metrics = run_epoch(
            model, val_loader, cfg, device, max_batches=args.max_batches, epoch=0
        )
        (output_dir / "evaluation.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        print(json.dumps(metrics, indent=2))
        return

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("lr", 1e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    start_epoch = 1
    best = float("inf")
    history = []
    if saved_resume is not None:
        if "optimizer_state_dict" not in saved_resume:
            raise ValueError("Resume checkpoint has no optimizer_state_dict; use --checkpoint for warm start")
        optimizer.load_state_dict(saved_resume["optimizer_state_dict"])
        start_epoch = int(saved_resume["epoch"]) + 1
        best = float(saved_resume.get("best_val_loss", float("inf")))
        history = list(saved_resume.get("history", []))

    epochs = int(args.max_epochs if args.max_epochs is not None else cfg.get("epochs", 30))
    if start_epoch > epochs:
        raise ValueError(f"Resume epoch {start_epoch - 1} already reaches requested max_epochs={epochs}")

    for epoch in range(start_epoch, epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, cfg, device, optimizer=optimizer,
            max_batches=args.max_batches, epoch=epoch,
        )
        val_metrics = run_epoch(
            model, val_loader, cfg, device,
            max_batches=args.max_batches, epoch=epoch,
        )
        row = {"epoch": epoch}
        row.update({"train_" + key: value for key, value in train_metrics.items()})
        row.update({"val_" + key: value for key, value in val_metrics.items()})
        history.append(row)

        ckpt = {
            "architecture": "TRACE_Adaptive_Anchor_Safety_Scorer_V1",
            "model_config": model_cfg,
            "candidate_generation": candidate_generation_resolved,
            "model_state": model.state_dict(),
            "epoch": epoch,
            "temperature": float(cfg.get("temperature", 0.3)),
            "teacher_config": cfg.get("teacher", {}),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": min(best, float(val_metrics["loss"])),
            "history": history,
            "training_pipeline": "TRACE_A2_SCORER_TRAINING_V3_SHARED_POOL",
        }
        torch.save(ckpt, output_dir / "last.pt")
        if val_metrics["loss"] < best:
            best = float(val_metrics["loss"])
            ckpt["best_val_loss"] = best
            torch.save(ckpt, output_dir / "best.pt")
        _save_history(output_dir / "history.csv", history)

        print(
            f"[A2] epoch={epoch}/{epochs} "
            f"train_CE={train_metrics['loss']:.5f} "
            f"val_CE={val_metrics['loss']:.5f} "
            f"val_safe={val_metrics['safe_ratio']:.3f} "
            f"oracle={val_metrics['oracle_safe_ratio']:.3f} "
            f"train_rate={train_metrics['samples_per_sec']:.1f} samples/s "
            f"train_time={train_metrics['elapsed_sec']/60:.1f} min",
            flush=True,
        )

    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    print(f"[A2] best scorer: {output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
