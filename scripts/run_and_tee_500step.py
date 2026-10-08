#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tee AllMerge RL training logs, then optionally launch in-repo visual evaluation.

By default 500-step runs are evaluated automatically. For other step counts use
--visual-eval; --no-visual-eval disables it. Post-eval failures NEVER destroy an
already completed training run: they are logged and may be retried manually.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _arg(cmd: list[str], name: str, default: str | None = None) -> str | None:
    for index, part in enumerate(cmd):
        if part == name and index + 1 < len(cmd):
            return cmd[index + 1]
        if part.startswith(name + "="):
            return part.split("=", 1)[1]
    return default


def _post_eval(cmd: list[str], log: Path, num_states: int, visual_seed: int) -> None:
    checkpoint = _arg(cmd, "--output")
    pretrained = _arg(cmd, "--checkpoint")
    if not checkpoint or not pretrained:
        print("[visual-eval][SKIP] missing --output or --checkpoint")
        return
    ckpt_path = Path(checkpoint).resolve()
    pre_path = Path(pretrained).resolve()
    if not ckpt_path.is_file():
        print(f"[visual-eval][SKIP] trained checkpoint not found: {ckpt_path}")
        return
    if not pre_path.is_file():
        print(f"[visual-eval][SKIP] pretrain checkpoint not found: {pre_path}")
        return
    if "--fake-reward" in cmd:
        print("[visual-eval][SKIP] fake-reward does not support W4 online paired evaluation")
        return
    eval_script = REPO_ROOT / "scripts" / "evaluate_rl_training.py"
    output_dir = ckpt_path.with_suffix("").with_name(ckpt_path.stem + "_visual_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    fixed_csv = _arg(cmd, "--fixed-validation-csv")
    if fixed_csv is None:
        fixed_csv = str(ckpt_path.with_name(ckpt_path.stem + ".fixed_validation.csv"))
    eval_cmd = [
        sys.executable, str(eval_script),
        "--checkpoint", str(ckpt_path),
        "--pretrain-checkpoint", str(pre_path),
        "--train-log", str(log.resolve()),
        "--output-dir", str(output_dir),
        "--scenario", str(_arg(cmd, "--scenario", "curved")),
        "--num-states", str(max(9, num_states)),
        "--group-size", str(_arg(cmd, "--group-size", "48")),
        "--group-action", str(_arg(cmd, "--group-action", "3")),
        "--seed", str(_arg(cmd, "--seed", "7")),
        "--eta", str(_arg(cmd, "--eta", "0.02")),
        "--task-reward-type", str(_arg(cmd, "--task-reward-type", "progress_comfort")),
        "--constraint-names", str(_arg(cmd, "--constraint-names",
                           "collision,road,ttc,background_gap,teammate_gap")),
        "--constraint-residual-cap", str(_arg(cmd, "--constraint-residual-cap", "5.0")),
        "--fixed-validation-seed-offset", str(_arg(cmd, "--fixed-validation-seed-offset", "10000")),
        "--visualization-seed", str(visual_seed),
    ]
    if Path(fixed_csv).is_file():
        eval_cmd.extend(["--fixed-validation-csv", str(Path(fixed_csv).resolve())])
    print(f"[visual-eval] training finished, launching {eval_script.relative_to(REPO_ROOT)}")
    proc = subprocess.run(eval_cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    (output_dir / "post_eval.log").write_text(proc.stdout, encoding="utf-8")
    print(proc.stdout)
    if proc.returncode:
        print(f"[visual-eval][WARN] evaluation exit={proc.returncode}; "
              "training checkpoint is intact; re-run scripts/evaluate_rl_training.py")
    else:
        print(f"[visual-eval][PASS] figures: {output_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--visual-eval", dest="visual_eval", action="store_true", default=None)
    mode.add_argument("--no-visual-eval", dest="visual_eval", action="store_false")
    parser.add_argument("--visual-eval-states", type=int, default=16)
    parser.add_argument("--visualization-seed", type=int, default=0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cmd = list(args.command)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        raise SystemExit("missing command after --")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    print("[run]", " ".join(cmd))
    print(f"[tee] {args.log}")
    with args.log.open("w", encoding="utf-8", newline="\n") as log:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        status = int(process.wait())
    if status != 0:
        return status
    if args.visual_eval is None:
        # Existing 500-step .bat files use this tee wrapper, so they become
        # automatically visualized without modifying baseline trainer semantics.
        enabled = int(_arg(cmd, "--steps", "0") or "0") >= 500
    else:
        enabled = args.visual_eval
    if enabled:
        try:
            _post_eval(cmd, args.log, args.visual_eval_states, args.visualization_seed)
        except Exception as exc:
            print(f"[visual-eval][WARN] {type(exc).__name__}: {exc}; training succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
