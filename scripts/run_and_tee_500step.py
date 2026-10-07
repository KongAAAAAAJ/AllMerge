#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BASELINE_CONVERGENCE_RUN_TEE_500STEP_V1_20261007
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("command", nargs=argparse.REMAINDER)
    args = p.parse_args()

    cmd = list(args.command)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        raise SystemExit("missing command after --")

    args.log.parent.mkdir(parents=True, exist_ok=True)
    print("[run]", " ".join(cmd))
    print(f"[tee] {args.log}")
    with args.log.open("w", encoding="utf-8", newline="\n") as log:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        return int(proc.wait())


if __name__ == "__main__":
    raise SystemExit(main())
