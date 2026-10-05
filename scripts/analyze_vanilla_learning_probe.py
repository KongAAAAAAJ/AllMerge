from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("csv", type=Path)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    if df.empty:
        raise SystemExit("empty probe CSV")
    out = args.output or args.csv.with_suffix(".png")

    step = df["step"].to_numpy()
    reward = df["probe/current_reward_mean"].to_numpy()
    frozen = df["probe/frozen_reward_mean"].to_numpy()
    gain = df["probe/reward_gain"].to_numpy()
    ema = df["probe/reward_gain_ema"].to_numpy()

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(step, reward, marker="o", label="Current policy reward")
    ax.plot(step, frozen, linestyle="--", label="Frozen pretrained reward")
    ax.set_xlabel("GRPO update step")
    ax.set_ylabel("Fixed-state task reward")
    ax.set_title("Vanilla GRPO Fixed-State Learnability Probe")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)

    gain_out = out.with_name(out.stem + "_gain" + out.suffix)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.axhline(0.0, linewidth=1.0)
    ax.plot(step, gain, marker="o", label="Reward gain vs frozen")
    ax.plot(step, ema, linewidth=2.0, label="Gain EMA")
    ax.set_xlabel("GRPO update step")
    ax.set_ylabel("Reward gain")
    ax.set_title("Vanilla GRPO Reward Gain Trend")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(gain_out, dpi=180)
    plt.close(fig)

    print(f"reward_plot={out}")
    print(f"gain_plot={gain_out}")
    print(f"initial_reward={reward[0]:.8f}")
    print(f"final_reward={reward[-1]:.8f}")
    print(f"final_gain={gain[-1]:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
