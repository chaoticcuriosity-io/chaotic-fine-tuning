#!/usr/bin/env python
"""plot_reward.py — plot GRPO reward_history.json (mean group reward per step)."""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir containing reward_history.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="GRPO mean reward")
    args = ap.parse_args()

    path = os.path.join(args.run, "reward_history.json") if os.path.isdir(args.run) else args.run
    hist = json.load(open(path))
    steps = [h["step"] for h in hist]
    rewards = [h["reward"] for h in hist]

    plt.figure(figsize=(8, 5))
    plt.plot(steps, rewards, marker="o", ms=3, linewidth=1.5)
    plt.axhline(2.0, ls="--", c="gray", alpha=0.6, label="max correctness reward (2.0)")
    plt.xlabel("GRPO step")
    plt.ylabel("mean group reward")
    plt.title(args.title)
    plt.ylim(0, 2.7)
    plt.grid(True, alpha=0.3)
    plt.legend()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
