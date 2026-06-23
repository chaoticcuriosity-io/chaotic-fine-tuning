#!/usr/bin/env python
"""
plot_loss.py — Turn a training run's loss_history.json into a PNG curve.

A falling loss curve is the first sign the model is learning the data (see docs/reports/00-primer.md §8).
Supports overlaying multiple runs for the ablation report (03-turning-the-knobs.md).

Usage:
    python scripts/plot_loss.py --run /workspace/outputs/exp1-qwen3b-reasoning --out docs/reports/assets/exp1-loss.png
    python scripts/plot_loss.py --run r16=/workspace/outputs/exp1 --run r64=/workspace/outputs/exp2 --out compare.png
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")   # headless: no display on the Spark
import matplotlib.pyplot as plt


def load_history(path: str):
    # accept either a run dir (containing loss_history.json) or the json file itself
    if os.path.isdir(path):
        path = os.path.join(path, "loss_history.json")
    with open(path) as f:
        hist = json.load(f)
    return [h["step"] for h in hist], [h["loss"] for h in hist]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    help="run dir, or 'label=dir' to overlay multiple runs")
    ap.add_argument("--out", required=True, help="output PNG path")
    ap.add_argument("--title", default="Training loss")
    args = ap.parse_args()

    plt.figure(figsize=(8, 5))
    for spec in args.run:
        label, _, path = spec.partition("=") if "=" in spec else (None, None, spec)
        steps, losses = load_history(path)
        plt.plot(steps, losses, label=label or "loss", linewidth=1.5)

    plt.xlabel("training step")
    plt.ylabel("loss (next-token prediction error)")
    plt.title(args.title)
    plt.grid(True, alpha=0.3)
    if any("=" in s for s in args.run):
        plt.legend()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
