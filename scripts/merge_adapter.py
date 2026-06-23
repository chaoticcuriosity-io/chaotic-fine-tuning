#!/usr/bin/env python
"""merge_adapter.py — bake a LoRA adapter into its base, save a standalone 16-bit model.
Used to turn an SFT'd reasoner into a starting point for GRPO (RL needs a plain model, not a stacked adapter).

    python scripts/merge_adapter.py --adapter outputs/exp3-qwen3b-gsm8k/adapter --out outputs/exp3-qwen3b-gsm8k/merged
"""
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max-seq-length", type=int, default=2048)
args = ap.parse_args()

from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=args.adapter, max_seq_length=args.max_seq_length, dtype=None, load_in_4bit=False)
model.save_pretrained_merged(args.out, tokenizer, save_method="merged_16bit")
print(f"MERGED -> {args.out}")
