#!/usr/bin/env python
"""
train_grpo.py — Reinforcement learning for reasoning, with GRPO.

Where SFT (train_lora.py) only says "imitate these examples," GRPO (Group Relative Policy Optimization) says
"try several answers, and I'll reward the ones that are actually correct." It's the method DeepSeek used to
build R1 — and it's a close cousin of the PPO the sister `g1-humanoid-rl` project uses to teach a robot to
walk: in both, a *reward signal* shapes a *policy*. See docs/reports/05-rl-the-reasoner.md.

How GRPO works each step:
  1. For each question, sample a GROUP of G candidate answers from the current model.
  2. Score each with reward functions (here: is the final number correct? did it use <think>?).
  3. Within the group, answers better than the group average are reinforced; worse ones are discouraged.
     (The "group relative" baseline is what removes PPO's separate value network — simpler, less memory.)

Usage (inside unsloth-dev):
    python scripts/train_grpo.py --steps 50 --out /workspace/outputs/exp5-grpo-3b
"""
import argparse
import json
import os
import re


def extract_number(text: str):
    text = text.split("</think>")[-1]  # prefer the post-reasoning final answer
    nums = re.findall(r"-?\$?\d[\d,]*\.?\d*", text.replace(",", ""))
    if not nums:
        return None
    try:
        return float(nums[-1].replace("$", "").rstrip("."))
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen2.5-3B-Instruct")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--num-generations", type=int, default=8, help="group size G")
    ap.add_argument("--prompts-per-step", type=int, default=1,
                    help="unique prompts averaged per step; higher = smoother (less noisy) reward curve")
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--max-completion-length", type=int, default=0, help="0 = max_seq_length - 256")
    ap.add_argument("--max-samples", type=int, default=1000)
    ap.add_argument("--skip", type=int, default=0,
                    help="skip the first N GSM8K train examples (use to avoid overlap with SFT data)")
    ap.add_argument("--out", default="/workspace/outputs/exp5-grpo-3b")
    ap.add_argument("--max-seq-length", type=int, default=1024)
    args = ap.parse_args()

    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from trl import GRPOConfig, GRPOTrainer

    os.makedirs(args.out, exist_ok=True)

    print(f"[1/4] Loading {args.model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        fast_inference=False,   # HF generation (no vLLM dependency); set True if vLLM is available for speed
        max_lora_rank=16,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=16, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=3407,
    )

    SYSTEM = ("Reason *briefly* inside <think> </think> — a few short steps, no second-guessing — "
              "then give the final answer. End with 'The answer is <number>'.")

    print(f"[2/4] Loading GSM8K [:{args.max_samples}]")
    def prep(ex):
        return {"prompt": [{"role": "system", "content": SYSTEM},
                           {"role": "user", "content": ex["question"]}],
                "answer": extract_number(ex["answer"].split("####")[-1])}
    lo, hi = args.skip, args.skip + args.max_samples
    dataset = load_dataset("openai/gsm8k", "main", split=f"train[{lo}:{hi}]").map(prep)

    # --- reward functions: the heart of GRPO ---
    def correctness_reward(prompts, completions, answer, **kwargs):
        out = []
        for comp, gold in zip(completions, answer):
            pred = extract_number(comp[0]["content"])
            out.append(2.0 if (pred is not None and gold is not None and abs(pred - gold) < 1e-3) else 0.0)
        return out

    def format_reward(completions, **kwargs):
        out = []
        for comp in completions:
            t = comp[0]["content"]
            out.append(0.5 if ("<think>" in t and "</think>" in t) else 0.0)
        return out

    print(f"[3/4] GRPO training: {args.steps} steps, G={args.num_generations}")
    max_completion = args.max_completion_length or (args.max_seq_length - 256)
    cfg = GRPOConfig(
        learning_rate=args.lr,
        per_device_train_batch_size=args.prompts_per_step * args.num_generations,
        gradient_accumulation_steps=1,
        num_generations=args.num_generations,
        max_prompt_length=256,
        max_completion_length=max_completion,
        max_steps=args.steps,
        logging_steps=1,
        save_steps=25,                      # checkpoint periodically so a long run is recoverable
        save_total_limit=2,
        use_vllm=False,                     # no vLLM on this image → slower; fine for a long background run
        optim="adamw_8bit",
        seed=3407,
        output_dir=os.path.join(args.out, "trainer"),
        report_to="none",
    )
    trainer = GRPOTrainer(
        model=model, processing_class=tokenizer,
        reward_funcs=[correctness_reward, format_reward],
        args=cfg, train_dataset=dataset,
    )
    trainer.train()

    print("[4/4] Saving adapter + reward history")
    model.save_pretrained(os.path.join(args.out, "adapter"))
    tokenizer.save_pretrained(os.path.join(args.out, "adapter"))
    hist = [{"step": h.get("step"), "reward": h.get("reward")}
            for h in trainer.state.log_history if "reward" in h]
    with open(os.path.join(args.out, "reward_history.json"), "w") as f:
        json.dump(hist, f, indent=2)
    print(f"Done. Final reward: {hist[-1]['reward'] if hist else 'n/a'}")


if __name__ == "__main__":
    main()
