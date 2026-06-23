#!/usr/bin/env python
"""
eval_before_after.py — Measure and *show* what a fine-tune changed.

Two kinds of evidence (see docs/reports/01-first-reasoner.md):
  1) Qualitative  — prompt the model and dump its raw output, so you can SEE whether it now
                    reasons inside <think>...</think> before answering.
  2) Quantitative — score accuracy on a held-out slice of GSM8K (grade-school math word problems),
                    so you can MEASURE the change with a number.

Run it once on the base model and once on the tuned adapter, then compare:

    # before
    python scripts/eval_before_after.py --model unsloth/Qwen2.5-3B-Instruct --tag base --gsm8k 100
    # after
    python scripts/eval_before_after.py --model /workspace/outputs/exp1-qwen3b-reasoning/adapter --tag tuned --gsm8k 100

Writes <out>/eval_<tag>.json (accuracy + metadata) and <out>/transcripts_<tag>.txt (readable samples).
"""
import argparse
import json
import os
import re


# A few prompts to eyeball the reasoning behavior qualitatively.
QUALITATIVE_PROMPTS = [
    "Natalia sold clips to 48 friends in April, then sold half as many in May. "
    "How many clips did she sell altogether?",
    "A robot battery lasts 90 minutes. If it walks for 25 minutes, rests for 10, then walks again, "
    "how many more minutes of walking does it have left?",
    "What is 17 times 23? Show your work.",
]


def extract_final_number(text: str):
    """Pull the last number out of a string (handles commas, $, trailing period)."""
    nums = re.findall(r"-?\$?\d[\d,]*\.?\d*", text.replace(",", ""))
    if not nums:
        return None
    last = nums[-1].replace("$", "").rstrip(".")
    try:
        return float(last)
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="base model name OR a saved adapter dir")
    ap.add_argument("--tag", required=True, help="label for output files, e.g. 'base' or 'tuned'")
    ap.add_argument("--gsm8k", type=int, default=100, help="how many GSM8K test questions to score (0=skip)")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--out", default="/workspace/outputs/exp1-qwen3b-reasoning")
    ap.add_argument("--max-seq-length", type=int, default=2048)
    args = ap.parse_args()

    from unsloth import FastLanguageModel
    from datasets import load_dataset

    os.makedirs(args.out, exist_ok=True)

    print(f"Loading model: {args.model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=False,
    )
    FastLanguageModel.for_inference(model)   # Unsloth's 2x-faster inference mode

    def generate(question: str) -> str:
        messages = [{"role": "user", "content": question}]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        ).to(model.device)
        # Greedy decoding (do_sample=False) → deterministic, reproducible eval.
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        # decode only the newly generated tokens
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    transcripts = []
    think_flags = []   # track the ACTUAL model output, not the transcript template

    # --- qualitative ---
    print("\n=== Qualitative samples ===")
    for p in QUALITATIVE_PROMPTS:
        ans = generate(p)
        thinks = "<think>" in ans
        think_flags.append(thinks)
        transcripts.append(f"PROMPT:\n{p}\n\nOUTPUT (has <think>={thinks}):\n{ans}\n{'='*80}\n")
        print(f"- prompt done, <think> present: {thinks}")

    # --- quantitative: GSM8K ---
    accuracy = None
    if args.gsm8k > 0:
        print(f"\n=== GSM8K: scoring {args.gsm8k} questions ===")
        gsm = load_dataset("openai/gsm8k", "main", split=f"test[:{args.gsm8k}]")
        correct = 0
        for i, ex in enumerate(gsm):
            gold = extract_final_number(ex["answer"].split("####")[-1])
            out = generate(ex["question"])
            pred = extract_final_number(out)
            ok = (pred is not None and gold is not None and abs(pred - gold) < 1e-3)
            correct += int(ok)
            if i < 5:
                transcripts.append(
                    f"[GSM8K {i}] gold={gold} pred={pred} ok={ok}\nQ: {ex['question']}\nA: {out}\n{'='*80}\n"
                )
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{args.gsm8k}  running acc={correct/(i+1):.1%}")
        accuracy = correct / len(gsm)
        print(f"\nGSM8K accuracy [{args.tag}]: {accuracy:.1%}  ({correct}/{len(gsm)})")

    # --- save ---
    with open(os.path.join(args.out, f"transcripts_{args.tag}.txt"), "w") as f:
        f.writelines(transcripts)
    result = {
        "tag": args.tag,
        "model": args.model,
        "gsm8k_n": args.gsm8k,
        "gsm8k_accuracy": accuracy,
        "qualitative_think_rate": sum(think_flags) / len(think_flags) if think_flags else 0.0,
    }
    with open(os.path.join(args.out, f"eval_{args.tag}.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved eval_{args.tag}.json and transcripts_{args.tag}.txt to {args.out}")


if __name__ == "__main__":
    main()
