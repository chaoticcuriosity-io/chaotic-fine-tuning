#!/usr/bin/env python
"""
train_lora.py — Config-driven LoRA / QLoRA fine-tuning with Unsloth.

Teaches a base chat model a new behavior (here: step-by-step reasoning) by training a small LoRA adapter
while the billions of original weights stay frozen. See docs/reports/00-primer.md for the concepts.

Usage (inside the unsloth-dev container on the Spark):
    python scripts/train_lora.py --config configs/exp1.yaml

Outputs (under output_dir):
    adapter/                 the trained LoRA adapter (small — tens of MB)
    loss_history.json        per-step training loss, for scripts/plot_loss.py
    run_config.json          the exact resolved config used (reproducibility)
"""
import argparse
import json
import os

import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def format_dataset(dataset, tokenizer, cfg):
    """Turn each example into a single 'text' string using the model's chat template.

    Two formats (set `dataset_format` in the config; default 'auto'):
      - 'auto'  : datasets with a 'messages' or 'conversations' field (e.g. Mixture-of-Thoughts),
                  where the assistant turn already contains the reasoning.
      - 'gsm8k' : openai/gsm8k, with 'question' + 'answer' columns. The answer is concise grade-school
                  step-by-step reasoning ending in '#### <final>'. We wrap the reasoning in <think>...</think>
                  and state the final answer — teaching the SAME reasoning format as Experiment 1, but with
                  *concise, matched-difficulty* content. This is the only change vs Experiment 1.
    """
    cols = dataset.column_names
    fmt = cfg.get("dataset_format", "auto")

    def to_msgs(example):
        if fmt == "gsm8k":
            ans = example["answer"]
            reasoning, _, final = ans.partition("####")
            reasoning = reasoning.strip()
            final = final.strip()
            assistant = f"<think>\n{reasoning}\n</think>\n\nThe answer is **{final}**."
            return [{"role": "user", "content": example["question"]},
                    {"role": "assistant", "content": assistant}]
        if example.get("messages"):
            return example["messages"]
        if example.get("conversations"):
            role_map = {"human": "user", "gpt": "assistant", "system": "system"}
            return [{"role": role_map.get(m.get("from"), m.get("from", "user")),
                     "content": m.get("value", m.get("content", ""))}
                    for m in example["conversations"]]
        raise KeyError(
            f"Don't know how to format this dataset (format={fmt}). Columns: {cols}. "
            "Expected 'messages'/'conversations', or set dataset_format: gsm8k."
        )

    def to_text(example):
        text = tokenizer.apply_chat_template(to_msgs(example), tokenize=False, add_generation_prompt=False)
        return {"text": text}

    return dataset.map(to_text, remove_columns=cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    # Import unsloth FIRST — it patches transformers/torch for speed and must come before them.
    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    os.makedirs(cfg["output_dir"], exist_ok=True)

    # 1) Load the base model (frozen) + tokenizer.
    print(f"[1/5] Loading base model: {cfg['model_name']} (load_in_4bit={cfg['load_in_4bit']})")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_name"],
        max_seq_length=cfg["max_seq_length"],
        dtype=None,                       # auto: bf16 on Blackwell
        load_in_4bit=cfg["load_in_4bit"],
    )

    # 2) Attach the LoRA adapter — the only thing that will learn.
    print(f"[2/5] Attaching LoRA adapter (r={cfg['lora_r']}, alpha={cfg['lora_alpha']})")
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora_r"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",   # saves memory on long sequences
        random_state=cfg["seed"],
    )

    # 3) Load + format the dataset.
    print(f"[3/5] Loading dataset: {cfg['dataset']} [{cfg.get('dataset_config')}]")
    split = "train"
    if cfg.get("max_samples"):
        split = f"train[:{cfg['max_samples']}]"
    dataset = load_dataset(cfg["dataset"], cfg.get("dataset_config"), split=split)
    dataset = format_dataset(dataset, tokenizer, cfg)
    print(f"      {len(dataset)} examples ready. Sample (first 400 chars):")
    print("      " + dataset[0]["text"][:400].replace("\n", "\n      "))

    # 4) Train.
    print("[4/5] Training...")
    sft_args = SFTConfig(
        dataset_text_field="text",
        max_seq_length=cfg["max_seq_length"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        warmup_steps=cfg["warmup_steps"],
        num_train_epochs=cfg["num_train_epochs"] if not cfg["max_steps"] else 1,
        max_steps=cfg["max_steps"] if cfg["max_steps"] else -1,
        learning_rate=cfg["learning_rate"],
        logging_steps=cfg["logging_steps"],
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=cfg["seed"],
        output_dir=os.path.join(cfg["output_dir"], "trainer"),
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=sft_args,
    )
    stats = trainer.train()

    # 5) Save adapter + loss history + the resolved config.
    print("[5/5] Saving adapter and logs...")
    adapter_dir = os.path.join(cfg["output_dir"], "adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    loss_history = [
        {"step": h["step"], "loss": h["loss"]}
        for h in trainer.state.log_history if "loss" in h
    ]
    with open(os.path.join(cfg["output_dir"], "loss_history.json"), "w") as f:
        json.dump(loss_history, f, indent=2)
    with open(os.path.join(cfg["output_dir"], "run_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\nDone. Adapter: {adapter_dir}")
    print(f"Final train loss: {loss_history[-1]['loss'] if loss_history else 'n/a'}")
    print(f"Runtime (s): {stats.metrics.get('train_runtime', 'n/a')}")


if __name__ == "__main__":
    main()
