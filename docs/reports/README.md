# The Chaotic Fine-Tuning Reports

A short series that takes a reader with **no machine-learning background** from "what is fine-tuning?" to
training, measuring, and chatting with their own reasoning model on an NVIDIA DGX Spark. Every term is defined
on first use. Each report is standalone and re-runnable from this repo plus the Spark.

This is the sister series to the [`g1-humanoid-rl`](https://github.com/chaoticcuriosity-io/g1-humanoid-rl) reinforcement-learning reports, and
it follows the same rules: define the jargon, show the data, embed the proof, and be honest when something
fails.

## Reading order

| # | Report | What you'll learn | Status |
|---|---|---|---|
| 00 | [Primer](00-primer.md) | Language models, fine-tuning vs RAG vs prompting, **LoRA**, **QLoRA**, why the Spark is special | ✅ |
| 01 | [Your first reasoner](01-first-reasoner.md) | A real LoRA fine-tune; teach a 3B model to *show its work* — and discover it got *worse* (74%→49% on GSM8K) | ✅ |
| 02 | [From adapter to chat](02-from-adapter-to-chat.md) | Merge → GGUF → Ollama → **chat with your own model in open-webui** | ✅ |
| 03 | [Turning the knobs](03-turning-the-knobs.md) | The controlled fix: matched data recovers 19 points (49%→68%) — data difficulty is the biggest knob | ✅ |
| 04 | [Scaling up with QLoRA](04-scaling-up-qlora.md) | Fine-tune a **72B** model on one desktop — fit in 56 GB, ~24× slower/step (capacity vs speed) | ✅ |
| 05 | [RL the reasoner](05-rl-the-reasoner.md) | GRPO (PPO's cousin); why RL needs headroom + an SFT cold-start — the bridge to the robot's PPO | ✅ |
| 06 | RL that actually climbs | The from-source **vLLM** build on aarch64/Blackwell that makes a real RL accuracy gain affordable — documented win or lose | 🚧 |

## The throughline

We teach a small language model to **reason** — to think step-by-step inside `<think> … </think>` before
answering — using **Supervised Fine-Tuning (SFT)** with **LoRA** on a public dataset of reasoning traces
(`open-r1/Mixture-of-Thoughts`). Then we scale the same recipe to a 70B model, and (in the future chapter)
polish it with reinforcement learning.

## Conventions

- **Hardware:** all training runs on the DGX Spark (GB10, 128 GB unified memory). See
  [`../dgx-spark-finetuning-manual.md`](../dgx-spark-finetuning-manual.md) for how to connect and run.
- **Artifacts:** small plots, screenshots, and transcripts live in [`assets/`](assets) and are embedded
  directly in the reports. Large model files and datasets stay on the Spark, never in git.
- **Reproducibility:** every report ends with the exact commands and configs used to produce its figures.
