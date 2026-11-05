# chaotic-fine-tuning

Hands-on **LLM fine-tuning experiments** on an NVIDIA DGX Spark (GB10 Grace Blackwell, 128 GB unified memory),
paired with a from-scratch, **no-jargon learning series** that takes a complete beginner from "what is
fine-tuning?" to chatting with a model they trained themselves.

A *Chaotic Curiosity* project. Sister to [`g1-humanoid-rl`](https://github.com/chaoticcuriosity-io/g1-humanoid-rl) (reinforcement learning for a
humanoid robot on the same Spark) — and it shows: the two projects bookend modern model training. One teaches a
*robot* to move with RL; this one teaches a *language model* to reason, first with supervised fine-tuning and
later with the same family of RL methods.

## Learn it with an AI tutor

Clone this repo, open it in Claude Code, Codex, Cursor or any coding agent, and say **"run me through this"**.
The agent follows [`AGENTS.md`](AGENTS.md): one report per session, questions to check your understanding, and
progress saved in `state/progress.md`. **No GPU needed:** by default you follow along with the recorded numbers,
logs and figures. If you have a DGX Spark or a similar big-memory NVIDIA machine, the tutor can walk you through
rerunning the experiments.

## The throughline

We teach a small language model to **reason** — to think step-by-step inside `<think> … </think>` before it
answers — and we do it the way the frontier labs do:

1. **Supervised Fine-Tuning (SFT)** with **LoRA** on public reasoning traces (`open-r1/Mixture-of-Thoughts`).
2. **Scale up** the same recipe to a 70B model using **QLoRA** — something a normal GPU physically cannot do.
3. **Reinforcement learning (GRPO)** to push past the examples — a cousin of the **PPO** the robotics project
   uses — and a from-source **vLLM** build to make that RL affordable on this exotic Blackwell box.

## Start here

🌐 **[Read it as a website →](https://chaoticcuriosity-io.github.io/chaotic-fine-tuning/)** — the curriculum,
rendered and linked, on GitHub Pages.

📚 **[The Reports](docs/reports/README.md)** — the learning series. If you're new, read
**[`00-primer.md`](docs/reports/00-primer.md)** first; it defines every term the rest of the work uses.

🔧 **[Ops Manual](docs/dgx-spark-finetuning-manual.md)** — how to connect to the Spark, build the container,
run a job, and serve the result.

## What's here

| Path | What it is |
|---|---|
| [`docs/reports/`](docs/reports) | The numbered, no-jargon learning series (00 → 06) + figures |
| [`docs/dgx-spark-finetuning-manual.md`](docs/dgx-spark-finetuning-manual.md) | Connect / build / run / serve, copy-pasteable |
| `scripts/` | `train_lora.py`, `export_gguf.py`, `eval_before_after.py`, `plot_loss.py` |
| `configs/` | One YAML per experiment (model, dataset, rank, alpha, learning rate, epochs) |
| [`setup-notes.md`](setup-notes.md) | Running log of what was actually done, with the gotchas |
| `Dockerfile_DGX_Spark` | Pinned copy of the Unsloth Spark image recipe |

## The stack

- **Hardware:** NVIDIA DGX Spark (GB10, aarch64, CUDA 13, sm_121, 121 GB unified memory).
- **Framework:** [Unsloth](https://unsloth.ai) (LoRA/QLoRA), via NVIDIA's
  [DGX Spark Playbooks](https://github.com/NVIDIA/dgx-spark-playbooks).
- **Serving:** Ollama + open-webui (already running on the Spark) — so you can chat with what you train.
- **Data:** [`open-r1/Mixture-of-Thoughts`](https://huggingface.co/datasets/open-r1/Mixture-of-Thoughts).

## Status

**Sessions 1–2 complete** — the full pipeline end to end, plus a 7-part report series (00 → 06):

- ✅ Connected to the Spark, built the Blackwell Unsloth container, ran a LoRA fine-tune, evaluated, exported
  GGUF, and served it in open-webui.
- ✅ **The arc, honestly told:** a reasoning fine-tune *installed* `<think>` behavior but *regressed* GSM8K
  accuracy (74%→49%, Report 01); matching the data difficulty recovered most of it (→68%, Report 03); a 72B
  model trained in 56 GB of unified memory (Report 04); and GRPO/RL closes the loop to the G1 robot's PPO
  (Report 05).
- ✅ **Built vLLM from source** for the Spark's aarch64 / CUDA-13 / sm_121 / NVIDIA-torch-2.10 stack — threading
  a torch-version vise via the v0.18.0 release window — and measured a 1.8× generation speedup over
  `transformers.generate` (Report 06).

See [`setup-notes.md`](setup-notes.md) for the blow-by-blow log and [`docs/reports/`](docs/reports) for the
series. Models, datasets, and checkpoints live on the Spark, not in git.
