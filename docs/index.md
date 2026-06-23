---
layout: default
title: Teaching a Model to Reason — a zero-background fine-tuning curriculum
---

# Teaching a Model to Reason — a zero-background fine-tuning curriculum

A small language model learned to *think step-by-step* — to write out its
reasoning inside `<think> … </think>` before committing to an answer — trained
overnight on a single desktop. That desktop is an NVIDIA DGX Spark, the same
machine that teaches a humanoid robot to walk in the sister project
[`g1-humanoid-rl`](https://github.com/chaoticcuriosity-io/g1-humanoid-rl). The
two projects bookend modern model training: one teaches a *robot* to move, this
one teaches a *language model* to reason — and, by the final chapter, with the
**same family of reinforcement-learning methods**.

This is a curriculum that takes you through that whole arc from the beginning,
with **zero background in machine learning assumed**. Every term is defined the
first time it appears. Every "it works" claim is paired with a number or an
artifact. And the failures stay failures — the single most useful chapter in the
series is the one where the thing we tried *didn't* improve the model, because
catching that honestly is the whole skill.

## Start here

**[Chapter 00 — The Primer](reports/00-primer.md)** is the on-ramp: what
fine-tuning even is, how it differs from prompting and RAG, and what LoRA,
QLoRA, and a "reasoning model" actually mean. Read the chapters in order — each
one builds on the concepts the previous one introduced.

> 📖 New here? The fastest way in is to read the primer, then Chapter 01, and
> let the honest plot-twist at the end of Chapter 01 pull you through the rest.

---

## The four parts

### Part I — Foundations
*Chapter 00*

The vocabulary you need before anything else can land: base vs. instruct
models, what a "weight" is, what an **adapter** is, why **LoRA** trains ~1% of a
model and still works, what **quantization** and **QLoRA** buy you, and why this
particular desktop — 128 GB of *unified* memory — can do things a normal GPU
cannot.

Start: [Chapter 00 — The Primer](reports/00-primer.md)

---

### Part II — A first reasoner, honestly told
*Chapters 01–03*

The core experiment, and the lesson at the heart of the series. We teach a 3B
model to show its work — and discover the fine-tune made it *worse* at the math
it was meant to get better at. Then we ship it anyway (so you can chat with it),
then we run the controlled experiment that explains and mostly fixes the
regression.

| Chapter | Hook |
|---|---|
| [01 — Your First Reasoner](reports/01-first-reasoner.md) | A real LoRA fine-tune installs the `<think>` habit — and drops GSM8K accuracy from 74% to 49%. *Visible ≠ measured* is the lesson the whole series turns on. |
| [02 — From Adapter to Chat](reports/02-from-adapter-to-chat.md) | Merge the adapter → export GGUF → register with Ollama → chat with the model you just trained, in the open-webui already running on the Spark. |
| [03 — Turning the Knobs](reports/03-turning-the-knobs.md) | The controlled fix: matching the *difficulty* of the training data to the test recovers 19 of the 25 lost points (49% → 68%). Data difficulty is the biggest knob on the board. |

Start: [Chapter 01 — Your First Reasoner](reports/01-first-reasoner.md)

---

### Part III — The 128 GB showcase
*Chapter 04*

The same recipe, scaled to a model 24× larger — a **72-billion-parameter** model
fine-tuned on one desktop, something no consumer GPU can physically hold.

| Chapter | Hook |
|---|---|
| [04 — Scaling Up with QLoRA](reports/04-scaling-up-qlora.md) | A 72B model fits in 56 GB of unified memory and trains at ~22 s/step. The Spark's trade is laid bare: capacity, not speed. |

Start: [Chapter 04 — Scaling Up with QLoRA](reports/04-scaling-up-qlora.md)

---

### Part IV — Reinforcement learning
*Chapter 05 — with Chapter 06 in progress*

Supervised fine-tuning can only *imitate* the examples; to push a model **past**
its teacher you have to reward outcomes, not copy answers. That's reinforcement
learning — and it's the same idea, mathematically, that the sister project uses
to teach a robot to walk.

| Chapter | Hook |
|---|---|
| [05 — RL the Reasoner](reports/05-rl-the-reasoner.md) | GRPO (the method behind DeepSeek-R1) is PPO minus the value network. We demonstrate its prerequisites cheaply — and prove a tempting +4-point "win" was sampling noise. *Never trust a small-sample delta.* |
| 06 — RL That Actually Climbs *(in progress)* | The honest cliffhanger from Chapter 05: the real accuracy gain needs fast sampling, which needs **vLLM** — a from-source build on this aarch64/Blackwell box. This chapter documents that build, win or lose. |

Start: [Chapter 05 — RL the Reasoner](reports/05-rl-the-reasoner.md)

---

## The throughline

Read end to end, the series independently re-derives the real recipe for a
reasoning model — the one the frontier labs use:

1. **SFT to plant the behavior** — supervised fine-tuning with LoRA installs the
   `<think>` reasoning habit (Chapters 01–03).
2. **Scale it** — QLoRA puts the same recipe on a 72B model (Chapter 04).
3. **RL to sharpen it** — GRPO, on problems with *headroom*, rewards correct
   reasoning the model discovers on its own (Chapters 05–06).

Each experiment exists to show why one of those pieces is necessary — including,
crucially, the experiments that *failed* to improve anything and taught us why.

## How to actually run this

Every chapter ends with the exact commands and configs that produced its
figures. The operational side — connecting to the Spark, building the Blackwell
container, launching a run, and serving the result — lives in the
**[DGX Spark fine-tuning manual](dgx-spark-finetuning-manual.md)**. The training
scripts, per-experiment configs, and the append-as-we-go gotcha log all live in
the [repository](https://github.com/chaoticcuriosity-io/chaotic-fine-tuning).
Model weights, datasets, and checkpoints stay on the Spark — never in git.

---

*A [Chaotic Curiosity](https://chaoticcuriosity.io) project. Trained on an
NVIDIA DGX Spark with [Unsloth](https://unsloth.ai), via NVIDIA's
[DGX Spark Playbooks](https://github.com/NVIDIA/dgx-spark-playbooks). The scripts,
configs, and full run logs behind these chapters live in the
[repository](https://github.com/chaoticcuriosity-io/chaotic-fine-tuning); the
sister reinforcement-learning series is
[`g1-humanoid-rl`](https://github.com/chaoticcuriosity-io/g1-humanoid-rl).*
