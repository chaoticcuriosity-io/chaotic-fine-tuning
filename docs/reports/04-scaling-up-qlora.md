# 04 · Scaling up with QLoRA: fine-tuning a 72B model on a desktop

> Everything so far used a 3B model. Here we run the *same kind* of fine-tune on a **72-billion-parameter**
> model — roughly 24× larger — on the same single desktop. This is the thing the DGX Spark can do that a
> normal GPU simply cannot, and the technique that makes it possible is **QLoRA**.
>
> Prerequisites: [`00-primer.md`](00-primer.md) (QLoRA, quantization, unified memory) and
> [`03-turning-the-knobs.md`](03-turning-the-knobs.md) (the recipe we're scaling).

---

## 1. Why this is normally impossible — and why the Spark can

A 72B model in its native 16-bit form is about **144 GB** just for the weights — before you add the gradients
and optimizer state that *training* needs (recall the ~16 bytes/parameter math from [`00-primer.md`](00-primer.md)
§4, which would demand well over a *terabyte*). A typical gaming GPU has 24 GB of VRAM. The model doesn't just
fail to train — it won't even *load*.

Two ideas, stacked, change the picture:

1. **Unified memory.** The Spark's GB10 shares **128 GB** between CPU and GPU (see [`00-primer.md`](00-primer.md)
   §7). That's the *capacity* to hold a large model — the resource a discrete GPU runs out of first.
2. **QLoRA.** Load the frozen 72B base in **4-bit** (~40 GB instead of 144 GB), then train only a small LoRA
   adapter on top. The giant is compressed; the part that learns stays tiny.

Together: a 72B fine-tune that fits in the memory of a machine that sits on your desk.

---

## 2. What we ran

Same matched-data recipe as Experiment 2 (the one that worked), pointed at a bigger model:

| | Experiment 2 (Report 03) | Experiment 3 (this report) |
|---|---|---|
| Model | Qwen2.5-**3B**-Instruct | Qwen2.5-**72B**-Instruct (4-bit) |
| Loading | bf16 (~6 GB) | **QLoRA, 4-bit (~40 GB)** |
| Data | GSM8K grade-school reasoning | *(same)* |
| LoRA | r=16, α=16 | *(same)* |

We run a **short demonstration** (`max_steps: 30`) rather than a full training pass. The goal of this report is
to prove the *capability* and **measure the reality** — does a 72B model fit, and how fast does this
bandwidth-modest chip actually train it? — not to maximize accuracy (which would take many hours).

Config: [`configs/exp4.yaml`](../../configs/exp4.yaml).

---

## 3. Does it fit? (the headline)

Yes — with room to spare. Here is `free -h` *during* training, with the 72B model loaded and 30 LoRA steps
running:

```
              total        used        free      buff/cache   available
Mem:          121Gi        56Gi        4.7Gi       62Gi          65Gi
```

The whole job — 72-billion-parameter base in 4-bit, plus the LoRA adapter and its optimizer, plus activations
— sat in about **56 GB of the 121 GB pool, leaving ~65 GB available.** For perspective: a 24 GB gaming GPU
can't hold even *one-third* of this model's 4-bit weights. On the Spark it not only loaded, it trained.

| Metric | Value |
|---|---|
| Total / trainable params | 72.9 B / **210 M (0.29%)** |
| Memory used during training | **~56 GB** of 121 GB (≈65 GB still free) |
| Throughput | **~22 s/step** (4-bit, batch 1, seq 1024) |
| 30-step demo wall-clock | ~4 min to load + ~11 min to train |

(`nvidia-smi` still reports VRAM as `N/A` — there's no separate VRAM; `free -h` is the real gauge. See
[`00-primer.md`](00-primer.md) §7.)

---

## 4. The honest trade-off: capacity vs. speed

Fitting is only half the story. Here's the cost, laid bare:

| | 3B (Exp 2) | 72B (Exp 3) |
|---|---|---|
| Time per step | ~0.9 s | **~22 s** |
| Slowdown | 1× | **~24×** |

The per-step time grew ~24×, which is almost exactly the ~24× jump in parameter count (3B → 72.9B). That's not
a bug — it's physics. Every step has to stream the model's weights through the chip's memory bus, and the
GB10's unified memory, while *huge*, has modest **bandwidth** (~273 GB/s) compared to a datacenter GPU. More
weights to move per step → proportionally slower steps.

So a full training pass on 72B (hundreds to thousands of steps) is an **overnight** job here, not a coffee-break
one. For learning and experimentation — "can I fine-tune a 70B model at all, on hardware I own?" — that's a
fantastic trade. For "I need 10,000 steps by lunch," it isn't the right machine.

`★ The mental model:` the Spark is a *capacity* machine, not a *throughput* machine. It turns "impossible on my
hardware" into "possible, if you're patient" — which for experimentation and learning is exactly the trade you
want.

---

## 5. What you learned

- **QLoRA + unified memory** is what lets a desktop fine-tune models that a 24 GB GPU can't even load.
- The binding constraint flips: on a normal GPU it's *"does it fit?"*; on the Spark it's *"how long will it
  take?"*.
- **We fine-tuned a 72-billion-parameter model on a desktop**, in ~56 GB of unified memory — training only
  0.29% of it (210 M LoRA params), at the cost of ~24×-slower steps. Capacity bought with patience.

Next: [`05-rl-the-reasoner.md`](05-rl-the-reasoner.md) — the principled way to make a reasoner *better* (not
just bigger): reinforcement learning with GRPO, the cousin of the robot's PPO.

---

## Reproduce

```bash
# Pre-download the 4-bit 72B (≈40 GB) into the HF cache
ssh spark "docker exec -d unsloth-dev bash -lc 'HF_HUB_ENABLE_HF_TRANSFER=1 \
  huggingface-cli download unsloth/Qwen2.5-72B-Instruct-bnb-4bit'"

# QLoRA demo run (configs/exp4.yaml)
ssh spark "docker exec -d unsloth-dev bash -lc 'cd /workspace && \
  python -u scripts/train_lora.py --config configs/exp4.yaml > logs/exp4.log 2>&1'"
```
