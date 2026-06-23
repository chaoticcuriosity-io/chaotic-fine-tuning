# 02 · From adapter to chat: serving your model in open-webui

> We have a trained LoRA adapter sitting on the Spark. In this report we turn it into a single portable file,
> hand it to the Ollama server already running on the box, and **chat with our own reasoner** in the open-webui
> we already have open — right alongside Llama, Mistral, and Gemma.
>
> Prerequisite: [`01-first-reasoner.md`](01-first-reasoner.md) (that's where the adapter came from).

A trained adapter is not yet something you can *talk to*. Three steps stand between "training finished" and "a
model in my chat UI": **merge**, **convert**, **register**. We'll do each and explain why it exists.

---

## 1. Why we can't just "run the adapter"

Remember from [`00-primer.md`](00-primer.md) that our LoRA adapter is a small set of extra matrices
(`adapter_model.safetensors`, ~119 MB) that only make sense *bolted onto* the frozen base model. Three problems
for serving it:

1. **It's not a whole model.** The adapter alone is meaningless without the 3B base weights.
2. **It's in training format.** PyTorch/Safetensors files are for *training*, not for fast local *inference*.
3. **The chat server speaks a different dialect.** Ollama (and the `llama.cpp` engine underneath it) want a
   single self-contained file in the **GGUF** format.

So the pipeline is: **merge** the adapter into the base → **convert** the merged model to GGUF → **register**
that GGUF with Ollama.

---

## 2. Merge + convert to GGUF (one script)

[`scripts/export_gguf.py`](../../scripts/export_gguf.py) does both heavy steps via Unsloth:

```bash
ssh spark "docker exec unsloth-dev bash -lc 'cd /workspace && \
  python scripts/export_gguf.py \
    --adapter outputs/exp1-qwen3b-reasoning/adapter \
    --out     outputs/exp1-qwen3b-reasoning/gguf \
    --quant   q4_k_m \
    --name    chaotic-reasoner-3b'"
```

- **Merge:** Unsloth loads the base model, applies the adapter, and bakes the two together into one set of
  weights. The 0.96% we trained is now fused into the full model.
- **Convert + quantize to GGUF:** it then writes a `.gguf` file, quantized to **`q4_k_m`** — 4-bit weights with
  a medium-quality mix (see *quantization* in [`00-primer.md`](00-primer.md) §6). This shrinks our ~6 GB
  bf16 model to a **1.9 GB** file (`Qwen2.5-3B-Instruct.Q4_K_M.gguf`) that loads fast and runs comfortably.

> The first run compiles `llama.cpp` from source inside the container (a few minutes). That's expected — GGUF
> conversion is a `llama.cpp` tool, and on the Spark's ARM/Blackwell platform it builds on demand.

> **Gotcha we hit:** Unsloth writes the GGUF (and its own ready-made `Modelfile`) into a sibling directory with
> a `_gguf` suffix — so `--out outputs/.../gguf` produces the merged 16-bit weights in `gguf/` and the actual
> `.gguf` + `Modelfile` in **`gguf_gguf/`**. Use Unsloth's generated `Modelfile`; it already contains the
> correct Qwen chat template and stop tokens. (It defaults to `temperature 1.5`, which is too hot for reliable
> reasoning — we lower it to `0.7`.)

---

## 3. Register it with Ollama

The Ollama server runs in its own container (**`ollama-compose`** — mind the suffix). It can't see our
container's files, so we copy the GGUF + Modelfile in, lower the temperature, then `ollama create`:

```bash
SRC=~/chaotic-finetuning/outputs/exp1-qwen3b-reasoning/gguf_gguf
ssh spark "
  sed 's/temperature 1.5/temperature 0.7/' \$SRC/Modelfile > /tmp/cr3b_Modelfile
  docker exec ollama-compose mkdir -p /tmp/cr3b
  docker cp \$SRC/Qwen2.5-3B-Instruct.Q4_K_M.gguf ollama-compose:/tmp/cr3b/
  docker cp /tmp/cr3b_Modelfile ollama-compose:/tmp/cr3b/Modelfile
  docker exec ollama-compose bash -lc 'cd /tmp/cr3b && ollama create chaotic-reasoner-3b -f Modelfile'
  docker exec ollama-compose ollama list | grep chaotic"
```

It now appears in `ollama list`:

```
NAME                          ID              SIZE      MODIFIED
chaotic-reasoner-3b:latest    131aadf5f461    1.9 GB    Less than a second ago
```

---

## 4. Chat with it

**From the command line** (quick proof it's served and reasoning):

```bash
ssh spark "docker exec ollama-compose ollama run chaotic-reasoner-3b \
  'A robot walks 3 km in 20 minutes. At the same speed, how far does it walk in 50 minutes?'"
```

Real output from the served model (abridged — it reasons through Ollama exactly as it did in training):

> ```
> <think>
> ... to find distance you multiply speed by time. The robot walks 3 km in 20 minutes.
> 20 minutes = 20/60 = 1/3 hour, so speed = 3 km / (1/3 h) = 9 km/h.
> For 50 minutes = 50/60 h: distance = 9 × (50/60) = 450/60 = 7.5 km.
> Let me double-check ... GCD(450,60)=30 → 15/2 = 7.5 km. Yes.
> </think>
> The robot walks 7.5 km in 50 minutes.
> ```

It gets the right answer (7.5 km) with its reasoning on full display — and, true to form, it over-checks
(it even invented multiple-choice options that weren't asked for). That's our Experiment 1 model, faithfully.

**In open-webui** (the fun way): open the web UI on the Spark (it owns port 8080), click the model dropdown at
the top, and pick **`chaotic-reasoner-3b`**. It sits right next to llama3.1, mistral, and gemma3. Ask it a math
question and watch it think.

> Reality check, carried over from [report 01](01-first-reasoner.md): this model *reasons*, which is fun to
> watch, but it is **less accurate on grade-school math than the base model** (49% vs 74%) and tends to
> over-think. It's a great demo of "what fine-tuning changed," not a better calculator. The fixes are in
> reports [03](03-turning-the-knobs.md) and [05](05-rl-the-reasoner.md).

---

## 5. What you just learned

- A LoRA adapter must be **merged** into its base before it's a usable model.
- **GGUF** is the portable, quantized format local inference engines (Ollama/llama.cpp) use; **quantization**
  trades a little quality for a much smaller, faster file.
- **Ollama + open-webui** turn that file into a chat experience in two commands — and because they were already
  running on the Spark, your fine-tune joins a whole shelf of local models you can compare against.

---

## Reproduce

Full command sequence is in §2–§4 above. Script: [`export_gguf.py`](../../scripts/export_gguf.py).
Next: [`03-turning-the-knobs.md`](03-turning-the-knobs.md) — the hyperparameters that control all of this.
