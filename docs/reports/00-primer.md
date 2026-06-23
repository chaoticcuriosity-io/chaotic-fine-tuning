# 00 · Primer: What is fine-tuning, and what is LoRA?

> The first report in a series that takes a reader with **no machine-learning background** from "what is
> this?" to training, measuring, and chatting with their own fine-tuned language model on an NVIDIA DGX Spark.
> Every term is defined the first time it appears. Each report is standalone and re-runnable from this repo
> plus the Spark.

If you read only one report, read this one. It builds the vocabulary the rest of the series leans on. Nothing
here requires you to type a single command — we are just getting the ideas straight.

---

## 1. What is a language model, really?

A **language model** (LM) is a program that, given some text, predicts what text is likely to come next. That
is the whole trick. You show it `"The capital of France is"` and it assigns a high probability to `" Paris"`.
A **large** language model (LLM) is the same idea with billions of internal numbers (we call them **weights**
or **parameters**) and trained on a large fraction of the public internet.

Think of the weights as the **billions of tiny knobs** inside the model. Every knob is a number. The specific
settings of all those knobs *are* the model's knowledge and skills. "Training" means turning the knobs;
"using" the model means leaving the knobs fixed and feeding text through them.

When you chat with a model, here is the loop, roughly 50 times a paragraph:

1. The model reads everything so far (your message + what it has written).
2. It produces a probability for *every possible next word-piece* (these pieces are called **tokens** — a
   token is usually a word or a chunk of one, like `"Par"` + `"is"`).
3. It picks one token, appends it, and repeats from step 1.

That is it. There is no "understanding" module bolted on the side. Fluent answers, working code, and
step-by-step reasoning are all *emergent* from "predict the next token" done extremely well.

---

## 2. Three stages: pretraining → instruction-tuning → your fine-tune

Models arrive in your hands at different stages of "cooked." Knowing which stage you have matters.

**Pretraining** is the giant, expensive first phase: read trillions of tokens of raw internet text, nudging
the knobs so the model gets good at next-token prediction in general. This costs millions of dollars and
months of compute. The result is a **base model** (e.g., `Llama-3.1-8B`). A base model is knowledgeable but
*feral* — ask it a question and it might continue with *more questions*, because internet text full of
questions is often followed by more questions, not answers.

**Instruction-tuning** is a second, much cheaper phase where the lab shows the base model many
`(instruction, good response)` examples until it learns the *behavior* of "when a human asks, I answer
helpfully." The result is an **instruct** (or **chat**) model, e.g., `Llama-3.1-8B-Instruct`. This is the
kind you talk to in ChatGPT-like apps.

> **Base vs. instruct, in one line:** the base model *knows things*; the instruct model *knows how to behave*
> in a conversation. They have nearly identical knowledge; they differ in manners.

**Fine-tuning** — what this whole series is about — is *you* running a small third phase: take an existing
model (base or instruct) and show it examples of a narrower skill, style, or knowledge you care about, nudging
a few knobs so it leans your way. You are not creating a model from scratch; you are **specializing one that
already exists**. That is why fine-tuning is affordable on a desktop while pretraining is not.

---

## 3. Fine-tuning vs. prompting vs. RAG — when do you even need it?

Newcomers often reach for fine-tuning when a cheaper tool would do. Three ways to change a model's output,
cheapest first:

| Technique | What it does | Analogy | Changes the weights? |
|---|---|---|---|
| **Prompting** | Put instructions/examples in the message | Giving someone a sticky note | No |
| **RAG** (Retrieval-Augmented Generation) | Look up relevant documents and paste them into the prompt | Letting someone open the textbook during the exam | No |
| **Fine-tuning** | Adjust the model's own weights with training examples | Making them *study* until it's second nature | **Yes** |

Rules of thumb:

- Need the model to **know a fact** (your API docs, today's prices)? → Reach for **RAG**. Facts change; you
  don't want to retrain every time.
- Need the model to **behave a certain way** every time (a consistent format, tone, or skill like step-by-step
  reasoning)? → Reach for **fine-tuning**. Behavior is stable and hard to fit in a prompt every single call.
- Can a better prompt solve it? → Always try that first; it's free.

We fine-tune in this series because we want a *behavior* — reasoning — baked in, not pasted in each time.

---

## 4. The core problem fine-tuning has to solve

A model like Llama-3.1-8B has **8 billion** weights. Llama-3.1-70B has **70 billion**. To "fully" fine-tune a
model the classic way, you must:

- Hold all the weights in memory, **and**
- Hold a similar-sized blob of "how should I nudge each weight" numbers (called **gradients**), **and**
- Hold the optimizer's bookkeeping (often **2× the weights again**).

The rough cost of *full* fine-tuning is **~16 bytes per parameter**. For an 8B model that's ~128 GB; for 70B
it's well over a terabyte. That is why, for years, fine-tuning "real" models needed a cluster. We need a
cleverer idea to do it on one desktop. That idea is LoRA.

---

## 5. LoRA: the idea that makes desktop fine-tuning possible

**LoRA** stands for **Low-Rank Adaptation**. Here is the insight, in plain language.

When you specialize a model for a narrow task, the *change* you need to make to its giant weight matrices turns
out to be **simple** — even though the matrices themselves are huge. "Simple" has a precise mathematical
meaning here: the change is **low-rank**, meaning it can be reconstructed from a couple of skinny matrices
multiplied together, instead of one enormous one.

An analogy. Imagine a giant spreadsheet, 1000 rows × 1000 columns — a million cells. That's a weight matrix.
Now suppose the *adjustment* you want to make to it is "add 3% to everyone in the sales region." You do **not**
need a second million-cell spreadsheet to express that. You can store it as a short "row factor" list and a
short "column factor" list and multiply them on the fly. Two lists of 1000 numbers (2000 total) reconstruct a
correction to all 1,000,000 cells. That compression — many small corrections captured by a few skinny
factors — is *low rank*.

LoRA does exactly this to a model:

1. **Freeze** every original weight. We never touch the billions of pretrained knobs.
2. Next to each big weight matrix, **inject two small matrices**, called **A** and **B**. Together they're the
   **adapter**.
3. During training, only **A** and **B** learn. The model's output becomes `original (frozen) + A×B (learned)`.

Because A and B are tiny, you train **well under 1%** of the parameters. The memory and compute for the
"nudge" math collapse accordingly. You get most of the benefit of full fine-tuning at a tiny fraction of the
cost, and the result — the adapter — is a small file (often just tens of megabytes) you can keep, share, or
swap.

### LoRA's two main knobs: rank and alpha

- **Rank (`r`)** — how skinny the A and B matrices are; i.e., how much "capacity" the adapter has. Small `r`
  (8, 16) = a light touch, fewer parameters, less risk of memorizing. Larger `r` (64, 128) = more capacity to
  learn complex changes, but more memory and more risk of **overfitting** (see §8). Think of `r` as *how many
  knobs you're allowed to add*.
- **Alpha (`lora_alpha`)** — a scaling factor on how loudly the adapter speaks relative to the frozen model.
  A common starting point is `alpha = r` or `alpha = 2×r`. Think of it as the adapter's *volume dial*.

We will literally turn these knobs and watch what happens in report `03-turning-the-knobs.md`. For now, just
know they exist and what they mean.

---

## 6. Quantization and QLoRA: fitting big models in a small space

One more idea unlocks the *really* big models. Each weight is normally stored as a 16-bit number (2 bytes).
**Quantization** means storing each weight more coarsely — for example in **4 bits** (half a byte) — like
saving a photo as a smaller JPEG. You lose a little fidelity; you save 4× the space.

**QLoRA** = **Quantized LoRA**: load the frozen base model in 4-bit (so even a 70B model fits in memory), then
attach the small LoRA adapter on top and train *that* in full precision. The frozen giant is compressed; the
tiny part you're actually learning stays sharp.

> QLoRA is the reason a single DGX Spark can fine-tune a **70B** (or even a ~120B) model. The base model is
> quantized down to fit the 128 GB unified memory; the adapter does the learning. We do exactly this in
> report `04-scaling-up-qlora.md`.

---

## 7. Why the DGX Spark is unusually good at this

A normal gaming GPU has, say, 24 GB of dedicated video memory (**VRAM**) — a hard wall. The model either fits
or it doesn't.

The DGX Spark (NVIDIA **GB10 Grace Blackwell**) has **128 GB of *unified* memory**: one big pool of LPDDR5x
RAM that **both** the CPU and the GPU read from directly. There is no separate "VRAM" to overflow — when you
run `nvidia-smi` it even reports VRAM as `N/A`, which surprises people. The trade-off is **bandwidth**: that
unified memory is slower to stream than a datacenter GPU's dedicated memory (~273 GB/s here). So the Spark's
superpower is **capacity, not raw speed**: it happily *holds* a 70B model that a 24 GB card can't load at all,
even if it crunches through tokens more slowly. For fine-tuning — where "does it fit?" is usually the binding
constraint — that's a fantastic trade.

---

## 8. The one failure mode to know up front: overfitting

**Overfitting** is when a model *memorizes* your training examples instead of *learning the general pattern*.
A student who memorizes the answer key aces the practice test and bombs the real exam. Signs: training loss
keeps dropping but the model gets worse at anything outside the training set; it parrots training examples
verbatim. The cures are all about restraint — fewer training passes (**epochs**), a smaller adapter (lower
`r`), more diverse data. We will deliberately *cause* overfitting in `03-turning-the-knobs.md` so you can
recognize its fingerprint.

A companion term: **loss**. During training the model is graded on how wrong its next-token predictions are on
your examples; that error number is the **loss**. Training tries to push loss down. A **loss curve** that
slopes down means the model is learning the data. (Down is good — but, as the cartwheel saga in the sister
`g1-humanoid-rl` project taught us, a good training number is *not* proof of a good model. Always check the
actual behavior too. We do, in §10's spirit, with a real held-out test.)

---

## 9. The fine-tuning toolbox (and which tool we picked)

You don't implement LoRA by hand — you use a framework. NVIDIA publishes official, GB10-tuned **DGX Spark
Playbooks** (<https://github.com/NVIDIA/dgx-spark-playbooks>) for several:

| Tool | Best for | Why you'd pick it |
|---|---|---|
| **Unsloth** | Fast, memory-light LoRA/QLoRA, easy export | 2× faster, lowest memory, one-line GGUF/Ollama export. Our choice. |
| **LLaMA Factory** | Menu-driven experiments | GUI/CLI over many models; SFT + preference tuning + QLoRA. |
| **PyTorch + PEFT** | Maximum control / research | You write the loop; see every gear turn. |
| **NeMo AutoModel** | Production / multi-node | Native FP8 on Blackwell, scales past one machine. |

We use **Unsloth** because it is the smoothest path to *learning the concepts*: it's the NVIDIA-blessed option
for the Spark (there's a purpose-built Docker image that compiles the tricky Blackwell dependencies for us),
it's fast enough that a first run finishes in minutes, and it exports straight to a format Ollama can serve —
which means we get to **chat with our own model** at the end, in the `open-webui` already running on the box.

---

## 10. What we're actually going to build

Our throughline is a **reasoning model**. Today's frontier models (DeepSeek-R1 and friends) don't just answer
— they first *think out loud* inside `<think> … </think>` tags, working through a problem step by step before
committing to an answer. That visible reasoning measurably improves their accuracy on hard problems.

That reasoning behavior is **taught**, in two phases:

1. **SFT (Supervised Fine-Tuning)** — show the model thousands of examples of good `question → <think>
   reasoning → answer` traces until it adopts the habit. *This is the LoRA fine-tune we'll do.* (Our data:
   `open-r1/Mixture-of-Thoughts`, a public dataset of reasoning traces distilled from R1.)
2. **RL (Reinforcement Learning), via a method called GRPO** — then *reward* the model for reasoning that
   leads to correct answers, letting it improve beyond the examples. This is a cousin of the **PPO** algorithm
   the sister `g1-humanoid-rl` project uses to teach a humanoid robot to walk: in both cases a reward signal
   shapes a policy. We flag this as the series' future chapter, `05-rl-the-reasoner.md`.

So the arc of this series:

- **`01-first-reasoner.md`** — teach a small 3B model to reason with a LoRA fine-tune; watch the loss fall;
  see the before/after; and *measure* the gain on held-out grade-school math (GSM8K).
- **`02-from-adapter-to-chat.md`** — turn the adapter into a servable model and chat with it in open-webui.
- **`03-turning-the-knobs.md`** — change rank, alpha, learning rate, epochs; cause and diagnose overfitting.
- **`04-scaling-up-qlora.md`** — repeat the whole thing on a 70B model via QLoRA — the Spark's headline trick.
- **`05-rl-the-reasoner.md`** *(future)* — GRPO, and how it mirrors the robot's PPO.

---

## Glossary (everything defined above, in one place)

- **Token** — a word or word-piece; the unit a model reads and writes.
- **Weight / parameter** — one of the billions of tunable numbers inside a model.
- **Pretraining** — the giant first training phase that produces a **base model**.
- **Instruct / chat model** — a base model further trained to answer helpfully.
- **Fine-tuning** — a small extra training phase you run to specialize a model.
- **RAG** — pasting retrieved documents into the prompt so the model can use facts it wasn't trained on.
- **LoRA** — Low-Rank Adaptation: freeze the model, train two small **adapter** matrices instead.
- **Rank (`r`) / alpha** — the adapter's capacity / volume knobs.
- **Quantization** — storing weights in fewer bits (e.g., 4-bit) to save memory.
- **QLoRA** — LoRA on top of a 4-bit-quantized frozen base; how big models fit on one machine.
- **Unified memory** — one RAM pool shared by CPU and GPU; the Spark's defining feature.
- **Loss** — the training error number; a falling **loss curve** means learning.
- **Overfitting** — memorizing the training set instead of generalizing.
- **Epoch** — one full pass over the training data.
- **Batch / batch size** — the number of training examples processed together in one step. Smaller batches = noisier loss curve; larger batches = smoother updates but slower.
- **Step** — one round of the training loop: feed a batch of examples through the model, measure the loss, compute gradients, update the weights. With *N* examples and batch size *B*, one epoch = *N ÷ B* steps (e.g. 2,000 examples ÷ batch 8 = 250 steps).
- **Gradient** — the calculated direction and size of each weight adjustment. Computed from the loss; tells the optimizer which way to nudge each knob.
- **Optimizer** — the algorithm that reads the gradients and applies them to the weights. AdamW is the default; its bookkeeping is the "2× the weights" memory cost in §4.
- **Learning rate** — how large a step the optimizer takes per update. Too high: learning is unstable and the model forgets existing skills. Too low: barely learns. `2e-4` is a well-worn LoRA starting point.
- **Inference** — running a trained model to produce output, as opposed to *training* it. Inference doesn't need gradients, so it uses far less memory.
- **Greedy decoding** — a generation strategy that always picks the single most-likely next token. Deterministic and reproducible, which is why it's used for evaluation.
- **Temperature** — a dial on generation randomness. Near 0: always picks the top token (like greedy). Higher (e.g. 1.5): more varied, sometimes creative, sometimes unreliable. Usually lowered for tasks where correctness matters.
- **bf16** — "brain float 16," a 16-bit number format for storing weights. Half the bytes of standard 32-bit floats, with enough precision for training. The default format on modern hardware like the Spark.
- **SFT** — Supervised Fine-Tuning: learning from example responses (what we do).
- **GRPO / PPO** — reinforcement-learning methods that shape behavior with a reward signal.
- **GGUF** — the compact file format Ollama/llama.cpp use to run a model locally (you'll meet it in report 02).

---

*Next: [`01-first-reasoner.md`](01-first-reasoner.md) — we stop talking and train one.*
