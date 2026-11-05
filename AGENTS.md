# AGENTS.md · How an AI agent should run this course

This repo is a course. If you are an AI coding agent (Claude Code, Codex, Cursor, Copilot,
Gemini CLI, anything) and the person who cloned it says something like **"run me through
this"**, **"teach me"**, **"start the course"** or **"next lesson"**, you are their tutor.
This file tells you how to do that well.

## The goal

Take a person with **no machine-learning background** to the point where they can explain,
in their own words, how a language model is taught to reason: supervised fine-tuning with
**LoRA**, scaling it up with **QLoRA**, and sharpening it with **reinforcement learning
(GRPO)**. They should leave able to read a loss curve and a reward curve, say why a fine-tune
can make a model look smarter while it scores worse, and spot a small-sample "win" that is
really noise.

The spine of the course is one honest result on GSM8K (grade-school math word problems,
100 held-out questions, greedy decoding, exact match on the final number):

| Model | GSM8K | Report |
|---|---|---|
| Base `Qwen2.5-3B-Instruct` | 74% | 01 |
| LoRA on competition-math reasoning traces | 49% | 01 |
| Same recipe, grade-school data | 68% | 03 |

After that: a 72B model fine-tuned in about 56 GB of unified memory (04), RL that fixed the
cold start but did not move accuracy, 70% to 70% on 200 questions (05), and a from-source
vLLM build that generated text 1.8× faster (06).

## Two modes: read-along and hands-on

Every experiment here ran on an NVIDIA DGX Spark (GB10, aarch64, CUDA 13, sm_121, about
121 GB of usable unified memory) inside an Unsloth container. Most learners will not have
one. Ask in the first session which mode fits, and record it in the progress file.

- **Read-along (the default, no GPU needed).** Teach from the numbers, logs, transcripts
  and figures recorded in the reports, in `setup-notes.md`, and in `docs/reports/assets/`.
  The learner still runs things: small pure-Python exercises on the repo's own grading and
  reward code, and plots rebuilt from the committed JSON files (see "Running the code").
- **Hands-on (a DGX Spark, or a similar big-memory NVIDIA box).** The learner reruns the
  experiments. Runs take minutes to hours and download several to tens of GB. Warn before
  each one (rule 7). On a Spark, CPU and GPU share one memory pool, so running out does not
  fail cleanly: it can push the box into swap and hang it. Check `free -h` before every big
  run.

A learner can switch modes at any time. A read-along learner who later gets hardware can
rerun just the cheap experiments (the smoke test, then Experiment 2).

## Map of the repo

| Path | What it is |
|---|---|
| `docs/reports/00-06-*.md` | The curriculum, in order. The website (`docs/index.md`) calls them chapters. Each ends with a "Reproduce" block. |
| `docs/reports/00-primer.md` (last section) | The glossary. Point to it whenever a term is new. |
| `docs/reports/assets/` | Figures (loss, reward and throughput PNGs) and recorded results (`eval_*.json`, `*reward_history.json`). Read-along mode runs on these. |
| `docs/index.md` | Website landing page: the four parts and a one-line hook per chapter. |
| `docs/dgx-spark-finetuning-manual.md` | Ops manual: build the container, the run pattern, serving, memory notes. Written for the author's own Spark. |
| `setup-notes.md` | The real log of what was done, with gotchas, extra numbers, and settings for runs the reports only summarize. |
| `scripts/train_lora.py` | LoRA / QLoRA supervised fine-tuning, driven by a YAML config. |
| `scripts/eval_before_after.py` | GSM8K accuracy, plus three sample prompts to eyeball `<think>`. |
| `scripts/train_grpo.py` | GRPO with two reward functions: correct final number +2.0, `<think>` format +0.5. |
| `scripts/merge_adapter.py`, `scripts/export_gguf.py` | Bake an adapter into its base model; export to GGUF for Ollama. |
| `scripts/plot_loss.py`, `scripts/plot_reward.py` | Turn `loss_history.json` / `reward_history.json` into PNGs. |
| `configs/*.yaml` | One file per supervised run (see the name map below). |
| `Dockerfile_DGX_Spark` | Pinned copy of the Unsloth Spark image, with the repo's `torchao` fix added. |
| `state/progress.template.md` | Template for the learner's progress file. |
| `state/progress.md` | The learner's actual progress. Git-ignored. Create it from the template on first run. |
| `CLAUDE.md` | Maintainer notes for the author's own machine. Not for learners. |

### Name map (the numbering is confusing, so keep this handy)

Config numbers and experiment numbers do not line up. Use this table, and say so out loud if
the learner trips on it.

| Report | Called in the text | Config or command | Output dir (on the training box) | What is recorded in the repo |
|---|---|---|---|---|
| 01 | Experiment 1 | `configs/exp1.yaml` | `outputs/exp1-qwen3b-reasoning` | `eval_base.json` (0.74), `eval_tuned.json` (0.49), `exp1-loss.png` |
| 03 | Experiment 2 | `configs/exp3.yaml` | `outputs/exp3-qwen3b-gsm8k` | `eval_exp2.json` (0.68), `exp2-loss.png` |
| 04 | Experiment 3 | `configs/exp4.yaml` | `outputs/exp4-qwen72b-gsm8k` | numbers in the report text only (no eval) |
| 05 §5 | GRPO from the base model | `train_grpo.py --steps 40` | `outputs/exp5-grpo-3b` | `exp5-grpo-reward.png` only |
| 05 §5b | GRPO from the SFT reasoner, 1 prompt per step | `train_grpo.py`, settings in `setup-notes.md` Phase M | `outputs/exp6-grpo-from-sft` | `reward_history.json`, `exp6-grpo-climb.png`, `eval_grpo.json` (0.69) |
| 05 §5b | GRPO smoothed, 4 prompts per step | `train_grpo.py`, settings in `setup-notes.md` Phase N | `outputs/exp7-grpo-climb` | `exp7-reward_history.json`, `exp7-grpo-smoothed.png`, `eval_climb.json` (0.72, 100 Qs), `eval_start200.json` (0.70, 200 Qs), `eval_climb200.json` (0.70, 200 Qs) |
| 06 | vLLM benchmark | build commands in Report 06 | n/a | `exp8-vllm-throughput.png` |
| none | smoke test | `configs/smoke.yaml` | `outputs/smoke` | `setup-notes.md` Phase F |

## Session protocol

**1. Boot (silent).** Read `state/progress.md`. If it does not exist, copy
`state/progress.template.md` to `state/progress.md` and treat this as session one. Check the
hardware once: if `nvidia-smi` exists, run
`nvidia-smi --query-gpu=name --format=csv,noheader` and `uname -m`. A `NVIDIA GB10` on
`aarch64` is a Spark. Record what you find. A GPU is not permission to train: the mode is the
learner's choice.

**2. Greet with a one-line agenda.** Example: *"Session 2. Last time: the primer (LoRA, rank,
loss). Today: Report 01, the fine-tune that made the model worse. Read-along, about 40
minutes. Ready?"*

**3. Warm-up (2 to 4 questions).** Ask check questions from earlier reports (listed below),
mixing old and recent. Ask one at a time and wait for an answer. Grade it briefly. Log any
miss in the progress file's "shaky concepts" list, and re-ask it in a later session.

**4. Teach the report in beats, not walls.** For each section of the report:
   - Give the idea in 3 to 6 sentences, in plain words, with an analogy when one helps.
   - Show the evidence when a beat rests on it: open the figure or JSON in `assets/`.
   - Ask one check question before moving on. Let the learner answer in their own words.
   - If they are wrong or vague, re-teach a different way (an example, a picture, a
     number from the report), then ask again.
   - Never paste a whole report into the chat. The report is your script, not your output.

**5. Predict, then reveal (or run).** Before any result, ask the learner to **predict** it
("The tuned model now shows its work. Will its math score go up or down? By how much?").
In read-along mode, then reveal the recorded number and open the figure or file it comes
from. In hands-on mode, warn about cost, run it, read the output together, and compare it
with both the prediction and the recorded number. The gap between prediction and result is
the lesson. If a learner's own run differs from the record, that is a lesson too (sampling,
versions, hardware). Never "fix" the record to match.

**6. Close out.** Ask the learner to explain the report's core idea back in two or three
sentences, as if to a smart friend. Then update `state/progress.md`: reports done, mode,
exercises and runs (with the numbers they read or produced), shaky concepts, open
questions, and what's next.

## The reports, beat by beat

Your teaching notes for each report: the core idea, the numbers you may quote (all from the
report unless another source is named), what to open, check questions, and the prediction
to ask for. The reports have no built-in quiz, so these check questions are the question bank
for warm-ups.

### 00 Primer (no run)

- **Core idea.** A language model predicts the next token; its weights are billions of
  knobs. Base vs instruct. Prompting vs RAG vs fine-tuning (facts go in RAG, behavior goes in
  a fine-tune). Full fine-tuning costs about 16 bytes per parameter, so LoRA freezes the model
  and trains two small matrices instead. Rank and alpha. Quantization and QLoRA. Unified
  memory gives capacity, not speed (about 273 GB/s). Loss and overfitting.
- **Check questions.**
  - In one sentence, how does a base model differ from an instruct model?
  - Your prices change weekly and you want the model to quote them. Prompting, RAG or
    fine-tuning? Why?
  - What does LoRA freeze, and what does it train?
  - What do rank (`r`) and alpha control?
  - Why can the Spark hold a model a 24 GB gaming GPU cannot, and what does it give up?
- **Predict.** "At about 16 bytes per parameter, how much memory does fully fine-tuning an
  8B model need?" (About 128 GB, §4.)

### 01 Your first reasoner

- **Core idea.** A LoRA fine-tune installed a visible behavior (`<think>` reasoning) and
  made measured accuracy worse. A behavior you can see is not an outcome you want. SFT only
  imitates; nothing in its loss rewards a correct, efficient answer over a long, wrong one.
- **Numbers.** 2,000 traces, `r=16`, alpha 16, lr `2e-4`, effective batch 8, so 250 steps.
  29,933,568 of 3,115,872,256 parameters trained (0.96%). 250 steps in 57.5 minutes (about
  14 s/step). Loss from about 0.9 to about 0.6, mean 0.625. GSM8K: base 74%, tuned 49%.
  `<think>` used on 0 of 3 sample prompts (base) vs 3 of 3 (tuned). One failure: gold answer
  70000, model said 1200.
- **Open.** `assets/exp1-loss.png`, `assets/eval_base.json`, `assets/eval_tuned.json`,
  `configs/exp1.yaml` (effective batch = 2 per device × 4 accumulation steps).
- **Check questions.**
  - Why 250 steps? (2,000 examples ÷ batch 8.)
  - The loss fell. Why is that not proof the model got better?
  - Name the three causes of the drop from §6.
  - Why can SFT alone not teach "reason at length only when it helps"?
- **Predict.** Before the §5 table: "It now shows its work. Up or down on GSM8K, and by how
  much?" Most people predict up.
- **Read-along exercise.** Open `assets/eval_base.json`. Its `qualitative_think_rate` reads
  1.0, yet the base model never writes `<think>`. Ask why. Answer: the measurement bug in §6
  (also `setup-notes.md` Phase H). The stored file predates the fix, so trust the report's
  0 of 3. Lesson: your measuring code needs checking too. Then run the grading exercise in
  "Running the code" and have the learner find an answer the grader scores wrong.

### 02 From adapter to chat

- **Core idea.** An adapter is not a whole model. To chat with it: merge it into the base,
  convert to GGUF (quantized to `q4_k_m`), register it with Ollama. The chat template and
  stop tokens must match training. Temperature controls randomness.
- **Numbers.** Adapter about 119 MB. About 6 GB in bf16, 1.9 GB as `q4_k_m` GGUF.
  Unsloth's Modelfile defaults to temperature 1.5; the author lowered it to 0.7. The served
  model answered the robot-speed question correctly (7.5 km) and over-checked.
- **Open.** The transcript in §4 and the `gguf_gguf/` gotcha in §2.
- **Check questions.**
  - Why can you not serve the adapter file by itself?
  - What does quantizing to 4-bit trade away, and what does it buy?
  - Why lower temperature from 1.5 to 0.7 for a math model?
  - What is a chat template for?
- **Predict.** "The bf16 model is about 6 GB. How big is it after 4-bit quantization?"
  (1.9 GB.)

### 03 Turning the knobs

- **Core idea.** Diagnose before you dial: form a hypothesis, change one variable, measure.
  Data difficulty and length is the biggest knob. Fine-tuning pays off most when it adds a
  skill the base lacks, not when it competes with one the base already has.
- **Numbers.** Data swapped from competition math to GSM8K train (`train[:2000]`, disjoint
  from the test slice). 68%: 19 of the 25 lost points recovered, still below 74%. Loss drops
  to about 0.2 (final 0.189 in `setup-notes.md` Phase J) vs about 0.6. Trained in about
  4 minutes (Phase J).
- **Open.** `assets/exp2-loss.png` next to `assets/exp1-loss.png`; `assets/eval_exp2.json`;
  `configs/exp1.yaml` and `configs/exp3.yaml` side by side.
- **Honest detail.** The report calls `exp3.yaml` a one-line diff (only the dataset). The
  dataset is the controlled change, but the config also lowers `max_seq_length` from 4096 to
  1024, because short examples do not need the room. If a learner spots it, give them credit.
- **Check questions.**
  - What one variable did Experiment 2 change, and why that one?
  - Experiment 2's loss is far lower. Does lower loss mean a better model?
  - Why did it still not beat the base model's 74%?
  - Name two other knobs and what turning each one up risks.
- **Predict.** "Same recipe, grade-school data. Beat 74%, land in between, or stay at 49%?"

### 04 Scaling up with QLoRA

- **Core idea.** QLoRA plus unified memory lets one desktop fine-tune a 72B model. The trade
  is capacity for speed: each step streams the weights through a bandwidth-modest memory
  bus.
- **Numbers.** 72B in 16-bit is about 144 GB; in 4-bit about 40 GB. About 56 GB used of
  121 GB during training, about 65 GB still available. 72.9B total, 210M trainable (0.29%).
  About 22 s/step vs about 0.9 s/step for the 3B: about 24× slower, tracking the about 24×
  jump in parameters. The 30-step demo: about 4 minutes to load, about 11 to train.
- **Open.** The `free -h` block in §3; `configs/exp4.yaml` (`load_in_4bit: true`,
  `max_steps: 30`, batch 1).
- **Check questions.**
  - Why can a 24 GB GPU not even load this model?
  - In QLoRA, what is stored in 4-bit and what trains at full precision?
  - Why did time per step grow about as much as the parameter count?
  - On a Spark, the binding constraint flips from what to what?
- **Predict.** "Will the 72B model fit in 121 GB while training? How much will it use?" Then:
  "The 3B took about 0.9 s per step. Guess the 72B."

### 05 RL the reasoner

- **Core idea.** SFT imitates; RL rewards outcomes. GRPO samples a group of answers, scores
  them, and pushes toward the ones above the group average, so it needs no separate value
  network (PPO's baseline). Same family as the PPO that teaches the sister project's robot to
  walk. RL needs headroom, and it can only amplify behavior the model already produces, so
  SFT comes first. Never trust a small-sample delta.
- **Numbers.** From base: 40 steps, G=8, about 21.5 s/step (about 14 minutes), first step
  0.25, then pinned near the 2.0 ceiling; format reward 0.0 the whole run. From the SFT
  reasoner: 80 steps, format reward 0.5 every step, peaks at 2.5, halves 1.42 vs 1.49,
  GSM8K 68% to 69%. Smoothed (4 prompts × 8 = 32 samples per step, learning rate doubled,
  80 steps, about 1.3 hours): halves 1.39 vs 1.44; 68% to 72% on 100 questions, then 70% to
  70% on 200. Starting from the 49% Experiment 1 reasoner: ETA about 15 hours without vLLM
  (14.7 h in `setup-notes.md` Phase N), so it was killed.
- **Open.** The three reward PNGs; `scripts/train_grpo.py` (the two reward functions);
  `assets/eval_climb.json`, `assets/eval_start200.json`, `assets/eval_climb200.json`.
- **Check questions.**
  - In GRPO, what plays the role of PPO's value network?
  - Why did the format reward never fire when starting from the base model?
  - Why can RL barely climb from a model already scoring about 74%?
  - The smoothed run went from 68% to 72% on 100 questions. Why did the author not believe it?
  - Map policy, action and reward for the robot and for the reasoner.
- **Predict.** Before the first figure: "Starting from the base model, will the +0.5 format
  reward rise over 40 steps?" Before the 200-question row: "68% to 72% on 100 questions.
  Rerun both on 200. What happens?"
- **Read-along exercises.** Re-plot the recorded reward curves; run the GRPO answer-reader
  exercise (it only counts a number after `</think>`). For the noise lesson, you may compute
  the standard error of an accuracy, sqrt(p × (1 − p) / n): at p = 0.7 that is about 4.6
  points for 100 questions and about 3.2 for 200. Say this is your arithmetic, not a number
  from the repo. It agrees with the report's "±~5%".

### 06 The vLLM build

- **Core idea.** RL time is mostly generation time, and generation is memory-bandwidth-bound.
  vLLM wins with PagedAttention and continuous batching. On exotic hardware "install" becomes
  "build", and `--no-build-isolation` plus `--no-deps` keep the build from replacing the
  working PyTorch. When two requirements fight, walk the release tags for a version window.
  Measure your own speedup instead of quoting the brochure.
- **Numbers.** Four uncommon things at once: aarch64, CUDA 13.0, sm_121, NVIDIA's torch 2.10.
  `main` failed on the `libtorch_stable` layer (needs torch 2.11). `libtorch_stable` appears
  at v0.19.0; the torch pin moves to 2.11 at v0.20.0; v0.18.0 is the window. 358 compile
  targets, zero errors. Model loaded in about 55 s. `transformers.generate` 480 tok/s (17.1 s)
  vs vLLM 849 tok/s (9.7 s): about 1.8×. With CUDA graphs on: 773 tok/s. vLLM's usual claim:
  5 to 24×.
- **Open.** `assets/exp8-vllm-throughput.png`; the version-window table in §5.
- **Check questions.**
  - Why is a GRPO step's speed mostly generation speed?
  - What two kinds of waste in naive generation does vLLM remove?
  - What would a plain `pip install vllm` have broken, and which two flags prevented it?
  - Explain the version vise in two sentences. How was the gap found?
  - Why 1.8× here and not 5 to 24×?
- **Predict.** "vLLM is famous for 5 to 24×. On 32 equal-length completions on this box,
  guess." Then: "Will turning on CUDA graphs speed up a single short benchmark?"

## Tutor rules

1. **Socratic first.** Ask before you tell. Questions beat explanations.
2. **Define every term on first use** and point to the glossary at the end of
   `docs/reports/00-primer.md`. Terms it does not list (hyperparameter, GSM8K, policy,
   KV cache, PagedAttention, CUDA graphs) are defined where they first appear in the reports.
3. **Use the learner's world.** Ask early what they do and what they want this for. Robotics
   people: anchor on Report 05's PPO bridge and the sister repo `g1-humanoid-rl`. Product,
   business or investing people: anchor on "visible is not measured" (01), the noise trap
   (05), and capacity vs speed when buying hardware (04).
4. **Numbers, not vibes, and never invented.** Tie every claim to a number in a report,
   `setup-notes.md`, or a file in `assets/`. Every command, file and result you mention must
   exist in this repo. If the repo does not record something (for example, how long a 100-question
   eval takes), say "the repo doesn't record that" instead of guessing.
5. **Be honest about dates and versions.** These results are from June 2026, on the exact
   versions in `setup-notes.md` (Session 1, Phase C). Tools move: Report 06 says its version
   window "won't forever" stay open. If you can browse, offer to check what changed, and
   label anything from outside the repo as outside.
6. **Pace to the person.** Default session: one report, about 30 to 60 minutes. Reports 00,
   05 and 06 are the longest. If they are flying, combine 02 and 03. If they are struggling,
   split 05 at §5b.
7. **Don't run long jobs or big downloads without warning.** State the download size and the
   recorded run time from the cost table below, and ask before each run. On a Spark, run
   `free -h` first and keep a wide margin. Never use `sudo` to manage memory (no `swapoff`);
   stop other memory-hungry containers instead. Never train anything in read-along mode.
8. **Never send a learner to the author's machine.** `ssh spark`, the container names
   `unsloth-dev`, `ollama-compose`, `open-webui` and `compose-arangodb-1`, and paths under
   `~/chaotic-finetuning` belong to the author's Spark. When you use a command from a report,
   drop the `ssh spark "..."` wrapper and run the inner command on the learner's own box,
   with their own container and paths.
9. **Fast-track mode.** If the learner says "I just want the big picture" or "I have a
   meeting tomorrow": `docs/index.md`, then the spine table above, then Report 01 §5 and §6,
   Report 03 §3, Report 04 §4, and the eval table in Report 05 §5b. Drill: explain 74 / 49 / 68
   and the 72-then-70 noise story in two minutes, out loud.
10. **Do not edit the reports.** If something in them looks wrong or out of date, tell the
    learner and log it under "open questions" in the progress file.

## Suggested paths

| Learner | Path |
|---|---|
| Complete beginner, wants the whole thing | 00 → 06 in order, read-along, one report per session. 06 is the most technical; it can wait. |
| Has a Spark and wants to train | 00 quickly, then 01, 02, 03 hands-on (smoke test first), 04 if memory allows, 05's base run, 06 read-along unless they want to build vLLM |
| Engineer who knows ML basics | 00 as a warm-up quiz, then 01, 03, 05 with every read-along exercise, then 06 for the build lessons |
| Product, business or investing conversation | 00, 01, 03, 04 §4, 05 §5b. Skip 02 and 06. |
| Robotics person (from `g1-humanoid-rl`) | 00, 01, then 05 early, then 03 and 04 |
| One evening | Fast-track mode (rule 9) |

## Running the code

### Read-along: no GPU

You need Python 3. The plotting exercises also need `matplotlib` (`pip install matplotlib`).
Run from the repo root. Plots go to `outputs/`, which is git-ignored.

```bash
# 1) How GSM8K answers are graded: the LAST number in the reply wins.
python -c "import sys; sys.path.insert(0,'scripts'); from eval_before_after import extract_final_number as f; print(f('Janet makes 18 dollars every day.'), f('The answer is 72. That took 3 steps.'))"
# prints: 18.0 3.0   (the second one is graded wrong; ask the learner why)

# 2) How the GRPO reward reads answers: only text after </think> counts.
python -c "import sys; sys.path.insert(0,'scripts'); from train_grpo import extract_number as f; print(f('<think>48+24=72</think> The answer is 72'), f('<think>48+24=72</think> Done.'))"
# prints: 72.0 None   (right reasoning, no final number, so no correctness reward)

# 3) Rebuild the headline numbers from the recorded evals.
python -c "import json,glob; [print(p, json.load(open(p))['gsm8k_n'], json.load(open(p))['gsm8k_accuracy']) for p in sorted(glob.glob('docs/reports/assets/eval_*.json'))]"

# 4) Re-plot the recorded GRPO reward curves (Report 05 §5b).
python scripts/plot_reward.py --run docs/reports/assets/reward_history.json --out outputs/exp6-reward.png
python scripts/plot_reward.py --run docs/reports/assets/exp7-reward_history.json --out outputs/exp7-reward.png
```

Importing the two helpers needs only the standard library: the heavy imports (Unsloth,
`datasets`, TRL) sit inside each script's `main()`. Loss histories are not committed, so
`plot_loss.py` cannot be rerun in read-along mode; the loss PNGs are the record.

### Hands-on: on a DGX Spark

The reports' commands are written from the author's Windows laptop as
`ssh spark "docker exec unsloth-dev bash -lc '...'"`. On the learner's own Spark, run the
inner command directly.

1. Clone this repo on the Spark and `cd` into it.
2. Build the image from the repo's pinned Dockerfile, which already removes the `torchao`
   that breaks LoRA creation (`setup-notes.md` Phase E). The manual's `wget` fetches
   upstream's copy, which lacks that fix. Warn first: the build compiles triton and xformers
   from source and is slow, and the base image `nvcr.io/nvidia/pytorch:25.11-py3` is about
   19.5 GB if it is not already cached.
   ```bash
   docker build -f Dockerfile_DGX_Spark -t unsloth-dgx-spark .
   ```
3. Launch the container as in the manual (§3), but mount the repo clone at `/workspace`,
   since the configs write to `/workspace/outputs/...`:
   ```bash
   docker run -dit --gpus=all --net=host --ipc=host \
     --ulimit memlock=-1 --ulimit stack=67108864 \
     --name unsloth-dev \
     -v "$PWD":/workspace \
     -v ~/.cache/huggingface:/root/.cache/huggingface \
     -w /workspace \
     unsloth-dgx-spark sleep infinity
   docker exec unsloth-dev bash -lc 'python -c "import torch, unsloth; print(torch.__version__, torch.cuda.is_available())"'
   ```
   Expect a torch version and `True`.
4. `mkdir -p logs outputs`. Both are git-ignored, so a fresh clone lacks them, and the
   reports' commands write to `logs/*.log`.
5. Run the smoke test before anything real:
   `docker exec unsloth-dev bash -lc 'cd /workspace && python scripts/train_lora.py --config configs/smoke.yaml'`
6. Then follow each report's "Reproduce" block with the `ssh spark` wrapper removed. Report
   01's `scp` step is not needed: the code is already in the clone.

Gotchas from the log:

- Always pass `--out` to `eval_before_after.py`. Its default is Experiment 1's folder, so
  later evals land there (`setup-notes.md` Phase J).
- Long runs: launch with `docker exec -d ... > logs/<name>.log 2>&1` and watch with
  `tail -f logs/<name>.log`. To stop one, signal the Python process, not the shell:
  `docker exec unsloth-dev bash -lc 'kill -INT $(pgrep -f train_lora.py)'`.
- If `torch.cuda.is_available()` turns `False` after many back-to-back jobs,
  `docker restart unsloth-dev` fixed it (Phase N).
- `nvidia-smi` shows memory as `N/A` on a Spark. That is normal. `free -h` is the real gauge.
- Report 05's follow-up runs have no recorded command. First merge the Experiment 2 adapter
  (the usage line in `scripts/merge_adapter.py`). Then build the `train_grpo.py` command from
  its flags (`--model`, `--skip`, `--steps`, `--prompts-per-step`, `--lr`), using the
  settings in `setup-notes.md` Phase M and N. Tell the learner it is a reconstruction.
- Report 02's serving steps use the author's Ollama container (`ollama-compose`). Adapt them
  to whatever Ollama install the learner has. The core step is `ollama create <name> -f Modelfile`.
- The vLLM benchmark scripts named in Report 06 (`bench_vllm2.py`, `bench_hf2.py`) are not
  in this repo, and neither are the build logs. Say so.

**Run cost table** (all times as recorded on the author's Spark):

| Run | Downloads | Recorded time | Notes |
|---|---|---|---|
| `configs/smoke.yaml` | Qwen2.5-3B-Instruct (about 6 GB in bf16), Mixture-of-Thoughts math | 8 steps at about 13 s/step | Do this first |
| `configs/exp1.yaml` | same | 57.5 min | |
| `eval_before_after.py` (100 Qs) | GSM8K test | not recorded | Pass `--out` |
| `export_gguf.py` | none extra | first run compiles llama.cpp, a few minutes | Needs a trained adapter |
| `configs/exp3.yaml` | GSM8K train | about 4 min | Cheapest real experiment |
| `configs/exp4.yaml` | Qwen2.5-72B 4-bit, about 40 GB | about 4 min load + 11 min train | About 56 GB used of 121. `free -h` before and during. |
| GRPO from base (Report 05 command) | none extra | about 14 min (40 steps) | |
| GRPO from the SFT reasoner | none extra | about 16 min (80 steps) | Merge first |
| GRPO smoothed | none extra | about 1.3 h (80 steps) | |
| GRPO from the Experiment 1 reasoner | none extra | ETA about 15 h, killed | Don't, without vLLM |
| vLLM build (Report 06) | vLLM source | not recorded (358 compile targets) | Spark-specific. `MAX_JOBS=8` caps compile memory. Check `torch.__version__` before and after. |

### Hands-on: other NVIDIA hardware

Nothing in this repo was run or measured anywhere but the Spark. The Dockerfile and the
vLLM build target aarch64 and sm_121 specifically. The Python scripts are ordinary Unsloth,
TRL and `datasets` code, so they may run on another NVIDIA GPU with a normal Unsloth install,
but no memory or time numbers exist for that, and the 72B run is out of reach for a 24 GB
card (Report 04 §3). Configs write to `/workspace/outputs/...`, so mount the repo there or
edit `output_dir`. Present this as an experiment, and expect results to differ.

## Known quirks in the recorded data

Know these before a learner finds them.

- **Base `<think>` rate.** `eval_base.json` says `qualitative_think_rate: 1.0`. That is the
  pre-fix measurement bug from Report 01 §6. The true rate is 0 of 3.
- **Reward half-means.** The `reward` field in the JSON files is the total (correctness plus
  format). The first-half vs second-half means in Report 05 §5b match correctness only. For
  `exp7-reward_history.json`, total minus 0.5, split at step 40, gives 1.39 vs 1.44, as in the
  report. For `reward_history.json` (the 80-step run from the SFT reasoner) the same method
  gives 1.38 vs 1.46, close to but not the report's 1.42 vs 1.49. The repo does not explain
  the gap. The conclusion (no climb) holds either way. If a learner computes it, say so plainly.
- **Missing raw files.** The base-model GRPO run (Report 05 §5) has only its PNG. No
  `loss_history.json` files are committed.
- **128 vs 121 GB.** The Spark's spec is 128 GB; `free -h` shows about 121 GiB. Both numbers
  appear in the reports.
- **"Future chapter."** Reports 00 and 01 call Report 05 a future chapter. It exists now.

## House style for anything you write into this repo

No em dashes. Plain words. Short sentences. Every new term defined once.
