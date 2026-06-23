# Setup Notes

An append-as-we-go log of what was actually done, in order, with the deviations and gotchas that don't belong
in the polished reports. (Same spirit as `g1-humanoid-rl/setup-notes.md`.)

---

## Session 1 — 2026-06-21 — Connect, verify, build the Unsloth container

### Phase A — Connectivity & inventory ✅
Connected first try with `ssh spark`. Verified:
- User `chaotic-curiosity`, arch `aarch64`, GPU `NVIDIA GB10`, driver `580.126.09`.
- Memory: 121 GiB total, **110 GiB available** (comfyui not running → lots of headroom).
- Disk: 3.7 TB total, 2.9 TB free on `/`.
- Running containers: `mjlab-dev`, `open-webui`, `ollama-compose`, `compose-arangodb-1`.

**Gotcha — Ollama container name:** it's **`ollama-compose`**, not `ollama`. The `ollama create` / `ollama
list` commands must `docker exec ollama-compose ...`. Noted in the manual and CLAUDE.md.

**Gotcha — SSH auto port-forward:** the `spark` SSH alias forwards local 8081 (the robotics project's Viser
port). When another session holds it, every command prints
`bind [127.0.0.1]:8081: Address already in use` to stderr. Harmless; the remote command still runs.

### Phase B — Fetch & inspect the Unsloth Dockerfile ✅
`~/chaotic-finetuning/` created on the Spark. Fetched `Dockerfile_DGX_Spark` from `unslothai/notebooks`.
Key facts from the file:
- Base image: `nvcr.io/nvidia/pytorch:25.11-py3` — **already cached** on the box (19.5 GB), so no big pull.
- Compiles **triton** (pinned commit `c5d671f9`) and **xformers** (`v0.0.33`) **from source** with
  `TORCH_CUDA_ARCH_LIST="12.1"` for Blackwell. This is the reason a plain `pip install unsloth` doesn't work
  here — no prebuilt aarch64/sm_121 wheels.
- Then `pip install --no-deps bitsandbytes==0.48.0 transformers==4.56.2 trl==0.22.2` and `unsloth unsloth_zoo`.

### Phase C — Build the image 🚧
Launched detached: `nohup docker build -f Dockerfile_DGX_Spark -t unsloth-dgx-spark . > build.log 2>&1 &`.

**Gotcha — backgrounding over SSH:** the harness backgrounded the launching call, but the remote `docker build`
process (and BuildKit) keep running independently. Confirmed alive via
`pgrep -f 'docker build -f Dockerfile_DGX'`. Track progress by **build stage**, not log line count — the
triton/xformers compiles are silent for minutes at a time (`Building wheel for triton: still running...`).

### Phase C — Build the image ✅
Build finished cleanly. Final versions in the image: `unsloth 2026.6.8`, `unsloth_zoo 2026.6.6`,
`transformers 4.56.2`, `trl 0.22.2`, `peft 0.19.1`, `datasets 4.3.0`, `accelerate 1.14.0`,
`bitsandbytes 0.49.2` (the `pip install unsloth` pulled a newer bnb than the Dockerfile's pinned 0.48.0 —
harmless), `xformers 0.0.33`, `triton 3.4.0`, `torch 2.10.0a0 (nv25.11)`. Two harmless
`UndefinedVar` build warnings (`$C_INCLUDE_PATH`, `$CPLUS_INCLUDE_PATH`).

### Phase D — Launch + verify container ✅
`unsloth-dev` launched (`--gpus=all --net=host --ipc=host`, workspace + HF cache mounted). Verified:
`torch 2.10`, `cuda available: True`, device `NVIDIA GB10`, `import unsloth` OK (🦥). Unsloth reports
`Max memory: 121.69 GB`, `Bfloat16 = TRUE`, `Xformers = 0.0.33`, `FA2 = True`.

### Phase E — First smoke test caught a real bug 🐞 → fixed ✅
Ran `scripts/train_lora.py --config configs/smoke.yaml` (64 samples, 8 steps). Model + dataset downloaded
fine; dataset schema confirmed (`messages`, `num_tokens`, `source`). Then it failed at LoRA creation:

```
ImportError: Found an incompatible version of torchao. Found version 0.14.0+git,
but only versions above 0.16.0 are supported   (peft/import_utils.py is_torchao_available)
```

**Cause:** the NGC base ships `torchao 0.14.0+git`; `peft 0.19.1`'s `is_torchao_available()` *raises* on a
present-but-too-old torchao (instead of returning False), and PEFT only walks that path because torchao is
importable at all.
**Fix:** `pip uninstall -y torchao` inside the container — we don't use torchao (bitsandbytes does our
quantization), so with it absent PEFT skips the path cleanly. Baked the uninstall into
`Dockerfile_DGX_Spark` so future rebuilds are clean.

**Other note:** the math traces are long (first sample = 3117 tokens), so bumped `max_seq_length` 2048 → 4096
in the configs to avoid truncating reasoning mid-thought (the 3B model has plenty of memory headroom).

### Phase F — Smoke re-run passed ✅
After removing torchao, `configs/smoke.yaml` ran clean: LoRA attached (**29,933,568 / 3,115,872,256 =
0.96% trainable**), dataset tokenized via chat template, 8 steps, loss 0.905 → 0.762, adapter saved.
Throughput ~13 s/step at batch 8 / seq 4096. Pipeline validated.

### Phase G — Full Experiment 1 launched 🚧
`configs/exp1.yaml`: Qwen2.5-3B-Instruct, 2000 math reasoning traces, 1 epoch = **250 steps**, ~50 min ETA.
Detached to `logs/exp1.log`. First-step loss 0.85.

### Serving recon (for Task: GGUF → chat) ✅
- Ollama container `ollama-compose` (v0.12.6) already serves: llama3.1:8b, mistral, gemma3:4b, codellama:13b,
  nemotron-mini, llama3.2-vision:11b. Models live in volume `compose_ollama_data` → `/root/.ollama`.
- `open-webui` uses host networking (no published port shown) — reached at the Spark address :8080.
- Plan to register our model: `docker cp <gguf> ollama-compose:/tmp/ && ollama create chaotic-reasoner-3b`.
- GSM8K test set pre-cached (100 Qs); answer parsing verified (gold extracted after `####`).

### Phase H — Experiment 1 complete + the headline result 📊
Training: 250 steps, **57.5 min**, mean train loss **0.625** (from ~0.9). Loss curve →
`docs/reports/assets/exp1-loss.png`. Adapter (119 MB) at `outputs/exp1-qwen3b-reasoning/adapter`.

**Before/after on 100 held-out GSM8K questions (greedy, exact-match):**
- Base `Qwen2.5-3B-Instruct`: **74%**, never uses `<think>` (0/3 prompts).
- Tuned reasoner: **49%**, always uses `<think>` (3/3 prompts).

**The fine-tune installed the visible behavior (reasoning) but DROPPED accuracy 25 points.** Diagnosis from
transcripts: failures are genuine reasoning errors (not truncation) — the base was already strong + concise at
grade-school math, and imitating long *competition*-math traces made the small model verbose and error-prone
(e.g., gold 70000 → answered 1200). This is the central lesson of report `01-first-reasoner.md`: a behavior you
can *see* ≠ an outcome you can *measure*. Mirrors the g1 cartwheel "scorer said success, video said faceplant."

**Bug found + fixed:** `eval_before_after.py`'s `qualitative_think_rate` matched the literal `<think>` in its
own transcript template → always 1.0. Fixed to track the real per-output flag. (The true rates above come
from the correct per-prompt log lines: base 0/3, tuned 3/3.)

### Phase I — GGUF export → Ollama → open-webui ✅
`export_gguf.py` merged the adapter to 16-bit and converted to GGUF (`q4_k_m`). **Result: 1.9 GB**
(`Qwen2.5-3B-Instruct.Q4_K_M.gguf`).

**Gotcha — output path:** Unsloth writes the `.gguf` + its own `Modelfile` into a **`<out>_gguf/`** sibling dir
(the `<out>` dir itself got the merged 16-bit safetensors). Fixed `export_gguf.py` to search both and to reuse
Unsloth's Modelfile (correct Qwen template) with `temperature` lowered 1.5 → 0.7.

Registered with the `ollama-compose` server: `docker cp` the gguf + Modelfile in, `ollama create
chaotic-reasoner-3b`. Now in `ollama list` (1.9 GB) and selectable in **open-webui**. Served-model test
(`ollama run`) confirmed it reasons through Ollama: "3 km in 20 min → 9 km/h → 7.5 km in 50 min" ✓ (with the
expected over-checking). Reports `01-first-reasoner.md` and `02-from-adapter-to-chat.md` written with real
artifacts.

---

### Phase J — Experiment 2: the regression fix (Report 03) ✅
Controlled experiment — identical recipe, only the **data** changed (competition math → GSM8K grade-school,
`configs/exp3.yaml`, `dataset_format: gsm8k` wraps the CoT in `<think>`). Trained in **4 min** (seq 1024,
~1.1 it/s), final loss **0.189** (vs exp1's 0.625 — short regular reasoning is easy to fit; curve →
`exp2-loss.png`).

**GSM8K (same held-out 100): base 74% · exp1 (competition) 49% · exp2 (matched) 68%.** Matched data recovered
**19 of 25** lost points AND eliminated the rambling (transcripts now concise: "48/2=24 … 48+24=72 … **72**").
Did NOT beat the base 74% → lesson: fine-tuning has most upside adding a *new* skill, not competing with an
already-expert base; exceeding it needs RL (Report 05). Reports 01/02/03 complete.

**Minor bug:** `eval_before_after.py --out` defaulted to exp1's dir, so exp2 eval files landed in
`outputs/exp1-qwen3b-reasoning/`. Cosmetic; results intact.

### Phase K — Report 04: 72B QLoRA showcase ✅
Pre-downloaded `unsloth/Qwen2.5-72B-Instruct-bnb-4bit` (~40 GB, non-gated) into HF cache while exp2 ran.
QLoRA demo (`configs/exp4.yaml`, 30 steps, GSM8K). **It fit and trained on the desktop:**
- 72.9 B params, **210 M trainable (0.29%)**.
- **~56 GB of 121 GB used** during training (≈65 GB still free) — a 24 GB GPU can't hold even 1/3 of the
  4-bit weights.
- **~22 s/step** (vs ~0.9 s/step for the 3B) ≈ 24× slower, tracking the 24× param count. Capacity, not speed.
- 30 steps in ~11 min after ~4 min load. `nvidia-smi` mem = N/A (unified memory; use `free -h`).
(Note: can't stop the shared `compose-arangodb-1` to free memory — denied as shared infra — but 109 GB was
available anyway, so no need.) Report 04 written with real numbers.

### Phase L — Report 05: GRPO (RL the reasoner) ✅
`scripts/train_grpo.py` (trl GRPOTrainer, GSM8K correctness +2 / `<think>`-format +0.5 rewards, vLLM-free).
Smoke (3 steps, G=4) validated the API. Real demo: 40 steps, G=8, from **base** Qwen2.5-3B, ~21.5 s/step
(~14 min). Reward pinned at the ~2.0 correctness ceiling from step 2; **format reward = 0.0 all 40 steps**.

Two honest lessons (the finale): (1) **no headroom** — base already ~74% on GSM8K, so reward can't climb;
(2) **cold start** — base never samples `<think>`, so the format reward never fires and RL can't bootstrap it.
Together they re-derive the real recipe: **SFT to plant the behavior → RL (with headroom) to sharpen it** —
exactly why reports 01–03 precede 05, and the bridge to the G1 robot's PPO. Reward curve →
`exp5-grpo-reward.png`.

### Phase M — The "climbing curve" follow-up (Report 05 §5b) ✅ (honest negative)
Merged exp2 → standalone reasoner (`merge_adapter.py`), then GRPO **from it** on **unseen** GSM8K
(`--skip 2000`, 80 steps, G=8, ~16 min). **Cold-start FIXED:** format reward now fires **every step (0.5)** vs
0.0 from base — the SFT→RL handoff works. **But no correctness climb:** reward ricochets 0.5↔2.5 (1 prompt/step
= pure sampling noise; first-half 1.42 vs second-half 1.49). Held-out GSM8K: **68% → 69%** (within noise, flat).
Honest conclusion: RL's *prerequisites* are cheap to demo on a desktop; its *gains* need datacenter-scale steps
(many prompts/step, thousands of steps, vLLM). Curve → `exp6-grpo-climb.png`. Documented in Report 05 §5b.

### Phase N — Chasing the climb properly (user asked "do the climbing curve" / "run it on the spark") ✅ (confirmed flat)
Tried to force a *visible* climb by fixing both problems (headroom + cold-start):
- **Stale-CUDA hiccup:** after ~a dozen back-to-back GPU jobs, `torch.cuda.is_available()` flipped to False in
  the container. `docker restart unsloth-dev` fixed it (workspace bind-mounted, torchao fix in writable layer —
  nothing lost). Worth remembering.
- **exp1 (49%, max headroom) was intractable:** merged exp1 → GRPO from it, but its verbose ~614-tok
  completions × 32 samples/step → **ETA ~14.7 h** without vLLM. Killed it. (That ETA *is* the lesson.)
- **exp2 smoothed run:** GRPO from concise exp2, **4 prompts × 8 = 32 samples/step** (kills the noise), lr 1e-5,
  80 steps, ~1.3 h. Cold-start still fixed (format 0.5). Reward still **flat** (first-half 1.39 → 1.44). Curve →
  `exp7-grpo-smoothed.png`.
- **The measurement-noise capstone:** 100-Q eval teased **68% → 72% (+4!)** — looked like a win. Re-ran on
  **200 Q: 70% → 70%.** The +4 was sampling noise. **Confirmed: GRPO did not improve accuracy at this scale.**

Final, airtight Report 05: cold-start cheap & fixed; gains need real scale (headroom + thousands of steps +
vLLM); and *never trust a small-sample delta* — the same trap as the g1 cartwheel. The vLLM-from-source build
(the genuine speed unlock) was offered but not attempted (risk on aarch64/Blackwell).

## Status after Session 1
End-to-end pipeline proven: connect → containerize → SFT (LoRA/QLoRA) → evaluate → export → serve → scale to
72B → RL (GRPO). **All 6 reports (00–05) complete**, each with real runs + honest findings:
- 01: reasoning LoRA installed `<think>` but regressed GSM8K 74→49%.
- 03: matched-data fix recovered to 68% (data difficulty is the biggest knob).
- 04: QLoRA'd 72B in 56 GB of unified memory (~24× slower/step — capacity, not speed).
- 05: GRPO solved cold-start (format 0→0.5) but no accuracy gain at desktop scale (70→70 on 200 Q).
Possible future work: install vLLM → redo the high-headroom exp1 GRPO run (the genuine path to a real climb);
`git init` the repo (not yet under version control).
