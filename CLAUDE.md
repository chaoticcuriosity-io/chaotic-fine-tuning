# CLAUDE.md

> **Learning the course?** If the person wants to learn this material ("run me through this",
> "teach me", "start the course", "next lesson"), read **[AGENTS.md](AGENTS.md)** and follow it
> exactly. It is the tutor's operating manual, and it applies to Claude Code as written.
>
> Everything below this box is for maintainers running experiments on the author's own DGX
> Spark. A learner does not have that machine: never tell a learner to `ssh spark`, and never
> point them at the author's containers or paths.

Guidance for Claude Code (claude.ai/code) and other agents working in this repository.

## What this project is

Hands-on **LLM fine-tuning experiments** on an NVIDIA DGX Spark, with a from-scratch, no-jargon **report
series** as a co-equal deliverable. The throughline: teach a small model to *reason* (step-by-step `<think>`
chains) via LoRA, then scale the recipe to a 70B model with QLoRA. Sister project: `g1-humanoid-rl` (robotics
RL on the same Spark) — we mirror its documentation discipline.

## Two machines

1. **This checkout (Windows)** — where code, configs, and the `docs/` reports live and are committed.
2. **The DGX Spark (remote, over Tailscale)** — where all training actually runs. Reach it with `ssh spark`.
   Datasets, model weights, and checkpoints live there under `~/chaotic-finetuning/` and are **never**
   committed to git.

## Running things on the Spark

All GPU work happens inside the `unsloth-dev` container. The pattern (same as the robotics project):

```sh
ssh spark "docker exec unsloth-dev bash -lc 'cd /workspace && <command>'"
```

- Build/launch/verify instructions: see [`docs/dgx-spark-finetuning-manual.md`](docs/dgx-spark-finetuning-manual.md).
- The host dir `~/chaotic-finetuning/` is bind-mounted to `/workspace` inside the container.
- Detached long run: `docker exec -d unsloth-dev bash -lc '... > logs/run.log 2>&1'`.
- Stop a run: `kill -INT $(pgrep -f train_lora.py)` (signal the python child, not the shell wrapper).

## Co-tenants and memory (important)

Other containers share the box: `mjlab-dev` (robotics, leave alone), `open-webui` (host port 8080, **keep
up**), `ollama-compose` (**keep up** — note the `-compose` suffix), `compose-arangodb-1` (stoppable for memory).
`sudo` is **not** passwordless — manage memory by stopping containers, never by `swapoff`. The box has ~121 GB
**unified** memory (no separate VRAM; `nvidia-smi` shows VRAM `N/A` — expected).

## Serving a finished model

Export GGUF → register with the **`ollama-compose`** container → it appears in open-webui:

```sh
ssh spark "docker exec ollama-compose ollama create <name> -f /workspace/<dir>/Modelfile"
```

## Repo layout

```
docs/reports/      # the numbered, no-jargon learning series (00-primer … 05) + assets/
docs/dgx-spark-finetuning-manual.md   # ops manual (connect, build, run, serve)
scripts/           # train_lora.py, export_gguf.py, eval_before_after.py, plot_loss.py
configs/           # one YAML per experiment (model, dataset, r, alpha, lr, epochs)
setup-notes.md     # append-as-we-go log of what was actually done + gotchas
Dockerfile_DGX_Spark   # pinned copy of the Unsloth Spark image recipe
```

## Documentation principles (inherited from g1-humanoid-rl)

1. **No background assumed** — define every term on first use; link back to `00-primer.md`.
2. **Reproducible** — every report ends with the exact commands/configs that produced its figures.
3. **Honest about failure** — document bugs, dead ends, and surprises as lessons, not blemishes.
4. **Measure, don't vibe** — pair every "it works" claim with a number (e.g., GSM8K accuracy) or an artifact.
5. **Media-first proof** — embed loss curves, transcripts, and screenshots from `docs/reports/assets/`.

## Conventions

- Report files: numbered `NN-kebab-name.md` for the series; `kebab-case.md` for standalone topics.
- Keep large files (`*.gguf`, datasets, checkpoints) out of git — they live on the Spark.
- Prefer editing config YAML over hardcoding hyperparameters in scripts.
