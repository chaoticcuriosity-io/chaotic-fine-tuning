# DGX Spark Fine-Tuning — Operations Manual

Everything you need to connect to the Spark and run fine-tuning jobs. This is the operational companion to the
conceptual [reports series](reports/README.md). Written to be copy-pasteable.

> Sister document: the `g1-humanoid-rl` repo's `docs/dgx-spark-manual.md` covers the same machine for
> robotics RL. This manual covers the LLM fine-tuning side. They share the hardware and the SSH conventions.

---

## 1. The machine

- **Model:** NVIDIA DGX Spark — **GB10 Grace Blackwell** superchip.
- **Architecture:** `aarch64` (ARM64), **not** x86_64. This matters: some Python wheels have no ARM build and
  must be compiled from source (the reason we use a purpose-built container — see §3).
- **GPU compute capability:** `sm_121` (Blackwell). Needs CUDA 13.x toolchains.
- **Memory:** ~121 GB **unified** LPDDR5x, shared by CPU and GPU. There is no separate VRAM; `nvidia-smi`
  reports VRAM total as `N/A` — that's expected, not a fault.
- **Driver:** 580.x.
- **Disk:** ~3.7 TB on `/`, ~2.9 TB free (as of setup).

Verify any time:

```bash
ssh spark "uname -m; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; free -h | head -2"
```

---

## 2. Connecting

The Spark is reachable over **Tailscale** (a private mesh VPN). An SSH alias `spark` is already configured on
this Windows host.

```bash
ssh spark                          # alias → chaotic-curiosity@100.66.32.8 (Tailscale, key-based, no password)
```

Run a one-off command remotely:

```bash
ssh spark "<command>"
```

> **Note on the auto port-forward:** the SSH config forwards local port 8081 (used by the sister robotics
> project's Viser viewer). If another session already holds it you'll see a harmless
> `bind [127.0.0.1]:8081: Address already in use` warning. Ignore it — the command still runs.

### Co-tenant containers (what else is on the box)

```
mjlab-dev            # robotics RL (g1-humanoid-rl) — leave alone
open-webui           # chat UI, owns host port 8080 — KEEP UP (our chat loop)
ollama-compose       # Ollama model server — KEEP UP (note the name: ollama-compose, not "ollama")
compose-arangodb-1   # a database — can be stopped to free memory during big runs
```

`sudo` is **not** passwordless here. We cannot `swapoff` or `systemctl stop`. To free memory for a large run,
**stop containers** instead:

```bash
ssh spark "docker stop compose-arangodb-1"     # before a 70B run; restart with 'docker start' after
```

Keep `open-webui` and `ollama-compose` running — they're how we chat with the finished model.

---

## 3. The fine-tuning container

We use **Unsloth** inside a container built from NVIDIA + Unsloth's purpose-built
`Dockerfile_DGX_Spark`. It starts from `nvcr.io/nvidia/pytorch:25.11-py3` and compiles **triton** and
**xformers** from source for Blackwell (this is why a plain `pip install unsloth` fails on this machine).

### Build (one time, slow — compiles from source)

```bash
ssh spark "mkdir -p ~/chaotic-finetuning && cd ~/chaotic-finetuning && \
  wget -O Dockerfile_DGX_Spark 'https://raw.githubusercontent.com/unslothai/notebooks/main/Dockerfile_DGX_Spark' && \
  docker build -f Dockerfile_DGX_Spark -t unsloth-dgx-spark ."
```

For a long build, run it detached and watch the log:

```bash
ssh spark "cd ~/chaotic-finetuning && nohup docker build -f Dockerfile_DGX_Spark -t unsloth-dgx-spark . > build.log 2>&1 &"
ssh spark "tail -f ~/chaotic-finetuning/build.log"     # Ctrl-C to stop watching; build keeps going
```

### Launch the container (persistent, detached)

```bash
ssh spark "docker run -dit --gpus=all --net=host --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --name unsloth-dev \
  -v ~/chaotic-finetuning:/workspace \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -w /workspace \
  unsloth-dgx-spark sleep infinity"
```

- `-v ~/chaotic-finetuning:/workspace` — our code, configs, and outputs persist on the host.
- `-v ~/.cache/huggingface:/root/.cache/huggingface` — downloaded models/datasets are cached across runs.
- `--net=host --ipc=host` + the `ulimit`s — recommended by Unsloth for the Spark.

### Verify the environment

```bash
ssh spark "docker exec unsloth-dev bash -lc 'python -c \"import torch, unsloth; print(torch.__version__, torch.cuda.is_available())\"'"
```

Expected: a torch version string and `True`.

---

## 4. The run pattern

The same convention as the robotics project: wrap commands in `docker exec`.

```bash
# Interactive shell inside the container
ssh spark "docker exec -it unsloth-dev bash"

# One-off command
ssh spark "docker exec unsloth-dev bash -lc 'cd /workspace && python scripts/train_lora.py --config configs/exp1.yaml'"

# Long training run, detached, logging to a file
ssh spark "docker exec -d unsloth-dev bash -lc 'cd /workspace && python -u scripts/train_lora.py --config configs/exp1.yaml > logs/exp1.log 2>&1'"

# Watch a running training log
ssh spark "tail -f ~/chaotic-finetuning/logs/exp1.log"
```

### Stopping a run cleanly

Send the interrupt to the **python** process, not the wrapping shell (same gotcha as the robotics project —
the `bash -c '... > log'` wrapper doesn't always forward the first SIGINT):

```bash
ssh spark "docker exec unsloth-dev bash -lc 'kill -INT \$(pgrep -f train_lora.py)'"
```

---

## 5. Hugging Face access (for gated models)

Some models (e.g., Llama) are gated and need a token. Set it inside the container once:

```bash
ssh spark "docker exec unsloth-dev bash -lc 'export HF_TOKEN=*** && huggingface-cli whoami'"
```

Qwen models (our Experiment 1 default) are **not** gated, so the first run needs no token.

---

## 6. Serving the result through Ollama + open-webui

After a fine-tune we export a **GGUF** file and register it with the already-running Ollama server, which makes
it appear in open-webui automatically.

```bash
# The Ollama container is named ollama-compose:
ssh spark "docker exec ollama-compose ollama list"                       # see installed models
ssh spark "docker exec ollama-compose ollama create chaotic-reasoner-3b -f /path/Modelfile"
```

Details and the exact Modelfile are in report [`02-from-adapter-to-chat.md`](reports/02-from-adapter-to-chat.md).

---

## 7. Memory & throughput notes (the Spark's personality)

- **Capacity is the gift, bandwidth is the cost.** Big models *fit* (that's the point), but tokens/sec is
  modest vs a datacenter GPU. Expect a 3B LoRA run to finish in minutes; a 70B QLoRA run is hours.
- **Watch memory during big runs:** `ssh spark "watch -n5 free -h"` in a side terminal, or
  `nvidia-smi`. QLoRA on a 70B uses a large fraction of the 121 GB pool; stop `compose-arangodb-1` first.
- **No swap rescue.** We can't `swapoff`, and a swap "death spiral" freezes the box instead of erroring
  cleanly. Stay within memory: prefer QLoRA (4-bit) for anything ≥13B, and keep an eye on `free -h`.
