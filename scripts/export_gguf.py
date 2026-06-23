#!/usr/bin/env python
"""
export_gguf.py — Turn a trained LoRA adapter into a servable GGUF + Ollama Modelfile.

Pipeline (see docs/reports/02-from-adapter-to-chat.md):
  1) Load base + adapter, merge the adapter INTO the weights (no more separate adapter).
  2) Export to GGUF — the compact single-file format llama.cpp / Ollama use to run models locally.
  3) Write a Modelfile so Ollama (the ollama-compose container) can register the model, after which it
     shows up automatically in open-webui.

Usage (inside unsloth-dev):
    python scripts/export_gguf.py \
        --adapter /workspace/outputs/exp1-qwen3b-reasoning/adapter \
        --out /workspace/outputs/exp1-qwen3b-reasoning/gguf \
        --quant q4_k_m --name chaotic-reasoner-3b

Then (outside the container) register with Ollama — see the printed instructions at the end.
"""
import argparse
import glob
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="trained adapter dir from train_lora.py")
    ap.add_argument("--out", required=True, help="output dir for the .gguf")
    ap.add_argument("--quant", default="q4_k_m",
                    help="GGUF quantization: q4_k_m (small/fast), q8_0 (bigger/closer to original), f16")
    ap.add_argument("--name", default="chaotic-reasoner-3b", help="Ollama model name to suggest")
    ap.add_argument("--max-seq-length", type=int, default=2048)
    args = ap.parse_args()

    from unsloth import FastLanguageModel

    os.makedirs(args.out, exist_ok=True)

    print(f"[1/3] Loading base + adapter: {args.adapter}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=False,
    )

    print(f"[2/3] Exporting GGUF ({args.quant}) → {args.out}  (this compiles llama.cpp the first time)")
    model.save_pretrained_gguf(args.out, tokenizer, quantization_method=args.quant)

    # NOTE: Unsloth writes the .gguf (and its own Modelfile) into a SIBLING dir with a "_gguf" suffix,
    # not into args.out itself (args.out gets the merged 16-bit safetensors). Search both.
    search_dirs = [args.out, args.out.rstrip("/") + "_gguf"]
    ggufs = []
    for d in search_dirs:
        ggufs += glob.glob(os.path.join(d, "*.gguf"))
    ggufs = sorted(ggufs)
    gguf = ggufs[-1] if ggufs else None
    gguf_dir = os.path.dirname(gguf) if gguf else args.out
    gguf_name = os.path.basename(gguf) if gguf else "model.gguf"

    print("[3/3] Preparing Modelfile")
    # Prefer Unsloth's generated Modelfile (correct chat template + stop tokens); just cool the temperature.
    unsloth_modelfile = os.path.join(gguf_dir, "Modelfile")
    modelfile_path = unsloth_modelfile
    if os.path.exists(unsloth_modelfile):
        with open(unsloth_modelfile) as f:
            mf = f.read()
        mf = mf.replace("temperature 1.5", "temperature 0.7")
        with open(unsloth_modelfile, "w") as f:
            f.write(mf)
        print("    Using Unsloth's Modelfile (temperature lowered to 0.7).")
    else:
        with open(modelfile_path, "w") as f:
            f.write(f"FROM {gguf_name}\nPARAMETER temperature 0.7\nPARAMETER stop \"<|im_end|>\"\n")
        print("    Wrote a minimal Modelfile.")

    print("\n" + "=" * 70)
    print(f"GGUF: {gguf}  ({os.path.getsize(gguf) / 1e9:.2f} GB)" if gguf else "GGUF: NOT FOUND")
    print(f"Modelfile: {modelfile_path}")
    print("=" * 70)
    print("\nNext — register with the Ollama server (run on the Spark HOST, not in this container):")
    print(f"  HOST_DIR=~/chaotic-finetuning/{os.path.relpath(gguf_dir, '/workspace')}")
    print(f"  docker exec ollama-compose mkdir -p /tmp/{args.name}")
    print(f"  docker cp $HOST_DIR/{gguf_name} ollama-compose:/tmp/{args.name}/")
    print(f"  docker cp $HOST_DIR/Modelfile  ollama-compose:/tmp/{args.name}/")
    print(f"  docker exec ollama-compose bash -lc 'cd /tmp/{args.name} && ollama create {args.name} -f Modelfile'")
    print(f"  docker exec ollama-compose ollama list")
    print("\nThen open open-webui and pick the model from the dropdown.")


if __name__ == "__main__":
    main()
