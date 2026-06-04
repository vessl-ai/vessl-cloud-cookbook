"""Build a packed ShareGPT dataset for LoRA SFT throughput measurement.

Renders ShareGPT conversations through the model's chat template, tokenizes
them, and greedy-packs the token stream into fixed-length sequences so every
training step processes a constant token budget (clean throughput numbers).

Source corpus: ShareGPT V3 unfiltered/cleaned split (downloaded via the
HuggingFace cache). Point HF_HOME at your cache, or pass --src to a local copy
of ShareGPT_V3_unfiltered_cleaned_split.json.

Inputs:
  ShareGPT V3 JSON (auto-discovered under $HF_HOME/hub, or via --src)
  Tokenizer for --model-id (default from the MODEL_ID env var)

Output:
  HF Arrow dataset (--out) with columns input_ids / labels / attention_mask,
  each row a packed sequence of length --max-length.

Usage:
  python data_prep/pack_sharegpt.py \
      --out data/datasets/sharegpt_packed_2048 \
      --max-length 2048 --num-sequences 1024 --seed 0

Copyright 2026 VESSL AI Inc. Licensed under Apache-2.0.
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer

HF_HOME = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface")))
DEFAULT_MODEL_ID = os.environ.get("MODEL_ID", "google/gemma-4-31B-it")


def find_sharegpt():
    cands = list(HF_HOME.glob(
        "hub/datasets--anon8231489123--ShareGPT_Vicuna_unfiltered/**/ShareGPT_V3_unfiltered_cleaned_split.json"
    ))
    if not cands:
        sys.exit(
            f"ShareGPT JSON not found under {HF_HOME}/hub/. "
            "Download the dataset (anon8231489123/ShareGPT_Vicuna_unfiltered) "
            "or pass --src pointing at ShareGPT_V3_unfiltered_cleaned_split.json"
        )
    return cands[0]


def conv_to_text(conv, tok):
    """Render a ShareGPT conversation as chat-template text."""
    msgs = []
    for turn in conv.get("conversations", []):
        role = turn.get("from")
        content = turn.get("value", "")
        if not content:
            continue
        if role == "human":
            msgs.append({"role": "user", "content": content})
        elif role == "gpt":
            msgs.append({"role": "assistant", "content": content})
    if len(msgs) < 2:
        return None
    if msgs[0]["role"] != "user":
        return None
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    except Exception:
        return None
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", default=None,
                    help="Path to ShareGPT_V3_unfiltered_cleaned_split.json "
                         "(default: auto-discover under $HF_HOME/hub)")
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID,
                    help="Tokenizer / model id (default: $MODEL_ID)")
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--num-sequences", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src = Path(args.src) if args.src else find_sharegpt()
    print(f"[pack] src: {src}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    print(f"[pack] tokenizer: {tok.__class__.__name__} vocab={tok.vocab_size}", flush=True)

    with open(src) as f:
        data = json.load(f)
    print(f"[pack] loaded {len(data)} conversations", flush=True)

    rng = random.Random(args.seed)
    rng.shuffle(data)

    # Render to text, tokenize, then greedy-pack into fixed buffers.
    target_tokens = args.max_length * args.num_sequences
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    eos = [tok.eos_token_id] if tok.eos_token_id is not None else []

    buf = []
    packs = []
    rendered_n = 0
    for conv in data:
        text = conv_to_text(conv, tok)
        if text is None:
            continue
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) < 32:
            continue
        # Truncate single example to <= max_length-len(eos) to avoid mid-conv split.
        ids = ids[: args.max_length - len(eos)]
        buf.extend(ids)
        buf.extend(eos)
        rendered_n += 1
        while len(buf) >= args.max_length:
            chunk = buf[: args.max_length]
            buf = buf[args.max_length :]
            packs.append(chunk)
            if len(packs) >= args.num_sequences:
                break
        if len(packs) >= args.num_sequences:
            break

    print(f"[pack] rendered={rendered_n} conversations  packs={len(packs)} (target={args.num_sequences})", flush=True)

    if len(packs) < args.num_sequences:
        sys.exit(f"[pack] not enough data: got {len(packs)} packs, need {args.num_sequences}")

    ds = Dataset.from_dict({
        "input_ids": packs,
        "labels": packs,  # causal LM, same as input_ids
        "attention_mask": [[1] * args.max_length for _ in packs],
    })
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_path))
    total_tok = len(packs) * args.max_length
    print(f"[pack] wrote {out_path}  ({len(packs)} packs x {args.max_length} tokens = {total_tok:,} tokens)", flush=True)


if __name__ == "__main__":
    main()
