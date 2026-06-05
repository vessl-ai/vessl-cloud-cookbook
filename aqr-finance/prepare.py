"""
prepare.py — FineWeb chronologically-filtered slice + Qwen pretokenization.

What this script does:
  1. Stream HuggingFaceFW/fineweb. (Fineweb is 15 TB; we never download it.)
  2. Keep only Common Crawl dumps published on or before 2017-06-30
     (CC-MAIN-YYYY-WW <= CC-MAIN-2017-26). This is the lookahead-bias cutoff:
     a model trained on this slice has never seen post-2017-06 web text,
     which is what makes the downstream JPX leakage premium measurable.
  3. Tokenize with the Qwen/Qwen3.5-35B-A3B-Base tokenizer.
  4. Write uint32 token shards to ~/.cache/aqr-finance/data/.

Usage:
    uv run prepare.py                          # full 1B-token prep
    uv run prepare.py --target_tokens 100M     # debug 100M slice
    uv run prepare.py --force                  # re-run even if cache matches

Cache layout (under ~/.cache/aqr-finance/):
    data/manifest.json    — tokens, shards, eos_id, vocab_size
    data/train_NNNNN.bin  — uint32 token shards (50M tokens each)
    data/val_NNNNN.bin    — uint32 token shards (val held-out)

Subsequent train.py runs reuse the cache if the manifest matches.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Config — fixed constants. Do not modify in experiments.
# ---------------------------------------------------------------------------

BASE_MODEL = "Qwen/Qwen3.5-35B-A3B-Base"

CACHE_DIR = Path(os.environ.get("AQR_CACHE_DIR", str(Path.home() / ".cache" / "aqr-finance")))
DATA_DIR = CACHE_DIR / "data"
MANIFEST_PATH = DATA_DIR / "manifest.json"

# Lookahead-bias cutoff. CC dump naming: CC-MAIN-YYYY-WW.
# 2017-W26 starts Mon 2017-06-26, so this slice captures everything published
# on or before 2017-06-30. Documents from later CC dumps are dropped.
CUTOFF_YEAR = 2017
CUTOFF_WEEK = 26

DEFAULT_TARGET_TOKENS = 1_000_000_000  # 1B tokens — sized for the LoRA budget.
DEFAULT_VAL_FRACTION = 0.005           # 0.5% held-out (~5M tokens at 1B).
SHARD_TOKENS = 50_000_000              # 50M tokens per shard (~200 MB uint32).
BATCH_DOCS = 256                       # tokenizer batch size.
BYTES_PER_TOKEN = 4                    # uint32.


def parse_dump_id(dump: str) -> tuple[int, int] | None:
    """Parse 'CC-MAIN-YYYY-WW' to (year, week). Returns None if not parseable."""
    parts = dump.split("-")
    if len(parts) != 4 or parts[0] != "CC" or parts[1] != "MAIN":
        return None
    try:
        return int(parts[2]), int(parts[3])
    except ValueError:
        return None


def is_pre_cutoff(dump: str) -> bool:
    parsed = parse_dump_id(dump)
    if parsed is None:
        return False
    yr, wk = parsed
    if yr < CUTOFF_YEAR:
        return True
    if yr == CUTOFF_YEAR and wk <= CUTOFF_WEEK:
        return True
    return False


def parse_count(s: str) -> int:
    """Parse '1e9', '1B', '100M', or a plain int."""
    s = s.strip().upper()
    if s.endswith("B"):
        return int(float(s[:-1]) * 1_000_000_000)
    if s.endswith("M"):
        return int(float(s[:-1]) * 1_000_000)
    if s.endswith("K"):
        return int(float(s[:-1]) * 1_000)
    return int(float(s))


def existing_manifest_matches(target_tokens: int, val_fraction: float) -> bool:
    if not MANIFEST_PATH.exists():
        return False
    try:
        m = json.loads(MANIFEST_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    if m.get("base_model") != BASE_MODEL:
        return False
    if m.get("cutoff") != f"CC-MAIN-{CUTOFF_YEAR}-W{CUTOFF_WEEK:02d}":
        return False
    if m.get("target_tokens") != target_tokens:
        return False
    if abs(m.get("val_fraction", -1) - val_fraction) > 1e-6:
        return False
    for shard in m.get("train_shards", []):
        if not (DATA_DIR / shard).exists():
            return False
    return True


def run(target_tokens: int, val_fraction: float) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"prepare.py: loading tokenizer for {BASE_MODEL}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise RuntimeError(f"{BASE_MODEL} tokenizer has no eos_token_id")

    print(
        f"prepare.py: streaming HuggingFaceFW/fineweb "
        f"(cutoff <= CC-MAIN-{CUTOFF_YEAR}-W{CUTOFF_WEEK:02d}, target {target_tokens:,} tokens)",
        flush=True,
    )
    ds = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)

    raw_shards: list[Path] = []
    buf = np.empty(SHARD_TOKENS, dtype=np.uint32)
    used = 0
    total = 0
    text_batch: list[str] = []
    docs_seen = 0
    docs_kept = 0
    t0 = time.time()
    last_log = t0

    pbar = tqdm(total=target_tokens, unit="tok", unit_scale=True, desc="tokens")

    def flush_buf_to_shard() -> None:
        nonlocal used
        if used == 0:
            return
        path = DATA_DIR / f"raw_{len(raw_shards):05d}.bin"
        buf[:used].tofile(path)
        raw_shards.append(path)
        used = 0

    def flush_text_batch() -> bool:
        """Returns True if target_tokens reached."""
        nonlocal used, total
        if not text_batch:
            return False
        enc = tokenizer(text_batch, add_special_tokens=False).input_ids
        text_batch.clear()
        for ids in enc:
            ids.append(eos_id)
            arr = np.asarray(ids, dtype=np.uint32)
            i = 0
            while i < len(arr) and total < target_tokens:
                room = SHARD_TOKENS - used
                take = min(room, len(arr) - i, target_tokens - total)
                buf[used:used + take] = arr[i:i + take]
                used += take
                i += take
                total += take
                pbar.update(take)
                if used == SHARD_TOKENS:
                    flush_buf_to_shard()
        return total >= target_tokens

    try:
        for row in ds:
            docs_seen += 1
            if not is_pre_cutoff(row.get("dump", "")):
                continue
            text = row.get("text") or ""
            if not text:
                continue
            docs_kept += 1
            text_batch.append(text)
            if len(text_batch) >= BATCH_DOCS:
                if flush_text_batch():
                    break

            now = time.time()
            if now - last_log > 30:
                kept_pct = 100.0 * docs_kept / max(1, docs_seen)
                rate = total / max(1.0, now - t0)
                print(
                    f"prepare.py: streamed {docs_seen:,} docs, "
                    f"kept {docs_kept:,} ({kept_pct:.1f}%), "
                    f"tokens {total:,} ({rate:,.0f}/s)",
                    flush=True,
                )
                last_log = now
        else:
            # Stream exhausted before target — flush remainder.
            flush_text_batch()
    finally:
        flush_buf_to_shard()
        pbar.close()

    if total < target_tokens:
        print(
            f"prepare.py: WARNING — stream exhausted at {total:,} tokens "
            f"(target {target_tokens:,}). Lowering target to actual.",
            flush=True,
        )
        target_tokens = total

    # Carve val off the tail. FineWeb chunks are not chronologically ordered
    # within a dump, so a tail slice is a representative mini-corpus.
    train_tokens = int(total * (1.0 - val_fraction))
    val_tokens = total - train_tokens

    train_shards_out: list[str] = []
    val_shards_out: list[str] = []
    seen = 0
    for path in raw_shards:
        size = path.stat().st_size // BYTES_PER_TOKEN
        if seen + size <= train_tokens:
            new = DATA_DIR / f"train_{len(train_shards_out):05d}.bin"
            path.rename(new)
            train_shards_out.append(new.name)
        elif seen >= train_tokens:
            new = DATA_DIR / f"val_{len(val_shards_out):05d}.bin"
            path.rename(new)
            val_shards_out.append(new.name)
        else:
            # Shard straddles the boundary — split it.
            data = np.fromfile(path, dtype=np.uint32)
            split_at = train_tokens - seen
            new_t = DATA_DIR / f"train_{len(train_shards_out):05d}.bin"
            new_v = DATA_DIR / f"val_{len(val_shards_out):05d}.bin"
            data[:split_at].tofile(new_t)
            data[split_at:].tofile(new_v)
            train_shards_out.append(new_t.name)
            val_shards_out.append(new_v.name)
            path.unlink()
        seen += size

    manifest = {
        "base_model": BASE_MODEL,
        "cutoff": f"CC-MAIN-{CUTOFF_YEAR}-W{CUTOFF_WEEK:02d}",
        "target_tokens": target_tokens,
        "val_fraction": val_fraction,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "train_shards": train_shards_out,
        "val_shards": val_shards_out,
        "vocab_size": tokenizer.vocab_size,
        "eos_token_id": eos_id,
        "docs_seen": docs_seen,
        "docs_kept": docs_kept,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))

    print(
        f"prepare.py: done — {train_tokens:,} train + {val_tokens:,} val tokens "
        f"({len(train_shards_out)} train shards, {len(val_shards_out)} val shards)",
        flush=True,
    )
    print(f"prepare.py: manifest at {MANIFEST_PATH}", flush=True)
    print(f"prepare.py: elapsed {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prep FineWeb chronological slice for AQR cookbook.")
    parser.add_argument(
        "--target_tokens",
        type=str,
        default=str(DEFAULT_TARGET_TOKENS),
        help="Total tokens to prep. Accepts '1e9', '1B', '100M', or a plain int.",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=DEFAULT_VAL_FRACTION,
        help="Fraction of total reserved for val (0-1).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if cached manifest matches.",
    )
    args = parser.parse_args()
    target = parse_count(args.target_tokens)
    if not args.force and existing_manifest_matches(target, args.val_fraction):
        print(
            f"prepare.py: cache hit — manifest matches target_tokens={target:,}, "
            f"val_fraction={args.val_fraction}. Skipping (use --force to override).",
            flush=True,
        )
    else:
        run(target, args.val_fraction)
