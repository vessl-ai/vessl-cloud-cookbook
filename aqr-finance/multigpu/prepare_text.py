"""
prepare_text.py — raw-TEXT point-in-time FineWeb slice for the axolotl/FSDP arm.

WHY: the cookbook's prepare.py writes pre-tokenized uint32 .bin shards. axolotl's
`type: completion` / pretraining datasets tokenize raw text internally, so they need
JSONL with a "text" field — NOT pre-tokenized ints. This script mirrors prepare.py's
chronological filter EXACTLY (CC-MAIN <= 2017-W26 = the lookahead-bias cutoff, the
scientific core) but emits raw text instead of tokens.

Token budget is estimated by characters (~4 chars/token) to stay fast and avoid loading
the tokenizer. For the full 1B-token run, prefer axolotl `pretraining_dataset` streaming
over a giant local JSONL.

Usage:
    uv run prepare_text.py --target_tokens 5M     # tiny slice for the FSDP dry-run
    uv run prepare_text.py --target_tokens 1B     # full run (large file — consider streaming)
"""
import argparse
import json
import os
import time
from pathlib import Path

from datasets import load_dataset

CUTOFF_YEAR, CUTOFF_WEEK = 2017, 26
EST_CHARS_PER_TOKEN = 4
OUT_DIR = Path(os.environ.get("AQR_CACHE_DIR", str(Path.home() / ".cache" / "aqr-finance"))) / "text"
OUT_PATH = OUT_DIR / "fineweb-pit-2017-06.jsonl"


def is_pre_cutoff(dump: str) -> bool:
    """IDENTICAL filter to prepare.py.is_pre_cutoff — keep in sync."""
    p = dump.split("-")
    if len(p) != 4 or p[0] != "CC" or p[1] != "MAIN":
        return False
    try:
        yr, wk = int(p[2]), int(p[3])
    except ValueError:
        return False
    return yr < CUTOFF_YEAR or (yr == CUTOFF_YEAR and wk <= CUTOFF_WEEK)


def parse_count(s: str) -> int:
    s = s.strip().upper()
    for suf, mul in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if s.endswith(suf):
            return int(float(s[:-1]) * mul)
    return int(float(s))


def main() -> None:
    ap = argparse.ArgumentParser(description="Raw-text point-in-time FineWeb slice for axolotl.")
    ap.add_argument("--target_tokens", default="5M", help="'5M', '1B', or int (est by chars).")
    args = ap.parse_args()
    target = parse_count(args.target_tokens)
    target_chars = target * EST_CHARS_PER_TOKEN

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)

    seen = kept = total_chars = 0
    t0 = last = time.time()
    with OUT_PATH.open("w") as f:
        for row in ds:
            seen += 1
            if not is_pre_cutoff(row.get("dump", "")):
                continue
            text = row.get("text") or ""
            if not text:
                continue
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            kept += 1
            total_chars += len(text)
            now = time.time()
            if now - last > 30:
                print(f"prepare_text.py: seen {seen:,} kept {kept:,} ~{total_chars // EST_CHARS_PER_TOKEN:,} tok", flush=True)
                last = now
            if total_chars >= target_chars:
                break

    est_tok = total_chars // EST_CHARS_PER_TOKEN
    print(f"prepare_text.py: wrote {kept:,} docs (~{est_tok:,} tokens) to {OUT_PATH} in {time.time() - t0:.0f}s", flush=True)
    # `datasets` streaming spawns background fetch threads whose teardown can crash the
    # interpreter (Bad file descriptor / GIL) at exit. The file is fully written + flushed
    # above, so hard-exit before finalization runs.
    os._exit(0)


if __name__ == "__main__":
    main()
