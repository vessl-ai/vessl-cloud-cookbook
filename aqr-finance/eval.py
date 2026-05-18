"""
eval.py — Kaggle JPX leakage on/off R^2 downstream evaluation.

What this script does:
  1. Load the LoRA adapter saved by train.py (~/.cache/aqr-finance/adapter/)
     stacked on top of the Qwen3.5-35B-A3B-Base.
  2. Load the Kaggle JPX Tokyo Stock Exchange Prediction dataset from the
     cache volume (~/.cache/aqr-finance/jpx/stock_prices.csv). The dataset
     must be pre-staged by batch-job/prep.sh (it requires Kaggle creds).
  3. For each unique stock, build a minimal text identifier and extract the
     last-layer last-token hidden state from the LLM. Concatenate with one
     numeric feature (log_close) per row.
  4. Fit a Ridge regression and compute two R^2 scores:
       r2_leakage_off : chronological split — train <= 2020-12-31,
                        test >= 2021-01-01. Model was trained on FineWeb
                        with a CC dump cutoff of 2017-06-30, so test years
                        are strictly out-of-sample for the LLM corpus.
       r2_leakage_on  : random 5-fold CV mean over the same rows. This is
                        the "ceiling" — pure model capacity without
                        chronological structure.
     leakage_premium = r2_leakage_on - r2_leakage_off.
  5. Print a summary block. Keys match batch-job/wait-jobs.sh grep pattern.

KNOWN LIMITATION (will be improved in a follow-up):
  The per-stock embedding is constant across all dates of a given stock
  (we encode only "securities code N", not date-conditional context). That
  makes the LLM feature ≈ a stock-fixed-effect from the regression's point
  of view. The numbers are still meaningful for a sanity-check baseline
  ("does the adapter shift R^2 vs base?"), but treat them as a first cut.
  A date-conditional prompt (latest news headline, recent price moves) is
  the natural upgrade.

Usage (inside the VESSL container, via batch-job/submit.sh):
    uv run eval.py
"""

import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

BASE_MODEL = "Qwen/Qwen3.5-35B-A3B-Base"
CACHE_DIR = Path(os.environ.get("AQR_CACHE_DIR", str(Path.home() / ".cache" / "aqr-finance")))
ADAPTER_DIR = CACHE_DIR / "adapter"
JPX_DIR = CACHE_DIR / "jpx"
JPX_CSV = JPX_DIR / "stock_prices.csv"

# Chronological split. The base model's FineWeb cutoff is 2017-06-30, so the
# 2021+ test window is strictly out-of-sample for the LLM corpus.
CHRONO_TRAIN_END = "2020-12-31"
CHRONO_TEST_START = "2021-01-01"

MAX_STOCKS = 200  # subset for a fast dry-run; raise after sanity passes
SEQ_LEN = 256
RIDGE_ALPHA = 1.0
KFOLD_N = 5


def load_jpx() -> pd.DataFrame:
    if not JPX_CSV.exists():
        raise FileNotFoundError(
            f"{JPX_CSV} not found. The Kaggle JPX dataset must be pre-staged on "
            f"the cache volume. Run `bash batch-job/prep.sh` with Kaggle creds "
            f"(KAGGLE_USERNAME + KAGGLE_KEY env vars or ~/.kaggle/kaggle.json) "
            f"or download it manually with `kaggle competitions download "
            f"-c jpx-tokyo-stock-exchange-prediction`."
        )
    df = pd.read_csv(JPX_CSV)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.dropna(subset=["Target"]).copy()
    df["log_close"] = np.log(df["Close"].clip(lower=1e-6))
    return df


def select_stocks(df: pd.DataFrame) -> pd.DataFrame:
    top = df["SecuritiesCode"].value_counts().head(MAX_STOCKS).index.tolist()
    return df[df["SecuritiesCode"].isin(top)].reset_index(drop=True)


def build_prompt(code: int) -> str:
    return f"Japanese Tokyo Stock Exchange listed company, securities code {code}."


@torch.no_grad()
def llm_embeddings(model, tokenizer, codes):
    """Last-layer last-token hidden state for each stock (cached per code)."""
    model.eval()
    device = next(model.parameters()).device
    embs = []
    for code in codes:
        text = build_prompt(int(code))
        ids = tokenizer(
            text,
            return_tensors="pt",
            max_length=SEQ_LEN,
            truncation=True,
        ).input_ids.to(device)
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
        last_hidden = out.hidden_states[-1]  # (1, T, H)
        emb = last_hidden[0, -1, :].float().cpu().numpy()
        embs.append(emb)
    return np.stack(embs, axis=0)


def main():
    print("eval.py: starting", flush=True)
    t0 = time.time()

    # Load base + adapter onto a single GPU. Eval is light enough that we
    # don't need multi-GPU here.
    print(f"eval.py: loading base model {BASE_MODEL}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    # device_map="auto" so 35B bf16 (~70 GB) spreads across whatever GPUs
    # the container exposes. On 8xH100 SXM single-node the model sits on
    # one or two devices and embedding extraction is fast.
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    if ADAPTER_DIR.exists() and any(ADAPTER_DIR.iterdir()):
        print(f"eval.py: loading LoRA adapter from {ADAPTER_DIR}", flush=True)
        model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
    else:
        print(
            f"eval.py: WARNING — no adapter at {ADAPTER_DIR}, "
            f"evaluating base model only (sanity-check mode)",
            flush=True,
        )
        model = base
    model.eval()

    # JPX data.
    df = load_jpx()
    df = select_stocks(df)
    print(
        f"eval.py: JPX rows = {len(df):,}, "
        f"unique stocks = {df['SecuritiesCode'].nunique()}",
        flush=True,
    )

    # Per-stock LLM embedding (cached, reused across all rows of that stock).
    unique_codes = df["SecuritiesCode"].unique().tolist()
    print(
        f"eval.py: computing LLM embeddings for {len(unique_codes)} stocks",
        flush=True,
    )
    embs = llm_embeddings(model, tokenizer, unique_codes)
    code_to_emb = dict(zip(unique_codes, embs))

    # Feature matrix.
    X_llm = np.stack([code_to_emb[c] for c in df["SecuritiesCode"]], axis=0)
    X_num = df[["log_close"]].values
    X = np.concatenate([X_llm, X_num], axis=1)
    y = df["Target"].values

    # r2_leakage_off — chronological split.
    mask_train = df["Date"] <= pd.Timestamp(CHRONO_TRAIN_END)
    mask_test = df["Date"] >= pd.Timestamp(CHRONO_TEST_START)
    Xtr, ytr = X[mask_train], y[mask_train]
    Xte, yte = X[mask_test], y[mask_test]
    if len(Xtr) < 10 or len(Xte) < 10:
        raise RuntimeError(
            f"chronological split too small: train={len(Xtr)}, test={len(Xte)}"
        )
    reg_off = Ridge(alpha=RIDGE_ALPHA).fit(Xtr, ytr)
    r2_off = r2_score(yte, reg_off.predict(Xte))
    print(
        f"eval.py: r2_leakage_off (chronological) = {r2_off:.4f} "
        f"(train n={len(Xtr):,}, test n={len(Xte):,})",
        flush=True,
    )

    # r2_leakage_on — random 5-fold CV mean.
    kf = KFold(n_splits=KFOLD_N, shuffle=True, random_state=42)
    r2s = []
    for fold_i, (tr_idx, te_idx) in enumerate(kf.split(X)):
        reg_on = Ridge(alpha=RIDGE_ALPHA).fit(X[tr_idx], y[tr_idx])
        r2s.append(r2_score(y[te_idx], reg_on.predict(X[te_idx])))
    r2_on = float(np.mean(r2s))
    print(
        f"eval.py: r2_leakage_on (random {KFOLD_N}-fold mean) = {r2_on:.4f}",
        flush=True,
    )

    leakage_premium = r2_on - r2_off

    print("\n--- eval.py summary ---", flush=True)
    print(f"r2_leakage_off:    {r2_off:.4f}", flush=True)
    print(f"r2_leakage_on:     {r2_on:.4f}", flush=True)
    print(f"leakage_premium:   {leakage_premium:.4f}", flush=True)
    print(f"n_test_samples:    {len(Xte)}", flush=True)
    print(f"n_total_samples:   {len(X)}", flush=True)
    print(f"n_unique_stocks:   {len(unique_codes)}", flush=True)
    print(f"eval_seconds:      {time.time() - t0:.1f}", flush=True)
    print("---", flush=True)


if __name__ == "__main__":
    main()
