"""
eval.py — Kaggle JPX leakage on/off R^2 with base-vs-adapter A/B and
date-conditional prompts.

What this script does:
  1. Load the Kaggle JPX Tokyo Stock Exchange Prediction dataset from the
     cache volume (~/.cache/aqr-finance/jpx/stock_prices.csv). The dataset
     must be pre-staged by batch-job/prep.sh.
  2. Compute per-row numeric features per stock (ret_5d, ret_30d, vol_20d,
     log_close, log_volume) from price history.
  3. Sample (stock, date) pairs — DATES_PER_STOCK rows per stock from a
     subset of MAX_STOCKS. This gives us a (firm, date) eval where the
     LLM prompt is date-conditional (not constant per stock).
  4. For BOTH the base model AND the LoRA-adapter model, extract the
     last-layer last-token hidden state from a date-conditional prompt
     that mentions the recent returns and volatility. This captures the
     LLM's date-specific reading rather than a stock fixed effect.
  5. Concatenate LLM embedding with the numeric features, fit Ridge
     regression, and compute four R^2 scores:
       base_r2_off    — base model, chronological split (train <= 2020-12-31, test >= 2021-01-01)
       base_r2_on     — base model, stock-disjoint GroupKFold CV mean
       adapter_r2_off — adapter, chronological split
       adapter_r2_on  — adapter, stock-disjoint GroupKFold CV mean
  6. The two leakage premiums:
       base_premium    = base_r2_on - base_r2_off
       adapter_premium = adapter_r2_on - adapter_r2_off
       premium_reduction = base_premium - adapter_premium
     If LoRA continued PT on a chronologically-filtered FineWeb slice did
     its job, adapter_premium should be smaller than base_premium —
     "premium_reduction" quantifies the lookahead-bias removal.
  7. Print a summary block. Keys match batch-job/wait-jobs.sh grep pattern.

Usage (inside the VESSL container, via batch-job/submit.sh):
    uv run eval.py
"""

import gc
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
from sklearn.model_selection import GroupKFold
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning)

BASE_MODEL = "Qwen/Qwen3.5-35B-A3B-Base"
CACHE_DIR = Path(os.environ.get("AQR_CACHE_DIR", str(Path.home() / ".cache" / "aqr-finance")))
ADAPTER_DIR = CACHE_DIR / "adapter"
JPX_DIR = CACHE_DIR / "jpx"
JPX_CSV = JPX_DIR / "stock_prices.csv"

CHRONO_TRAIN_END = "2020-12-31"
CHRONO_TEST_START = "2021-01-01"

MAX_STOCKS = int(os.environ.get("AQR_MAX_STOCKS", "1000"))             # subset for the eval; raise after sanity passes
DATES_PER_STOCK = int(os.environ.get("AQR_DATES_PER_STOCK", "30"))         # (stock, date) pairs sampled per stock => ~6,000 rows
SEQ_LEN = 256
BATCH_SIZE = 16              # forward batch for embedding; 6000 rows / 16 ≈ 375 batches
RIDGE_ALPHA = 1.0
KFOLD_N = 5
N_BOOTSTRAP = int(os.environ.get("AQR_N_BOOTSTRAP", "200"))   # clustered-bootstrap iterations for the premium CIs

NUMERIC_FEATURE_COLS = ["log_close", "log_volume", "ret_5d", "ret_30d", "vol_20d"]


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
    return df


def compute_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-stock rolling features. Sorts by (code, date)."""
    df = df.sort_values(["SecuritiesCode", "Date"]).reset_index(drop=True)
    df["log_close"] = np.log(df["Close"].clip(lower=1e-6))
    df["log_volume"] = np.log(df["Volume"].clip(lower=1.0))
    g = df.groupby("SecuritiesCode", group_keys=False)
    df["ret_5d"] = g["log_close"].diff(5)
    df["ret_30d"] = g["log_close"].diff(30)
    df["vol_20d"] = g["log_close"].diff().rolling(20).std().reset_index(level=0, drop=True)
    df = df.dropna(subset=NUMERIC_FEATURE_COLS).reset_index(drop=True)
    return df


def select_subset(df: pd.DataFrame, max_stocks: int, dates_per_stock: int) -> pd.DataFrame:
    """Top-N stocks by row count, then random K dates per stock."""
    top = df["SecuritiesCode"].value_counts().head(max_stocks).index.tolist()
    sub = df[df["SecuritiesCode"].isin(top)]
    rng = np.random.default_rng(42)

    def sample_group(g):
        if len(g) <= dates_per_stock:
            return g
        idx = rng.choice(len(g), size=dates_per_stock, replace=False)
        return g.iloc[idx]

    # Manual concat instead of groupby.apply — pandas 3.0 made
    # include_groups=False the default for DataFrameGroupBy.apply, which
    # drops the SecuritiesCode column from the result and downstream
    # df['SecuritiesCode'] lookups KeyError. Manual loop sidesteps the
    # version-sensitive behavior entirely.
    parts = [sample_group(g) for _, g in sub.groupby("SecuritiesCode")]
    if not parts:
        return sub.iloc[:0].copy()
    return pd.concat(parts, ignore_index=True)


def build_prompt(row) -> str:
    """Date-conditional prompt: stock + recent return/volatility context."""
    return (
        f"On {row['Date'].strftime('%Y-%m-%d')}, "
        f"Japanese Tokyo Stock Exchange securities code {int(row['SecuritiesCode'])} "
        f"had recent 5-day log return {row['ret_5d']:+.4f}, "
        f"30-day log return {row['ret_30d']:+.4f}, "
        f"20-day volatility {row['vol_20d']:.4f}."
    )


@torch.no_grad()
def llm_embeddings(model, tokenizer, prompts):
    """Last-layer last-non-pad-token hidden state per prompt, batched.
    Returns (N, H) float32.

    Per-prompt forward is ~1.5 s for 35 B bf16 on H100; with batch=16 the
    throughput goes up ~10-15x (the bottleneck shifts from launch overhead
    to attention compute) and 6k prompts finish in ~5-10 min per model.
    """
    model.eval()
    device = next(model.parameters()).device
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    embs = []
    t0 = time.time()
    last_log = t0
    for batch_start in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[batch_start:batch_start + BATCH_SIZE]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=SEQ_LEN,
        )
        ids = enc.input_ids.to(device)
        mask = enc.attention_mask.to(device)
        out = model(
            input_ids=ids,
            attention_mask=mask,
            output_hidden_states=True,
            use_cache=False,
        )
        last_hidden = out.hidden_states[-1]  # (B, T, H)
        # Index of the last non-pad token per sequence.
        seq_lens = mask.sum(dim=1) - 1  # (B,)
        for i in range(len(batch)):
            emb = last_hidden[i, int(seq_lens[i].item()), :].float().cpu().numpy()
            embs.append(emb)
        done = batch_start + len(batch)
        now = time.time()
        if now - last_log > 30:
            rate = done / max(1.0, now - t0)
            eta = (len(prompts) - done) / max(1.0, rate)
            print(
                f"eval.py: embedded {done}/{len(prompts)} prompts "
                f"({rate:.1f} prompts/s, ETA {eta:.0f}s)",
                flush=True,
            )
            last_log = now
    return np.stack(embs, axis=0)


def fit_and_score(X: np.ndarray, y: np.ndarray, dates: pd.Series, groups: pd.Series) -> dict:
    """Return r2_off (chronological), r2_on (stock-disjoint GroupKFold), sizes.

    r2_off  — chronological split (train <= 2020-12-31, test >= 2021-01-01):
              the honest, no-leakage estimate of out-of-sample skill.
    r2_on   — GroupKFold on SecuritiesCode so a stock never lands in both
              train and test. A plain shuffle KFold lets the SAME stock sit on
              both sides, inflating r2_on via a memorized stock fixed effect
              rather than genuine future-reading. GroupKFold strips that
              shortcut, leaving the temporal-leakage component we want.
    """
    mask_train = dates <= pd.Timestamp(CHRONO_TRAIN_END)
    mask_test = dates >= pd.Timestamp(CHRONO_TEST_START)
    Xtr, ytr = X[mask_train.values], y[mask_train.values]
    Xte, yte = X[mask_test.values], y[mask_test.values]
    if len(Xtr) < 10 or len(Xte) < 10:
        raise RuntimeError(
            f"chronological split too small: train={len(Xtr)}, test={len(Xte)}"
        )
    reg_off = Ridge(alpha=RIDGE_ALPHA).fit(Xtr, ytr)
    r2_off = r2_score(yte, reg_off.predict(Xte))

    gkf = GroupKFold(n_splits=KFOLD_N)
    r2s = []
    for tr_idx, te_idx in gkf.split(X, y, groups=groups.values):
        reg_on = Ridge(alpha=RIDGE_ALPHA).fit(X[tr_idx], y[tr_idx])
        r2s.append(r2_score(y[te_idx], reg_on.predict(X[te_idx])))
    r2_on = float(np.mean(r2s))

    return {
        "r2_off": r2_off,
        "r2_on": r2_on,
        "n_train": len(Xtr),
        "n_test": len(Xte),
        "n_total": len(X),
    }


def _premium(scores: dict) -> float:
    return scores["r2_on"] - scores["r2_off"]


def bootstrap_premium_ci(
    X_base: np.ndarray,
    X_adapter: np.ndarray,
    y: np.ndarray,
    dates: pd.Series,
    groups: pd.Series,
    n_bootstrap: int,
) -> dict:
    """Clustered bootstrap: resample whole stocks (not rows) with replacement.

    Each draw rebuilds a panel of the same number of stocks, refits the
    chronological + GroupKFold Ridge for both base and adapter, and records
    base_premium, adapter_premium, and premium_reduction. Resampling stocks
    (clusters) rather than naive rows keeps the within-stock date correlation
    intact, so the 95% interval is honest under the panel structure. A stock
    drawn more than once keeps all its rows in a single cluster (grouped by
    original stock id) so GroupKFold never splits identical rows across folds.
    """
    uniq = np.array(sorted(pd.unique(groups.values)))
    by_stock = {s: np.where(groups.values == s)[0] for s in uniq}
    rng = np.random.default_rng(123)
    reductions, base_ps, adapter_ps = [], [], []
    for b in range(n_bootstrap):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by_stock[s] for s in pick])
        # Group by the ORIGINAL stock id so a stock drawn more than once keeps
        # all its (identical) rows together in one GroupKFold fold. Distinct
        # per-draw ids would let duplicated rows land on opposite sides of a
        # fold — the model would test on rows it trained on, inflating r2_on
        # and biasing the premium upward.
        g_labels = np.concatenate(
            [np.full(len(by_stock[s]), s, dtype=np.int64) for s in pick]
        )
        d = pd.Series(dates.values[idx])
        g = pd.Series(g_labels)
        try:
            bs = fit_and_score(X_base[idx], y[idx], d, g)
            asc = fit_and_score(X_adapter[idx], y[idx], d, g)
        except (RuntimeError, ValueError):
            continue
        bp, ap = _premium(bs), _premium(asc)
        base_ps.append(bp)
        adapter_ps.append(ap)
        reductions.append(bp - ap)
        if (b + 1) % 25 == 0:
            print(
                f"eval.py: bootstrap {b + 1}/{n_bootstrap} (kept {len(reductions)})",
                flush=True,
            )

    def ci(vals: list) -> dict:
        a = np.asarray(vals, dtype=float)
        if a.size == 0:
            return {"p2.5": float("nan"), "p50": float("nan"),
                    "p97.5": float("nan"), "std": float("nan")}
        return {
            "p2.5": float(np.percentile(a, 2.5)),
            "p50": float(np.percentile(a, 50)),
            "p97.5": float(np.percentile(a, 97.5)),
            "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        }

    return {
        "premium_reduction": ci(reductions),
        "base_premium": ci(base_ps),
        "adapter_premium": ci(adapter_ps),
        "n_bootstrap_completed": len(reductions),
    }


def embed_with_model(model_loader, tokenizer, prompts, label: str) -> np.ndarray:
    """Load a model via the loader, embed prompts, free the model."""
    print(f"eval.py: loading {label} model", flush=True)
    t0 = time.time()
    model = model_loader()
    print(f"eval.py: {label} loaded in {time.time() - t0:.0f}s", flush=True)
    embs = llm_embeddings(model, tokenizer, prompts)
    print(f"eval.py: {label} embedded {len(embs)} prompts", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return embs


def main():
    print("eval.py: starting", flush=True)
    t0 = time.time()

    # JPX data + numeric features + subset sampling.
    df = load_jpx()
    print(f"eval.py: raw JPX rows = {len(df):,}", flush=True)
    df = compute_numeric_features(df)
    print(f"eval.py: with numeric features = {len(df):,} (post-NaN drop)", flush=True)
    df = select_subset(df, MAX_STOCKS, DATES_PER_STOCK)
    print(
        f"eval.py: subset = {len(df):,} rows over "
        f"{df['SecuritiesCode'].nunique()} stocks",
        flush=True,
    )

    # Date-conditional prompts.
    prompts = df.apply(build_prompt, axis=1).tolist()
    print(f"eval.py: built {len(prompts)} date-conditional prompts", flush=True)
    print(f"eval.py: example prompt — {prompts[0]!r}", flush=True)

    # Numeric feature matrix + target.
    X_num = df[NUMERIC_FEATURE_COLS].values
    y = df["Target"].values
    dates = df["Date"].reset_index(drop=True)
    groups = df["SecuritiesCode"].reset_index(drop=True)

    # Tokenizer (shared between base + adapter).
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # Loaders for base and (if present) adapter. device_map="auto" spreads
    # the 35B bf16 base (~70 GB) across whatever GPUs the container exposes.
    def load_base():
        return AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )

    def load_adapter():
        base = load_base()
        return PeftModel.from_pretrained(base, str(ADAPTER_DIR))

    have_adapter = ADAPTER_DIR.exists() and any(ADAPTER_DIR.iterdir())
    if not have_adapter:
        print(
            f"eval.py: WARNING — no adapter at {ADAPTER_DIR}; "
            f"evaluating BASE only (sanity-check mode)",
            flush=True,
        )

    # Embed with base, then (if present) with adapter.
    base_embs = embed_with_model(load_base, tokenizer, prompts, "base")
    if have_adapter:
        adapter_embs = embed_with_model(load_adapter, tokenizer, prompts, "adapter")
    else:
        adapter_embs = None

    # Concatenate LLM embedding with numeric features for the regression.
    X_base = np.concatenate([base_embs, X_num], axis=1)
    base_scores = fit_and_score(X_base, y, dates, groups)
    print(
        f"eval.py: base r2_off = {base_scores['r2_off']:.4f}, "
        f"r2_on = {base_scores['r2_on']:.4f}",
        flush=True,
    )

    if adapter_embs is not None:
        X_adapter = np.concatenate([adapter_embs, X_num], axis=1)
        adapter_scores = fit_and_score(X_adapter, y, dates, groups)
        print(
            f"eval.py: adapter r2_off = {adapter_scores['r2_off']:.4f}, "
            f"r2_on = {adapter_scores['r2_on']:.4f}",
            flush=True,
        )
    else:
        adapter_scores = {"r2_off": float("nan"), "r2_on": float("nan")}

    base_premium = base_scores["r2_on"] - base_scores["r2_off"]
    adapter_premium = adapter_scores["r2_on"] - adapter_scores["r2_off"]
    premium_reduction = base_premium - adapter_premium

    # Clustered-bootstrap CIs (resample stocks). Only meaningful when both
    # base and adapter embeddings exist.
    if adapter_embs is not None:
        print(
            f"eval.py: clustered bootstrap ({N_BOOTSTRAP} draws over "
            f"{groups.nunique()} stocks) — the slow part",
            flush=True,
        )
        tb = time.time()
        cis = bootstrap_premium_ci(X_base, X_adapter, y, dates, groups, N_BOOTSTRAP)
        print(f"eval.py: bootstrap done in {time.time() - tb:.0f}s", flush=True)
    else:
        nan_ci = {"p2.5": float("nan"), "p97.5": float("nan")}
        cis = {
            "premium_reduction": nan_ci,
            "base_premium": nan_ci,
            "adapter_premium": nan_ci,
            "n_bootstrap_completed": 0,
        }

    # Summary (keys must match batch-job/wait-jobs.sh grep pattern).
    # Top-level r2_leakage_* keys mirror adapter scores so the cookbook's
    # "real metric" sits at the top of the block; base scores follow.
    print("\n--- eval.py summary ---", flush=True)
    print(f"r2_leakage_off:        {adapter_scores['r2_off']:.4f}", flush=True)
    print(f"r2_leakage_on:         {adapter_scores['r2_on']:.4f}", flush=True)
    print(f"leakage_premium:       {adapter_premium:.4f}", flush=True)
    print(f"base_r2_leakage_off:   {base_scores['r2_off']:.4f}", flush=True)
    print(f"base_r2_leakage_on:    {base_scores['r2_on']:.4f}", flush=True)
    print(f"base_leakage_premium:  {base_premium:.4f}", flush=True)
    print(f"premium_reduction:     {premium_reduction:.4f}", flush=True)
    # Bootstrap median (p50) reported alongside the percentile CI: the
    # clustered bootstrap of an R²-difference is right-skewed, so the
    # full-data plug-in point estimate can sit near (or just below) the 2.5th
    # percentile. The median is the self-consistent central estimate — it
    # always lies inside [p2.5, p97.5] — so the cookbook leads with it.
    print(
        f"premium_reduction_ci95: [{cis['premium_reduction']['p2.5']:.4f}, "
        f"{cis['premium_reduction']['p97.5']:.4f}] (median {cis['premium_reduction']['p50']:.4f})",
        flush=True,
    )
    print(
        f"base_premium_ci95:     [{cis['base_premium']['p2.5']:.4f}, "
        f"{cis['base_premium']['p97.5']:.4f}] (median {cis['base_premium']['p50']:.4f})",
        flush=True,
    )
    print(
        f"adapter_premium_ci95:  [{cis['adapter_premium']['p2.5']:.4f}, "
        f"{cis['adapter_premium']['p97.5']:.4f}] (median {cis['adapter_premium']['p50']:.4f})",
        flush=True,
    )
    print(f"n_bootstrap_completed: {cis['n_bootstrap_completed']}", flush=True)
    print(f"n_test_samples:        {base_scores['n_test']}", flush=True)
    print(f"n_total_samples:       {base_scores['n_total']}", flush=True)
    print(f"n_unique_stocks:       {df['SecuritiesCode'].nunique()}", flush=True)
    print(f"dates_per_stock:       {DATES_PER_STOCK}", flush=True)
    print("eval_methodology:      GroupKFold(stock-disjoint)+clustered-bootstrap", flush=True)
    print(f"eval_seconds:          {time.time() - t0:.1f}", flush=True)
    print("---", flush=True)


if __name__ == "__main__":
    main()
