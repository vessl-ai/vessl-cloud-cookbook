# autoresearch on VESSL Cloud

Run [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — the
"AI agent does its own LLM research while you sleep" experiment — without
owning a GPU. Every training run is a `vesslctl job create` against a single
GPU spec; you edit code locally, the agent loop submits jobs, you wake up to
~50 experiments and a (hopefully) better model.

| GPU | Cost / experiment | Wall time per experiment | Peak VRAM | Baseline val_bpb |
|-----|------------------:|-------------------------:|----------:|-----------------:|
| A100 SXM 80 GB × 1 (betelgeuse-na, $1.55/hr) | ~$0.23 | ~8m46s (5 min train + ~3:46 startup) | 44.0 GB | 1.109645 |

Numbers measured 2026-05-03 on VESSL Cloud — full breakdown in [benchmarks.md](./benchmarks.md).

## What this recipe is

The autoresearch idea (verbatim from karpathy): give an AI agent a small but
real LLM training setup and let it experiment autonomously overnight. It
modifies the code, trains for 5 minutes, checks if val_bpb improved, keeps
or discards, repeats.

This recipe is a thin VESSL Cloud adaptation:

- **`prepare.py` and `train.py` are unmodified** from the upstream repo. The
  agent edits `train.py` directly, same as the original.
- **`program.md` is rewritten** so the agent submits VESSL jobs instead of
  running `uv run train.py` locally. The loop semantics (per-experiment
  branch, keep-or-revert, NEVER STOP) are preserved.
- **`batch-job/submit.sh` and `batch-job/prep.sh` are new.** They wrap
  `vesslctl` so the agent never needs to know the CLI.

You analyze results locally with `analysis.ipynb` and `results.tsv`, the
same way you would with a local-GPU run.

## Prerequisites

- A VESSL Cloud account with credits.
- `vesslctl` installed and authenticated (`vesslctl auth status`).
- A VESSL org and team active on `vesslctl`. The interactive `vesslctl auth
  login` flow prompts you to pick both, and `vesslctl auth status` shows the
  resolved context. To change them later without re-logging-in:
  ```bash
  vesslctl config set default_org  <your-org>
  vesslctl config set default_team <your-team>
  vesslctl auth status   # confirm
  ```
  Or override per-command with `--org` / `--team` flags (or the
  `VESSLCTL_ORG` / `VESSLCTL_TEAM` env vars). All `vesslctl` invocations in
  this recipe — `volume create`, `prep.sh`, `submit.sh` — pick up whichever
  org and team are currently active.
- An object volume to hold the data cache (~10 GB). Create one once:
  ```bash
  vesslctl volume create \
    --name autoresearch-cache \
    --storage <your-object-storage-slug> \
    --teams <your-team>
  vesslctl volume list  # grab the new volume's slug
  export AUTORESEARCH_CACHE_VOLUME=objvol-...
  ```
- A coding agent that can run shell commands locally (Claude Code, Codex,
  Cursor, etc.).

Find your object storage slug with `vesslctl storage list`. Find resource
specs and clusters with `vesslctl resource-spec list` and
`vesslctl cluster list`.

## Two ways to run

This recipe is fundamentally Path B (batch jobs) — there is no notebook
walk-through because the workflow is "agent runs in a loop overnight." The
two ways below differ in *who* drives the loop.

### Path A — Drive the agent yourself (one-off)

Useful for sanity-checking the setup before turning the agent loose
overnight.

```bash
# 1. One-time data prep (downloads ~10 GB into AUTORESEARCH_CACHE_VOLUME).
bash batch-job/prep.sh

# 2. Cut a branch, run the baseline once.
git checkout -b autoresearch/sanity-check
bash batch-job/submit.sh > run.log 2>&1
grep "^val_bpb:\|^peak_vram_mb:" run.log
```

If `val_bpb` shows up, the recipe is wired correctly.

### Path B — Hand it to the agent (overnight loop)

```bash
# In your coding agent (Claude Code, Codex, etc.), with this directory open:
> Have a look at program.md and let's kick off a new experiment.
> Let's do the setup first.
```

The agent reads `program.md`, walks the setup checklist (cuts a branch,
verifies the cache volume), then enters the experiment loop. Every iteration
is a `bash batch-job/submit.sh` call. You wake up to a populated
`results.tsv` and an `autoresearch/<tag>` branch on `vessl-cloud-cookbook`
with one commit per kept experiment.

## How submit.sh works

```
agent edits train.py → git commit
                     ↓
              bash batch-job/submit.sh
                     ↓
   git push origin autoresearch/<tag>   (force-with-lease)
                     ↓
   vesslctl job create --object-volume CACHE:/root/.cache/autoresearch
                     ↓
   container: clone cookbook @ branch, uv sync, uv run train.py
                     ↓
   vesslctl job logs -f → streamed to stdout → captured to local run.log
                     ↓
   exit 0 if job succeeded, non-zero otherwise
```

The cache volume holds `~/.cache/autoresearch` between jobs, so `prepare.py`
runs only once (in `prep.sh`) and every subsequent job skips it.

## Configuration

`submit.sh` and `prep.sh` read these env vars:

| Var | Required | Default |
|---|---|---|
| `AUTORESEARCH_CACHE_VOLUME` | yes | — |
| `AUTORESEARCH_RESOURCE_SPEC` | no | `resourcespec-a100x1` (A100 SXM × 1, betelgeuse-na) |
| `AUTORESEARCH_IMAGE` | no | `pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel` |
| `AUTORESEARCH_REPO_URL` | no | `https://github.com/vessl-ai/vessl-cloud-cookbook.git` |
| `AUTORESEARCH_TIMEOUT_S` | no | `1200` (submit.sh only) |

For an H100 baseline that matches karpathy's numbers, set
`AUTORESEARCH_RESOURCE_SPEC=resourcespec-5qp3iq5lcd90` (deneb-kr H100x1).

## Known limitations

- **Per-experiment overhead is real.** Each VESSL job pays ~3-4 min of
  startup (image pull, `uv sync` / torch reinstall, `train.py` compile) on
  top of the 5-min training budget. Measured throughput is ~6.8
  experiments/hour vs. ~12/hour on a dedicated local GPU. Over an 8-hour
  overnight run, that's ~50 completed experiments.
- **A100 ≠ H100.** This recipe defaults to A100 SXM ×1 because that's the
  high-availability single-GPU spec on VESSL Cloud. `train.py` falls back to
  `kernels-community/flash-attn3` on non-Hopper GPUs (the upstream code
  already handles this) so it runs, but **val_bpb numbers are not
  comparable** to H100 runs — neither karpathy's reference nor other
  autoresearch forks. Switch the spec env var to a Hopper resource for
  apples-to-apples comparison.
- **Torch wheel mismatch.** `pyproject.toml` pins `torch==2.9.1` from the
  cu128 wheel index; the default container ships torch 2.4.1 + CUDA 12.4.
  `uv sync` reinstalls torch in-container, and the cu128 wheels bundle their
  own CUDA so this works on any modern driver — but it adds ~30 s to every
  job. If you want to skip it, build a custom image with the project venv
  baked in.
- **Branch hygiene.** The agent runs entirely on a `autoresearch/<tag>`
  branch and force-pushes to origin. Do not run two agents on the same tag
  concurrently — the second one will clobber the first's commits.
- **Cost is unbounded by default.** A runaway loop = real spend. Set a
  daily-cap routine on `vesslctl billing show` if you're nervous.

## Further reading

- Upstream repo and intent: <https://github.com/karpathy/autoresearch>
- Karpathy's announcement tweet: <https://x.com/karpathy/status/2029701092347630069>
- "Dummy's Guide" to autoresearch: <https://x.com/hooeem/status/2030720614752039185>
- VESSL Cloud docs: <https://docs.vessl.ai>
