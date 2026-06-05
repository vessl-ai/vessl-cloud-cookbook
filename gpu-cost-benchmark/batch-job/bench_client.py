"""
Multi-instance round-robin chat-completions benchmark client for the GPU cost benchmark.

Drives an OpenAI-compatible /v1/chat/completions endpoint on N local vLLM
instances (one per GPU, ports BASE_PORT .. BASE_PORT+N-1) at a fixed
per-instance concurrency, then reports throughput and latency percentiles
(TTFT / TPOT), per-instance throughput skew, and speculative-decoding
(MTP) accept-rate metrics scraped from each instance's /metrics endpoint.

It runs a warmup window followed by a timed measurement window, replaying
prompts round-robin from a JSON subset file. Results are written to
`<outdir>/<tag>.json` and echoed to stdout.

Args:
  --model        Model name sent in each request. For a hot-loaded LoRA
                 adapter use the vLLM alias (e.g. "sft-lora"); otherwise the
                 served base model id. Required.
  --concurrency  In-flight requests PER instance. Total node concurrency =
                 concurrency * N. Required.
  --tag          Output tag; result file is <outdir>/<tag>.json. Required.
  --outdir       Output directory. Required.
  --subset       JSON file: list of {"prompt", "expected_output_len"} objects.
                 Default: /shared/datasets/bench_subset.json
  --warmup-s     Warmup seconds (default 30).
  --measure-s    Measurement seconds (default 60).

Environment variables:
  BASE_PORT      First instance port (default 8000).
  NUM_INSTANCES  Number of instances / ports to fan out across (default 8).

Expected infra:
  * N vLLM instances already serving locally on 127.0.0.1:BASE_PORT.. (see
    serve_and_bench.sh, which launches them and then calls this client).
  * Runs on the same 8-GPU node as the servers; no external network needed.

Copyright 2026 VESSL AI Inc. Licensed under Apache-2.0.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

import aiohttp

_BASE_PORT = int(os.environ.get("BASE_PORT", "8000"))
_NUM_INSTANCES = int(os.environ.get("NUM_INSTANCES", "8"))
PORTS = [_BASE_PORT + i for i in range(_NUM_INSTANCES)]
DEFAULT_WARMUP_S = 30
DEFAULT_MEASURE_S = 60  # inference cells are kept short on purpose
DEFAULT_SUBSET = os.environ.get("BENCH_SUBSET", "/shared/datasets/bench_subset.json")

SPEC_METRICS = (
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
)


async def fetch_metrics(session, port: int) -> dict:
    try:
        async with session.get(f"http://127.0.0.1:{port}/metrics",
                               timeout=aiohttp.ClientTimeout(total=5)) as r:
            text = await r.text()
    except Exception:
        return {}
    out = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        try:
            head, val = line.rsplit(" ", 1)
            name = head.split("{", 1)[0]
            if name in SPEC_METRICS:
                out[name] = out.get(name, 0.0) + float(val)
        except Exception:
            continue
    return out


async def one_request(session, port: int, model: str, prompt: str, max_tokens: int):
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    ttft = None
    completion_tokens = 0
    last_token_t = None
    inter_token_latencies = []
    try:
        async with session.post(url, json=body, timeout=aiohttp.ClientTimeout(total=300)) as resp:
            if resp.status != 200:
                txt = await resp.text()
                return {"ok": False, "err": f"HTTP {resp.status}: {txt[:120]}"}
            async for raw in resp.content:
                line = raw.decode(errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except Exception:
                    continue
                if obj.get("usage") is not None:
                    completion_tokens = obj["usage"].get("completion_tokens", completion_tokens)
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if content is None:
                    continue
                now = time.perf_counter()
                if ttft is None:
                    ttft = now - t0
                    last_token_t = now
                else:
                    inter_token_latencies.append(now - last_token_t)
                    last_token_t = now
    except Exception as e:
        return {"ok": False, "err": f"{type(e).__name__}: {e}"}
    total = time.perf_counter() - t0
    if ttft is None:
        return {"ok": False, "err": "no tokens received"}
    return {
        "ok": True,
        "ttft": ttft,
        "tpot_mean": statistics.mean(inter_token_latencies) if inter_token_latencies else 0.0,
        "total": total,
        "completion_tokens": completion_tokens or (1 + len(inter_token_latencies)),
    }


async def worker(session, port, slot_id, model, prompts_iter, results, stop_at):
    while True:
        if time.monotonic() >= stop_at:
            return
        try:
            req = next(prompts_iter)
        except StopIteration:
            return
        prompt, expected_out = req["prompt"], req["expected_output_len"]
        max_tok = min(expected_out + 8, 512)
        r = await one_request(session, port, model, prompt, max_tok)
        r["port"] = port
        r["slot"] = slot_id
        r["t_complete"] = time.monotonic()
        results.append(r)


async def run_bench(model, concurrency, warmup_s, measure_s, subset_path, tag, outdir):
    with open(subset_path) as f:
        subset = json.load(f)

    def gen():
        idx = 0
        while True:
            yield subset[idx % len(subset)]
            idx += 1
    prompts_iter = gen()

    connector = aiohttp.TCPConnector(limit=concurrency * len(PORTS) * 2, ttl_dns_cache=300)
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=10)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        metrics0 = await asyncio.gather(*[fetch_metrics(session, p) for p in PORTS])
        warmup_results, measure_results = [], []
        warmup_end = time.monotonic() + warmup_s
        measure_end = warmup_end + measure_s

        async def warmup_then_measure():
            tasks = []
            for slot in range(concurrency):
                for p in PORTS:
                    tasks.append(worker(session, p, slot, model, prompts_iter, warmup_results, warmup_end))
            await asyncio.gather(*tasks, return_exceptions=True)
            tasks = []
            for slot in range(concurrency):
                for p in PORTS:
                    tasks.append(worker(session, p, slot, model, prompts_iter, measure_results, measure_end))
            await asyncio.gather(*tasks, return_exceptions=True)

        t_start = time.monotonic()
        await warmup_then_measure()
        wall = time.monotonic() - t_start
        metrics1 = await asyncio.gather(*[fetch_metrics(session, p) for p in PORTS])

    ok = [r for r in measure_results if r.get("ok")]
    errs = [r for r in measure_results if not r.get("ok")]
    ttfts = sorted(r["ttft"] for r in ok)
    tpots = sorted(r["tpot_mean"] for r in ok if r["tpot_mean"] > 0)
    n = len(ok)
    out_toks = sum(r["completion_tokens"] for r in ok)
    total_attempted = n + len(errs)

    def pct(arr, q):
        if not arr:
            return None
        i = max(0, min(len(arr) - 1, int(round((q / 100) * (len(arr) - 1)))))
        return arr[i]

    # Per-port throughput breakdown
    port_tok = {p: 0 for p in PORTS}
    port_req = {p: 0 for p in PORTS}
    for r in ok:
        port_tok[r["port"]] = port_tok.get(r["port"], 0) + r["completion_tokens"]
        port_req[r["port"]] = port_req.get(r["port"], 0) + 1
    port_tps = {p: port_tok[p] / float(measure_s) for p in PORTS}
    port_tps_values = list(port_tps.values())
    import statistics as st
    port_tps_mean = st.mean(port_tps_values) if port_tps_values else 0
    port_tps_std = st.stdev(port_tps_values) if len(port_tps_values) > 1 else 0
    port_tps_cv_pct = (port_tps_std / port_tps_mean * 100) if port_tps_mean > 0 else 0

    accept_num = sum((m1.get("vllm:spec_decode_num_accepted_tokens_total", 0) -
                      m0.get("vllm:spec_decode_num_accepted_tokens_total", 0))
                     for m0, m1 in zip(metrics0, metrics1))
    accept_den = sum((m1.get("vllm:spec_decode_num_draft_tokens_total", 0) -
                      m0.get("vllm:spec_decode_num_draft_tokens_total", 0))
                     for m0, m1 in zip(metrics0, metrics1))
    draft_total = sum((m1.get("vllm:spec_decode_num_drafts_total", 0) -
                       m0.get("vllm:spec_decode_num_drafts_total", 0))
                      for m0, m1 in zip(metrics0, metrics1))

    summary = {
        "tag": tag,
        "model_in_request": model,
        "concurrency_per_instance": concurrency,
        "node_concurrency": concurrency * len(PORTS),
        "num_instances": len(PORTS),
        "warmup_s": warmup_s,
        "measure_s": measure_s,
        "measure_wall_s": wall - warmup_s,
        "num_warmup_requests": len(warmup_results),
        "num_measure_ok": n,
        "num_measure_err": len(errs),
        "error_rate_pct": (len(errs) / total_attempted * 100) if total_attempted else 0,
        # Throughput
        "throughput_req_per_s": n / float(measure_s) if n else 0,
        "throughput_out_tok_per_s": out_toks / float(measure_s) if n else 0,
        # Latency percentiles (TTFT)
        "ttft_p10_s": pct(ttfts, 10),
        "ttft_p25_s": pct(ttfts, 25),
        "ttft_p50_s": pct(ttfts, 50),
        "ttft_p75_s": pct(ttfts, 75),
        "ttft_p90_s": pct(ttfts, 90),
        "ttft_p95_s": pct(ttfts, 95),
        "ttft_p99_s": pct(ttfts, 99),
        # Latency percentiles (TPOT)
        "tpot_p10_s": pct(tpots, 10),
        "tpot_p25_s": pct(tpots, 25),
        "tpot_p50_s": pct(tpots, 50),
        "tpot_p75_s": pct(tpots, 75),
        "tpot_p90_s": pct(tpots, 90),
        "tpot_p95_s": pct(tpots, 95),
        "tpot_p99_s": pct(tpots, 99),
        # Per-instance throughput skew
        "per_port_tps_mean": port_tps_mean,
        "per_port_tps_std": port_tps_std,
        "per_port_tps_cv_pct": port_tps_cv_pct,
        "per_port_tps_min": min(port_tps_values) if port_tps_values else 0,
        "per_port_tps_max": max(port_tps_values) if port_tps_values else 0,
        "per_port_tps_dict": port_tps,
        # MTP (speculative decoding) accept-rate metrics
        "mtp_accept_rate": (accept_num / accept_den) if accept_den > 0 else None,
        "mtp_accepted_tokens": accept_num,
        "mtp_draft_tokens": accept_den,
        "mtp_num_drafts": draft_total,
        "mtp_draft_per_accept": (accept_den / accept_num) if accept_num > 0 else None,
        "mtp_tokens_per_draft_call": (accept_den / draft_total) if draft_total > 0 else None,
        "errors_sample": errs[:3],
    }
    outdir.mkdir(parents=True, exist_ok=True)
    out_json = outdir / f"{tag}.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "errors_sample"}, indent=2))
    if errs:
        print(f"  errors_sample[:3]: {summary['errors_sample']}", file=sys.stderr)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help='model name for request (e.g. "sft-lora" or the served base id)')
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--warmup-s", type=int, default=DEFAULT_WARMUP_S)
    ap.add_argument("--measure-s", type=int, default=DEFAULT_MEASURE_S)
    ap.add_argument("--subset", default=DEFAULT_SUBSET,
                    help="JSON list of {prompt, expected_output_len}")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    asyncio.run(run_bench(args.model, args.concurrency, args.warmup_s, args.measure_s,
                          args.subset, args.tag, Path(args.outdir)))


if __name__ == "__main__":
    main()
