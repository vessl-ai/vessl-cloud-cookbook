# Dataset — ShareGPT subset (benchmark workload)

**Bundled file:** none — the workload is built at runtime, not redistributed here.
**Source:** [`anon8231489123/ShareGPT_Vicuna_unfiltered`](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) on Hugging Face (public mirror of user-shared ChatGPT conversations).
**Use in this recipe:** a fixed-size subset, greedy-packed to `seq_len=2048`, used purely as a **representative chat workload to measure training/inference throughput**. It is not a curated training set and the recipe does not ship a fine-tuned model.
**Language:** multilingual, predominantly English.
**Last updated:** 2026-05.

## Why no bundled data file

This recipe measures *cost and throughput*, not model quality, so it does not bundle or redistribute the conversations. `data_prep/pack_sharegpt.py` downloads the public ShareGPT dataset from Hugging Face and packs it into a fixed-length token dataset at run time. Inference uses a 1,000-prompt subset of the same data.

## Format (after packing)

`pack_sharegpt.py` renders each conversation through the model chat template, tokenizes, and greedy-packs into fixed-length sequences saved as a Hugging Face Arrow dataset:

```text
{ "input_ids": [...2048 ints...], "labels": [...], "attention_mask": [...] }
```

## License & redistribution

The upstream ShareGPT data consists of user-submitted ChatGPT conversations; its licensing is **ambiguous and it is not ours to relicense**. We therefore do **not** redistribute it — you pull it from the public Hugging Face mirror yourself, subject to that source's terms and OpenAI's usage policies. Treat it as **benchmark-only**.

## Known limitations & PII

- **Possible PII.** User-shared conversations can contain personal information. Because nothing is bundled here, this recipe ships no PII; if you persist the packed dataset, scan it before sharing.
- **Not for shipping a model.** This is a throughput workload, not a quality-curated SFT set. Adapters trained on it are for benchmarking only.

## Attribution

Cite the Hugging Face source above and the original ShareGPT collection. This recipe's own code and docs are Apache-2.0 / CC-BY-4.0 per the repository [README](../../README.md#license); that license covers the recipe, **not** the third-party ShareGPT data.
