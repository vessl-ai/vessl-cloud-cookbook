# Benchmarks — <recipe-name>

Measured on <GPU model + memory>, <YYYY-MM-DD>.

## Summary

<TODO: one paragraph. What was run, headline numbers, the single most important finding.>

## Run statistics

| Metric | Value |
|--------|------:|
| Dataset | <name + sample count> |
| Container image | <image:tag> |
| Training runtime | <seconds> |
| Final loss | <number> |
| Peak VRAM | <GB> |
| Total wall time (script) | <seconds> |
| GPU | <model + memory> |

## Cost

| Run | Duration (script wall time) | Cost at $<rate>/hr |
|-----|----------------------------:|-------------------:|
| <run-name> | <minutes> | $<X> |

Image pull + dependency install accounts for ~<X> min of full job wall time (not included above).

Prices as of <YYYY-MM-DD>.

## Inference samples

<TODO: optional. Before/after qualitative comparison or held-out test results. Cite quoted outputs verbatim — translation invents content.>

## Known limitations

See `data/DATASET_CARD.md` for dataset-specific caveats.

<TODO: recipe-level caveats here. Be specific about what this recipe does NOT do.>
