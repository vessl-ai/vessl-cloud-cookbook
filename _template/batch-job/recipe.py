"""
<Recipe title> — vesslctl batch-job script.

<TODO: one-paragraph description. What does this script do, what
infrastructure does it expect (Object volume mount, GPU resource spec)?>

Usage: submit via `vesslctl job create`. See `submit.sh` in the same
directory for a reference invocation.

Environment variables:
  TODO_ENV_1   <Description. Required/optional? Default value?>
  TODO_ENV_2   <Description.>

Outputs (written to $OUTPUT_BASE/...):
  final/        <TODO: describe artefacts.>
  metrics.json  <TODO: structured run metrics.>

Tested: <GPU model> on <container image>, <YYYY-MM-DD>.

Copyright 2026 VESSL AI Inc. Licensed under Apache-2.0.
"""

import os
import sys

# TODO: implement the recipe pipeline here. The reference implementation
# in `../../gemma4-finetuning/batch-job/finetune_gemma4.py` follows this
# rough shape and is a good starting point:
#
#   1. Parse env vars and validate.
#   2. Set up the output directory.
#   3. Initialise a metrics dict (mode, run_tag, started_at, stages).
#   4. Stage: model load.
#   5. Stage: prepare adapters / training-time modifications.
#   6. Stage: dataset load + format.
#   7. Stage: train.
#   8. Stage: evaluate / inference samples.
#   9. Stage: save artefacts.
#   10. Persist metrics.json on every stage transition so partial
#       progress is visible if the job is killed mid-run.

if __name__ == "__main__":
    print("TODO: implement this recipe.", file=sys.stderr)
    sys.exit(1)
