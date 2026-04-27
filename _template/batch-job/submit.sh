#!/usr/bin/env bash
# submit.sh — reference vesslctl invocation for this recipe.
#
# Prereqs:
#   - vesslctl installed and authenticated (vesslctl version tested: TODO).
#   - An Object volume in your org to mount at /shared.
#
# Usage:
#   VESSL_OBJECT_VOLUME=<your-volume-name> ./submit.sh <args>
#
# Tested <YYYY-MM-DD> on <GPU>.

set -euo pipefail

VOLUME="${VESSL_OBJECT_VOLUME:?set VESSL_OBJECT_VOLUME to your object-volume name}"

# 1. Upload script and any data assets to the object volume (one-time per volume)
# vesslctl volume upload "$VOLUME" recipe.py --remote-prefix scripts/
# vesslctl volume upload "$VOLUME" ../data/<your-dataset> --remote-prefix datasets/

# 2. Submit the job
# vesslctl job create \
#   --name "<recipe-name>-${TAG:-default}" \
#   --resource-spec resourcespec-... \
#   --image <image:tag> \
#   --object-volume "${VOLUME}:/shared" \
#   --env TODO_ENV_1="value" \
#   --cmd "pip install -r /shared/scripts/requirements.txt && python -u /shared/scripts/recipe.py"

echo "TODO: fill in the vesslctl commands above before using submit.sh."
exit 1
