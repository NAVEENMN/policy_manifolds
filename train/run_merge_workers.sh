#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "$0")/policy-manifolds/scripts/run_merge_workers.sh" "$@"
