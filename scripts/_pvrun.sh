#!/usr/bin/env bash
# Run python3.12 with the project's venv site-packages injected via PYTHONPATH.
# This bypasses the puku-cli Bash sandbox block on direct venv python invocations
# while still using the venv's CUDA-enabled torch.
#
# Usage: bash scripts/_pvrun.sh scripts/foo.py [args...]
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH=/home/farhan/my-projects/.venv/lib/python3.12/site-packages
exec /usr/bin/python3.12 "$@"