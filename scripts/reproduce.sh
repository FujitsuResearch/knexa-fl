#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}

# Resolve release root (one directory up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="$RELEASE_ROOT"${PYTHONPATH:+":$PYTHONPATH"}

$PYTHON -m knexa_fl_release.reproduce_paper --from-artifacts "$@"
