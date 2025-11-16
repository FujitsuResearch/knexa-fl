#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 /path/to/HumanEval.jsonl /path/to/MBPP.jsonl [OUT_DIR]"
  exit 1
fi

HE_PATH="$1"
MBPP_PATH="$2"
OUT_DIR="${3:-}"

PYTHON=${PYTHON:-python3}

# Resolve release root (one directory up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="$RELEASE_ROOT"${PYTHONPATH:+":$PYTHONPATH"}

CMD=("$PYTHON" -m knexa_fl_release.split_serializer --human-eval "$HE_PATH" --mbpp "$MBPP_PATH")
if [ -n "$OUT_DIR" ]; then
  CMD+=(--out "$OUT_DIR")
fi

"${CMD[@]}"

