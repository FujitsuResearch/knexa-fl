#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}

# Resolve release root (one directory up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="$RELEASE_ROOT"${PYTHONPATH:+":$PYTHONPATH"}

# Preflight: check for NumPy
if ! $PYTHON - <<'EOF'
try:
    import numpy
except Exception as e:
    raise SystemExit(1)
EOF
then
  echo "NumPy not found. Install dependencies:"
  echo "  python3 -m venv .venv && source .venv/bin/activate"
  echo "  pip install -r knexa-fl-release/requirements.txt"
  exit 1
fi

$PYTHON -m knexa_fl_release.simulate_cpm
