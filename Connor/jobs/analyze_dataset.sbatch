#!/usr/bin/env bash
# CPU-only Grace job for YOLO label analysis.
#SBATCH --job-name=dataset_analysis
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: sbatch jobs/analyze_dataset.sbatch DATASET_ROOT CONFIG_PATH OUTPUT_DIR" >&2
  exit 2
fi

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ -f "$SUBMIT_DIR/scripts/analyze_and_balance.py" ]]; then
  REPO_ROOT="$SUBMIT_DIR"
elif [[ -f "$SUBMIT_DIR/../scripts/analyze_and_balance.py" ]]; then
  REPO_ROOT="$(cd -- "$SUBMIT_DIR/.." && pwd)"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
fi
DATASET_ROOT="$1"
CONFIG_PATH="$2"
OUTPUT_DIR="$3"

# Grace has an old system Python on the login path. Set PYTHON_MODULE when
# submitting, for example: Python/3.11.5.
PYTHON_MODULE="${PYTHON_MODULE:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -n "$PYTHON_MODULE" ]]; then
  if ! type module >/dev/null 2>&1; then
    echo "ERROR: the Grace module command is unavailable; cannot load $PYTHON_MODULE" >&2
    exit 2
  fi
  module load "$PYTHON_MODULE"
fi
if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'; then
  echo "ERROR: Python 3.11+ is required. Set PYTHON_MODULE before sbatch." >&2
  exit 2
fi

# Permit config paths such as configs/example_config.toml when sbatch is
# submitted from outside the cloned repository.
if [[ "$CONFIG_PATH" != /* && ! -f "$CONFIG_PATH" && -f "$REPO_ROOT/$CONFIG_PATH" ]]; then
  CONFIG_PATH="$REPO_ROOT/$CONFIG_PATH"
fi

echo "SLURM job ID: ${SLURM_JOB_ID:-not-under-slurm}"
echo "Hostname: $(hostname)"
echo "Python version: $("$PYTHON_BIN" --version 2>&1)"
echo "Dataset path: $DATASET_ROOT"
echo "Config path: $CONFIG_PATH"
echo "Output path: $OUTPUT_DIR"

"$PYTHON_BIN" "$REPO_ROOT/scripts/analyze_and_balance.py" \
  --dataset-root "$DATASET_ROOT" \
  --config "$CONFIG_PATH" \
  --output-dir "$OUTPUT_DIR"
