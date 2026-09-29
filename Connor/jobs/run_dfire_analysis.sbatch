#!/usr/bin/env bash
#SBATCH --job-name=dfire_analysis
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --export=ALL
#SBATCH --output=dfire-analysis-%j.out
#SBATCH --error=dfire-analysis-%j.err

set -euo pipefail

if [[ $# -ne 0 && $# -ne 3 ]]; then
  echo "Usage: sbatch jobs/run_dfire_analysis.sbatch [DATASET_ROOT CONFIG_PATH OUTPUT_DIR]" >&2
  exit 2
fi

REPO_ROOT="${REPO_ROOT:-$HOME/dataset_pipeline/Connor}"
DATASET_ROOT="${1:-/scratch/user/$USER/DFire/dataset/data}"
CONFIG_PATH="${2:-$REPO_ROOT/configs/dfire.toml}"
OUTPUT_DIR="${3:-/scratch/user/$USER/DFire_Analysis}"
PROGRESS_EVERY="${PROGRESS_EVERY:-1000}"
echo "Starting D-Fire job ${SLURM_JOB_ID:-unknown} at $(date)"
echo "Loading Grace Python modules..."

if [[ "$CONFIG_PATH" != /* ]]; then
  CONFIG_PATH="$REPO_ROOT/$CONFIG_PATH"
fi

# Initialize the Grace module command if the batch shell did not inherit it.
if ! type module >/dev/null 2>&1; then
  for module_init in /etc/profile.d/lmod.sh /usr/share/lmod/lmod/init/bash; do
    if [[ -r "$module_init" ]]; then
      # shellcheck disable=SC1090
      source "$module_init"
      break
    fi
  done
fi
if ! type module >/dev/null 2>&1; then
  echo "ERROR: Grace module command is unavailable" >&2
  exit 2
fi

module purge
module load GCCcore/13.2.0
module load Python/3.11.5
hash -r
PYTHON_BIN="$(command -v python3)"

echo "============================================================"
echo "DFire DATASET ANALYSIS AND BALANCING"
echo "============================================================"
echo "SLURM job ID: ${SLURM_JOB_ID:-unknown}"
echo "Hostname: $(hostname)"
echo "Python: $($PYTHON_BIN --version 2>&1)"
echo "Repository: $REPO_ROOT"
echo "Dataset root: $DATASET_ROOT"
echo "Config: $CONFIG_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Progress interval: $PROGRESS_EVERY images"
echo "Started: $(date)"

if [[ ! -f "$REPO_ROOT/scripts/analyze_and_balance.py" ]]; then
  echo "ERROR: analyzer not found: $REPO_ROOT/scripts/analyze_and_balance.py" >&2
  exit 2
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: config not found: $CONFIG_PATH" >&2
  exit 2
fi
if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "ERROR: dataset root not found: $DATASET_ROOT" >&2
  exit 2
fi

echo "Training image count: $(find "$DATASET_ROOT/train/images" -type f 2>/dev/null | wc -l)"
echo "Training label count: $(find "$DATASET_ROOT/train/labels" -type f -name '*.txt' 2>/dev/null | wc -l)"
echo "Starting analyzer: $(date)"

set +e
"$PYTHON_BIN" -u "$REPO_ROOT/scripts/analyze_and_balance.py" \
  --dataset-root "$DATASET_ROOT" \
  --config "$CONFIG_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --progress-every "$PROGRESS_EVERY"
STATUS=$?
set -e

echo "Analyzer exit code: $STATUS"
if [[ "$STATUS" -eq 0 ]]; then
  cat "$OUTPUT_DIR/analysis_summary.txt"
fi
echo "Finished: $(date)"
echo "Generated files:"
find "$OUTPUT_DIR" -maxdepth 1 -type f -printf '  %f\n' 2>/dev/null | sort || true
echo "============================================================"
exit "$STATUS"
