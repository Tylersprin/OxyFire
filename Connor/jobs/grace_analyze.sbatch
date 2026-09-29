#!/usr/bin/env bash
#SBATCH --job-name=dfire_analysis
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --export=ALL
#SBATCH --output=dfire-%j.out
#SBATCH --error=dfire-%j.err

# This script is intentionally self-contained. Submit it with:
#   sbatch jobs/grace_analyze.sbatch
# or, for a real dataset:
#   sbatch jobs/grace_analyze.sbatch DATASET_ROOT CONFIG_PATH OUTPUT_DIR

set -u

REPO_ROOT="${REPO_ROOT:-$HOME/dataset_pipeline/Connor}"
DEFAULT_DATASET="/scratch/user/${USER}/DFire_pipeline_test/sample_dataset"
DEFAULT_CONFIG="$REPO_ROOT/configs/example_config.toml"
DEFAULT_OUTPUT="/scratch/user/${USER}/DFire_pipeline_test/sample_output_job"

DATASET_ROOT="${1:-$DEFAULT_DATASET}"
CONFIG_PATH="${2:-$DEFAULT_CONFIG}"
OUTPUT_DIR="${3:-$DEFAULT_OUTPUT}"
PROGRESS_EVERY="${PROGRESS_EVERY:-1000}"

if [[ "$CONFIG_PATH" != /* ]]; then
  CONFIG_PATH="$REPO_ROOT/$CONFIG_PATH"
fi

echo "============================================================"
echo "DFire Grace analysis job"
echo "============================================================"
echo "SLURM job ID: ${SLURM_JOB_ID:-unknown}"
echo "Hostname: $(hostname)"
echo "Submit directory: ${SLURM_SUBMIT_DIR:-unknown}"
echo "Repository: $REPO_ROOT"
echo "Dataset: $DATASET_ROOT"
echo "Config: $CONFIG_PATH"
echo "Output: $OUTPUT_DIR"
echo "Progress interval: $PROGRESS_EVERY"
echo "Start time: $(date)"

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

# Batch shells do not always inherit the interactive module function.
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
echo "Python executable: $PYTHON_BIN"
echo "Python version: $($PYTHON_BIN --version 2>&1)"
if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'; then
  echo "ERROR: Python 3.11+ is required" >&2
  exit 2
fi

TRAIN_IMAGES="$DATASET_ROOT/train/images"
TRAIN_LABELS="$DATASET_ROOT/train/labels"
if [[ -d "$TRAIN_IMAGES" ]]; then
  IMAGE_COUNT="$(find "$TRAIN_IMAGES" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' -o -iname '*.webp' \) | wc -l)"
else
  IMAGE_COUNT=0
fi
if [[ -d "$TRAIN_LABELS" ]]; then
  LABEL_COUNT="$(find "$TRAIN_LABELS" -type f -iname '*.txt' | wc -l)"
else
  LABEL_COUNT=0
fi
echo "Training image files found: $IMAGE_COUNT"
echo "Training label files found: $LABEL_COUNT"
echo "Starting analyzer: $(date)"

set +e
"$PYTHON_BIN" "$REPO_ROOT/scripts/analyze_and_balance.py" \
  --dataset-root "$DATASET_ROOT" \
  --config "$CONFIG_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --progress-every "$PROGRESS_EVERY"
ANALYZER_STATUS=$?
set -e

echo "Analyzer exit code: $ANALYZER_STATUS"
echo "End time: $(date)"
if [[ -d "$OUTPUT_DIR" ]]; then
  echo "Generated outputs:"
  find "$OUTPUT_DIR" -maxdepth 1 -type f -printf '  %f\n' | sort
fi
exit "$ANALYZER_STATUS"
