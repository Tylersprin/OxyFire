#!/usr/bin/env bash
# CPU-only Grace job for YOLO label analysis.
#SBATCH --job-name=dataset_analysis
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: sbatch jobs/analyze_dataset.sbatch DATASET_ROOT CONFIG_PATH OUTPUT_DIR" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DATASET_ROOT="$1"
CONFIG_PATH="$2"
OUTPUT_DIR="$3"

# Permit config paths such as configs/example_config.toml when sbatch is
# submitted from outside the cloned repository.
if [[ "$CONFIG_PATH" != /* && ! -f "$CONFIG_PATH" && -f "$REPO_ROOT/$CONFIG_PATH" ]]; then
  CONFIG_PATH="$REPO_ROOT/$CONFIG_PATH"
fi

echo "SLURM job ID: ${SLURM_JOB_ID:-not-under-slurm}"
echo "Hostname: $(hostname)"
echo "Python version: $(python3 --version 2>&1)"
echo "Dataset path: $DATASET_ROOT"
echo "Config path: $CONFIG_PATH"
echo "Output path: $OUTPUT_DIR"

python3 "$REPO_ROOT/scripts/analyze_and_balance.py" \
  --dataset-root "$DATASET_ROOT" \
  --config "$CONFIG_PATH" \
  --output-dir "$OUTPUT_DIR"
