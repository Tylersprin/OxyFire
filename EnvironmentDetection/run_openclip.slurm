#!/bin/bash
#SBATCH --job-name=dfire-openclip
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=openclip-%j.out
#SBATCH --error=openclip-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$SCRATCH/OxyFireRepo}"
DATA_DIR="${DATA_DIR:-$SCRATCH/datasets/data/train/images}"
OUTPUT_CSV="${OUTPUT_CSV:-$SCRATCH/environment_data/kaggle-environment-${SLURM_JOB_ID}.csv}"
VENV_DIR="${VENV_DIR:-$SCRATCH/venvs/openclip}"

mkdir -p "$PROJECT_DIR/logs" "$(dirname "$OUTPUT_CSV")" "$VENV_DIR" "$SCRATCH/huggingface-cache"
cd "$PROJECT_DIR"

source "$VENV_DIR/bin/activate"
export HF_HOME="${HF_HOME:-$SCRATCH/huggingface-cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1

python EnvironmentDetection/run_openclip.py \
    --data_dir "$DATA_DIR" \
    --output "$OUTPUT_CSV" \
    --batch_size "${BATCH_SIZE:-64}" \
    --num_workers "${SLURM_CPUS_PER_TASK:-1}" \
    --device cuda