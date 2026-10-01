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
OUTPUT_CSV="${OUTPUT_CSV:-$SCRATCH/kaggle-environment-${SLURM_JOB_ID}.csv}"
VENV_DIR="${VENV_DIR:-$SCRATCH/venvs/openclip}"

mkdir -p "$PROJECT_DIR/logs" "$(dirname "$OUTPUT_CSV")" "$VENV_DIR" "$SCRATCH/huggingface-cache"
cd "$PROJECT_DIR"

# Select versions available on Grace with `module spider Python` and `module spider CUDA`.
module purge
module load Python/3.8.2 CUDA/11.0.2

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    python -m venv "$VENV_DIR"
    "$VENV_DIR/bin/python" -m pip install --upgrade pip
    # Install torch/torchvision using the wheel index that matches the loaded CUDA module.
    "$VENV_DIR/bin/pip" install torch torchvision
    "$VENV_DIR/bin/pip" install -r requirements.txt
fi

source "$VENV_DIR/bin/activate"
export HF_HOME="${HF_HOME:-$SCRATCH/huggingface-cache}"
export TRANSFORMERS_CACHE="$HF_HOME"

python EnvironmentDetection/run_openclip.py \
    --data_dir "$DATA_DIR" \
    --output "$OUTPUT_CSV" \
    --batch_size "${BATCH_SIZE:-64}" \
    --num_workers "${SLURM_CPUS_PER_TASK:-1}" \
    --device cuda