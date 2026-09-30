#!/bin/bash
# Color evidence + contact sheets for azimjaan21's unknown class ids.
# Needs manifest.csv / boxes.csv from jobs/manifest.slurm first.
#
# Submit from the repo root:  sbatch jobs/class_check.slurm
#SBATCH --job-name=oxy_class_check
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out   # relative to the submit dir (repo root)

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source jobs/env.sh || exit 1
set -euo pipefail

echo "Started $(date) on $(hostname)"
python -u scripts/class_check.py --workers "$SLURM_CPUS_PER_TASK"
echo "Finished $(date)"
