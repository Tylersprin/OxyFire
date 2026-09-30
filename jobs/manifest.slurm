#!/bin/bash
# Builds the combined manifest (manifest.csv, boxes.csv, ...) for all datasets.
# Reads image headers and label files only; copies and changes nothing in raw/.
#
# Submit from the repo root:  sbatch jobs/manifest.slurm
#SBATCH --job-name=oxy_manifest
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out   # relative to the submit dir (repo root)

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source jobs/env.sh || exit 1
set -euo pipefail

echo "Started $(date) on $(hostname) with $SLURM_CPUS_PER_TASK cores"
python -u scripts/build_manifest.py --workers "$SLURM_CPUS_PER_TASK" --out "$OXY_OUT"
echo
wc -l "$OXY_OUT"/manifest.csv "$OXY_OUT"/boxes.csv
echo "Finished $(date)"
