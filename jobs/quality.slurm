#!/bin/bash
# Phase 3 data-quality stats: decodes every image once (brightness, pHash,
# corruption check), then writes tables, plots and summary.md to outputs/report/.
# Needs manifest.csv / boxes.csv from jobs/manifest.slurm. Safe to re-run: the
# per-image pass is cached in report/image_stats.csv.
#
# Submit from the repo root:  sbatch jobs/quality.slurm
#SBATCH --job-name=oxy_quality
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out   # relative to the submit dir (repo root)

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source jobs/env.sh || exit 1
set -euo pipefail

echo "Started $(date) on $(hostname) with $SLURM_CPUS_PER_TASK cores"
python -u scripts/quality_stats.py --workers "$SLURM_CPUS_PER_TASK"
echo "Finished $(date)"
