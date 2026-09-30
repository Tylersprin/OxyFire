#!/bin/bash
# Builds the merged YOLO dataset in $SCRATCH/oxyfire_data/outputs/merged/
# (symlinked images + new labels + data.yaml). Needs manifest.slurm and
# quality.slurm outputs first. Re-run with FORCE=1 to rebuild.
#
# Submit from the repo root:  sbatch jobs/merge.slurm
#SBATCH --job-name=oxy_merge
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out   # relative to the submit dir (repo root)

FILE_BUDGET="${FILE_BUDGET:-250000}"
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source jobs/env.sh || exit 1
set -euo pipefail

# Each image costs 2 files (symlink + label). Stop early rather than hit the quota.
images=$(( $(wc -l < "$OXY_OUT/manifest.csv") - 1 ))
used=$(showquota 2>/dev/null | awk -v s="$SCRATCH" '$1==s {print $4}')
echo "Scratch files in use: ${used:-unknown}. Upper bound to add: $(( 2 * images )). Budget: $FILE_BUDGET"
if [[ -n "$used" ]] && (( used + 2 * images > FILE_BUDGET )); then
    echo "STOP: the merged dataset could exceed the file budget."
    exit 2
fi

echo "Started $(date) on $(hostname)"
python -u scripts/build_merged.py ${FORCE:+--force}
showquota
echo "Finished $(date)"
