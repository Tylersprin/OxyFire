#!/usr/bin/env bash
#SBATCH --job-name=pull_dataset
#SBATCH --time=00:15:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1

set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: sbatch jobs/pull_dataset.sbatch TYPE SOURCE DESTINATION [--force]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
echo "SLURM job ID: ${SLURM_JOB_ID:-not-under-slurm}"
echo "Hostname: $(hostname)"
if [[ $# -eq 4 && "$4" != "--force" ]]; then
  echo "The optional fourth argument must be --force" >&2
  exit 2
fi
PULL_ARGS=(--type "$1" --source "$2" --destination "$3")
if [[ $# -eq 4 ]]; then
  PULL_ARGS+=(--force)
fi
bash "$REPO_ROOT/scripts/pull_dataset.sh" "${PULL_ARGS[@]}"
