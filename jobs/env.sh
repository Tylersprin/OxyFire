#!/bin/bash
# Shared environment for OxyFire jobs. Usage: source jobs/env.sh
# Heavy packages (numpy, pandas, scipy, matplotlib, Pillow) come from Grace modules
# so the venv stays small in files. The venv only adds what the modules lack.

module purge
# WebProxy gives compute nodes internet access (sets http(s)_proxy); purge removes it.
module load WebProxy
module load GCC/13.2.0 Python/3.11.5 SciPy-bundle/2023.11 matplotlib/3.8.2 Pillow/10.2.0

export OXY_DATA="$SCRATCH/oxyfire_data"
export OXY_RAW="$OXY_DATA/raw"
export OXY_OUT="$OXY_DATA/outputs"
VENV="$OXY_DATA/venv"

# .ready is written only after installs succeed, so a failed setup is retried.
if [[ ! -f "$VENV/.ready" ]]; then
    echo "Setting up venv at $VENV"
    [[ -x "$VENV/bin/python" ]] || python -m venv --system-site-packages "$VENV"
    "$VENV/bin/pip" install --quiet --no-cache-dir -U pip \
        && "$VENV/bin/pip" install --quiet --no-cache-dir "kaggle>=1.7" gdown ImageHash \
        && touch "$VENV/.ready" \
        || { echo "ERROR: venv setup failed"; return 1; }
fi
source "$VENV/bin/activate"
