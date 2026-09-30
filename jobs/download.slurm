#!/bin/bash
# Downloads the public fire/smoke datasets into $SCRATCH/oxyfire_data/raw/<dataset>/.
# Counts files inside each archive BEFORE extracting and stops if the total
# would exceed FILE_BUDGET. Safe to re-run: finished steps are skipped.
#
# Submit from the repo root:  sbatch jobs/download.slurm
#SBATCH --job-name=oxy_download
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=logs/%x_%j.out   # relative to the submit dir (repo root)

FILE_BUDGET="${FILE_BUDGET:-250000}"   # $SCRATCH file quota per showquota

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source jobs/env.sh || exit 1   # before set -u: module scripts reference unset variables
set -euo pipefail

ARCHIVES="$OXY_RAW/_archives"
SOURCES="$OXY_OUT/sources.csv"
mkdir -p "$ARCHIVES" "$OXY_OUT"
[[ -f "$SOURCES" ]] || echo "dataset,source,license,archive,bytes,sha256,files_in_archive,downloaded_at" > "$SOURCES"

# Kaggle's new-style token. Read into the environment only; never printed.
export KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)"

# name | type | source | license
DATASETS=(
    "dfire|kaggle|sayedgamal99/smoke-fire-detection-yolo|CC0 (mirror of gaia-solutions-on-demand/DFireDataset)"
    "aiformankind|gdrive_folder|https://drive.google.com/drive/folders/1IKXN2-hxTrEQsIIKOxiUAuLgoxubA9Wq|CC BY-NC-SA 4.0"
    "azimjaan21|kaggle|azimjaan21/fire-and-smoke-dataset-object-detection-yolo|CDLA-Permissive-1.0"
)

scratch_files() {
    showquota 2>/dev/null | awk -v s="$SCRATCH" '$1==s {print $4}'
}

count_entries() {  # number of files (not folders) inside an archive
    case "$1" in
        *.zip)            unzip -Z1 "$1" | grep -vc '/$' ;;
        *.tar.gz|*.tgz)   tar -tzf "$1" | grep -vc '/$' ;;
        *.tar)            tar -tf  "$1" | grep -vc '/$' ;;
    esac
}

extract() {
    case "$1" in
        *.zip)            unzip -q "$1" -d "$2" ;;
        *.tar.gz|*.tgz)   tar -xzf "$1" -C "$2" ;;
        *.tar)            tar -xf  "$1" -C "$2" ;;
    esac
}

# ---- Step 1: download all archives (no extraction yet) ----
for entry in "${DATASETS[@]}"; do
    IFS='|' read -r name kind src license <<< "$entry"
    dest="$ARCHIVES/$name"
    if compgen -G "$dest/*.zip" >/dev/null || compgen -G "$dest/*.tar*" >/dev/null || compgen -G "$dest/*/*.tar*" >/dev/null; then
        echo "[$name] archive already downloaded, skipping"
        continue
    fi
    mkdir -p "$dest"
    echo "[$name] downloading from $src  ($(date +%T))"
    case "$kind" in
        kaggle)        kaggle datasets download "$src" -p "$dest" ;;
        gdrive_folder) gdown --folder "$src" -O "$dest" ;;
    esac
done

# ---- Step 2: count files inside each archive, check against the budget ----
declare -A ARCHIVE_OF COUNT_OF
total_new=0
for entry in "${DATASETS[@]}"; do
    IFS='|' read -r name kind src license <<< "$entry"
    archive=$(find "$ARCHIVES/$name" -type f \( -name '*.zip' -o -name '*.tar*' -o -name '*.tgz' \) | head -1)
    [[ -n "$archive" ]] || { echo "[$name] ERROR: no archive found"; exit 1; }
    n=$(count_entries "$archive")
    ARCHIVE_OF[$name]="$archive"; COUNT_OF[$name]="$n"
    echo "[$name] $(basename "$archive"): $(du -h "$archive" | cut -f1), $n files inside"
    [[ -d "$OXY_RAW/$name" ]] || total_new=$(( total_new + n ))
done

used=$(scratch_files); used=${used:-0}
echo "Scratch files in use: $used. To be extracted: $total_new. Budget: $FILE_BUDGET"
if (( used + total_new > FILE_BUDGET )); then
    echo "STOP: extracting would exceed the file budget. Nothing extracted."
    exit 2
fi

# ---- Step 3: extract one dataset at a time and record provenance ----
for entry in "${DATASETS[@]}"; do
    IFS='|' read -r name kind src license <<< "$entry"
    archive="${ARCHIVE_OF[$name]}"
    if [[ -d "$OXY_RAW/$name" ]]; then
        echo "[$name] already extracted, skipping"
        continue
    fi
    echo "[$name] extracting  ($(date +%T))"
    mkdir -p "$OXY_RAW/$name.partial"
    extract "$archive" "$OXY_RAW/$name.partial"
    mv "$OXY_RAW/$name.partial" "$OXY_RAW/$name"   # only named final once complete
    sha=$(sha256sum "$archive" | cut -d' ' -f1)
    bytes=$(stat -c %s "$archive")
    echo "$name,\"$src\",\"$license\",$(basename "$archive"),$bytes,$sha,${COUNT_OF[$name]},$(date -Iseconds)" >> "$SOURCES"
    echo "[$name] done. Scratch files now: $(scratch_files)"
done

# ---- Step 4: Phase 1 discovery summary (structure + counts, no file contents) ----
echo
echo "===== Folder structure (depth 3) with file counts ====="
for entry in "${DATASETS[@]}"; do
    IFS='|' read -r name _ <<< "$entry"
    echo "--- $name"
    find "$OXY_RAW/$name" -mindepth 1 -maxdepth 3 -type d | sort | while read -r d; do
        printf '  %-70s %8d files\n' "${d#$OXY_RAW/}" "$(find "$d" -type f | wc -l)"
    done
    echo "  file types:"; find "$OXY_RAW/$name" -type f | sed 's/.*\.//' | tr 'A-Z' 'a-z' | sort | uniq -c | sort -rn | head -8 | sed 's/^/   /'
done
echo
showquota
echo "Finished $(date)"
