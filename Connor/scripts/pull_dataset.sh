#!/usr/bin/env bash
# Download or clone a dataset into a destination directory on Grace.

set -euo pipefail

usage() {
  echo "Usage: bash scripts/pull_dataset.sh --type git|zip|tar|tar.gz --source SOURCE --destination DESTINATION [--force]" >&2
}

TYPE=""
SOURCE=""
DESTINATION=""
FORCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --type) TYPE="${2:-}"; shift 2 ;;
    --source) SOURCE="${2:-}"; shift 2 ;;
    --destination) DESTINATION="${2:-}"; shift 2 ;;
    --force) FORCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$TYPE" || -z "$SOURCE" || -z "$DESTINATION" ]]; then
  usage
  exit 2
fi
case "$TYPE" in
  git|zip|tar|tar.gz) ;;
  *) echo "Unsupported type: $TYPE" >&2; exit 2 ;;
esac

USER_NAME="${USER:-$(id -un)}"
echo "User: $USER_NAME"
echo "Source: $SOURCE"
echo "Destination: $DESTINATION"

if [[ -e "$DESTINATION" ]]; then
  if [[ "$FORCE" -ne 1 ]]; then
    echo "Destination already exists; use --force to replace it: $DESTINATION" >&2
    exit 1
  fi
  rm -rf -- "$DESTINATION"
fi
mkdir -p -- "$(dirname -- "$DESTINATION")"

if [[ "$TYPE" == "git" ]]; then
  if ! git clone "$SOURCE" "$DESTINATION"; then
    echo "Failed to clone Git source." >&2
    exit 1
  fi
else
  ARCHIVE="$(mktemp "${TMPDIR:-/tmp}/dataset.XXXXXX")"
  cleanup() { rm -f -- "$ARCHIVE"; }
  trap cleanup EXIT
  if [[ "$SOURCE" == http://* || "$SOURCE" == https://* ]]; then
    curl -fL --retry 2 "$SOURCE" -o "$ARCHIVE"
  else
    cp -- "$SOURCE" "$ARCHIVE"
  fi
  mkdir -p -- "$DESTINATION"
  case "$TYPE" in
    zip) unzip -q "$ARCHIVE" -d "$DESTINATION" ;;
    tar|tar.gz) tar -xf "$ARCHIVE" -C "$DESTINATION" ;;
  esac
fi

echo "Dataset pull completed successfully."
du -sh -- "$DESTINATION"
