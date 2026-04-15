#!/usr/bin/env bash
set -euo pipefail

# Incrementally sync this BasicSR repo to a remote machine via rsync+ssh.
#
# By default, it EXCLUDES:
# - experiments/       (large training outputs)
# - results/, wandb/, tb_logger/, __pycache__/ (local artifacts)
# - .git/              (optional; keep excluded by default to reduce transfer)
#
# Usage examples:
#   REMOTE_HOST=ssh.zw1.paratera.com REMOTE_PORT=2222 REMOTE_USER=root REMOTE_DIR=/root/BasicSR \
#     bash scripts/sync_basicsr_to_remote.sh
#
# If your SSH "username" literally contains '@' (e.g. root@ackcs-00gjg17n), set REMOTE_USER to that:
#   REMOTE_USER='root@ackcs-00gjg17n' ...
#
# Dry-run first:
#   DRY_RUN=1 ... bash scripts/sync_basicsr_to_remote.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

REMOTE_HOST="${REMOTE_HOST:-}"
REMOTE_PORT="${REMOTE_PORT:-2222}"
REMOTE_USER="${REMOTE_USER:-root}"
REMOTE_DIR="${REMOTE_DIR:-/root/BasicSR}"
DRY_RUN="${DRY_RUN:-0}"
SYNC_DATASETS="${SYNC_DATASETS:-0}"
# DATASETS_MODE:
# - 0 / none: do not sync datasets/ (default)
# - needed: only sync the dataset files needed for current SPECT229 60-view training
# - all: sync all datasets/ (can be huge)
DATASETS_MODE="${DATASETS_MODE:-none}"
# If set to 1, mirror local -> remote by deleting remote files that no longer exist locally
# (only applies to non-excluded paths; excluded paths are untouched).
DELETE_EXTRA="${DELETE_EXTRA:-0}"

if [[ -z "${REMOTE_HOST}" ]]; then
  echo "ERROR: REMOTE_HOST is required (e.g. ssh.zw1.paratera.com)" >&2
  exit 2
fi

RSYNC_OPTS=(
  -a
  -v
  -h
  -z
  --partial
  # shared-nvme 这类挂载经常不允许 chown/chmod；禁用这些属性同步，避免 code 23
  --no-owner
  --no-group
  --no-perms
  --stats
  --info=stats2,progress2
  --exclude 'experiments/'
  --exclude 'results/'
  --exclude 'wandb/'
  --exclude 'tb_logger/'
  --exclude '__pycache__/'
  --exclude '.pytest_cache/'
  --exclude '.mypy_cache/'
  --exclude '.ruff_cache/'
  --exclude '.git/'
)

if [[ "${DRY_RUN}" == "1" ]]; then
  RSYNC_OPTS+=(--dry-run)
fi

if [[ "${DELETE_EXTRA}" == "1" ]]; then
  # Mirror mode: delete remote files that are not present locally
  RSYNC_OPTS+=(--delete --delete-delay)
fi

SSH_CMD=(
  ssh
  -l "${REMOTE_USER}"
  -p "${REMOTE_PORT}"
  -o UpdateHostKeys=no
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
)

echo "Local : ${PROJECT_ROOT}/"
echo "Remote: ${REMOTE_HOST}:${REMOTE_DIR}/ (via ssh -l '${REMOTE_USER}')"
if [[ "${DELETE_EXTRA}" == "1" ]]; then
  echo "Mode  : DELETE_EXTRA=1 (mirror: remote extra files will be deleted, excluded paths untouched)"
fi

# Backward-compat: SYNC_DATASETS=1 means DATASETS_MODE=all
if [[ "${SYNC_DATASETS}" == "1" && "${DATASETS_MODE}" == "none" ]]; then
  DATASETS_MODE="all"
fi

if [[ "${DATASETS_MODE}" == "none" || "${DATASETS_MODE}" == "0" ]]; then
  RSYNC_OPTS+=(--exclude 'datasets/')
  echo "Exclude: experiments/ datasets/ results/ wandb/ tb_logger/ __pycache__/ .git/ ..."
elif [[ "${DATASETS_MODE}" == "all" ]]; then
  echo "Exclude: experiments/ results/ wandb/ tb_logger/ __pycache__/ .git/ ... (datasets/ INCLUDED: ALL)"
elif [[ "${DATASETS_MODE}" == "needed" ]]; then
  # Include only the files needed for SPECT229 60-view training:
  # - projections: datasets/SPECT229/<patient>/*_Proj4Filter.dat  (uint16, 60x128x128)
  # - mu-map:      datasets/SPECT229/<patient>/*_PostAtten.dat   (float32, 128^3)
  # - aux cache:   datasets/SPECT229_umap_aux_cache_60/*views60*.npz
  # Note: include rules must come before exclude rules.
  RSYNC_OPTS+=(
    --include 'datasets/'
    --include 'datasets/SPECT229/'
    --include 'datasets/SPECT229/*/'
    --include 'datasets/SPECT229/*/*_Proj4Filter.dat'
    --include 'datasets/SPECT229/*/*_PostAtten.dat'
    --include 'datasets/SPECT229_umap_aux_cache_60/'
    --include 'datasets/SPECT229_umap_aux_cache_60/**'
    --exclude 'datasets/**'
  )
  echo "Exclude: experiments/ results/ wandb/ tb_logger/ __pycache__/ .git/ ... (datasets/ INCLUDED: needed subset)"
else
  echo "ERROR: Unsupported DATASETS_MODE='${DATASETS_MODE}'. Use none|needed|all." >&2
  exit 2
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Mode  : DRY_RUN=1 (no changes will be made)"
fi

rsync "${RSYNC_OPTS[@]}" -e "${SSH_CMD[*]}" "${PROJECT_ROOT}/" "${REMOTE_HOST}:${REMOTE_DIR}/"
