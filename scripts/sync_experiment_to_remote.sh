#!/usr/bin/env bash
set -euo pipefail

# Sync a specific experiment directory to remote server
# Usage:
#   REMOTE_HOST=ssh.zw1.paratera.com REMOTE_PORT=2222 REMOTE_USER=root REMOTE_DIR=/root/shared-nvme/BasicSR \
#     EXPERIMENT_NAME=n2n_spect229_singleview_patch64 \
#     bash scripts/sync_experiment_to_remote.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

REMOTE_HOST="${REMOTE_HOST:-}"
REMOTE_PORT="${REMOTE_PORT:-2222}"
REMOTE_USER="${REMOTE_USER:-root}"
REMOTE_DIR="${REMOTE_DIR:-/root/shared-nvme/BasicSR}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -z "${REMOTE_HOST}" ]]; then
  echo "ERROR: REMOTE_HOST is required (e.g. ssh.zw1.paratera.com)" >&2
  exit 2
fi

if [[ -z "${EXPERIMENT_NAME}" ]]; then
  echo "ERROR: EXPERIMENT_NAME is required (e.g. n2n_spect229_singleview_patch64)" >&2
  exit 2
fi

LOCAL_EXP_DIR="${PROJECT_ROOT}/experiments/${EXPERIMENT_NAME}"
if [[ ! -d "${LOCAL_EXP_DIR}" ]]; then
  echo "ERROR: Experiment directory not found: ${LOCAL_EXP_DIR}" >&2
  exit 2
fi

RSYNC_OPTS=(
  -a
  -v
  -h
  -z
  --partial
  --progress
  --stats
  --no-owner
  --no-group
  --no-perms
)

if [[ "${DRY_RUN}" == "1" ]]; then
  RSYNC_OPTS+=(--dry-run)
fi

SSH_CMD=(
  ssh
  -l "${REMOTE_USER}"
  -p "${REMOTE_PORT}"
  -o UpdateHostKeys=no
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
)

echo "Local : ${LOCAL_EXP_DIR}/"
echo "Remote: ${REMOTE_HOST}:${REMOTE_DIR}/experiments/${EXPERIMENT_NAME}/"
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Mode  : DRY_RUN=1 (no changes will be made)"
fi

rsync "${RSYNC_OPTS[@]}" -e "${SSH_CMD[*]}" \
  "${LOCAL_EXP_DIR}/" \
  "${REMOTE_HOST}:${REMOTE_DIR}/experiments/${EXPERIMENT_NAME}/"

echo ""
echo "✓ Sync completed!"






