#!/usr/bin/env bash
set -euo pipefail

# Run the full "compare_two_models" pipeline sequentially for a list of patients.
# Intended to be launched inside tmux.
#
# Stages per patient:
#  1) Stage1: generate low-dose caches (thin/BM3D), infer projections for two models,
#     export projection .dat, render 8-col projection MP4.
#  2) Extra MP4 #1: 5x6 proj+recon MP4 (orig/ema/poi projections + their recons),
#     auto-runs missing poisson recon (others are reused if present).
#  3) Extra MP4 #2: OSEM iteration sweep MP4 (2x6) on 20s (orig vs ema_a).
# After all patients:
#  4) Update summary plot (existing patients under compare_two_models/).
#
# Requirements:
#  - Run from recipe repo root
#  - PYTHON_BIN points to your training environment python (default: python)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_ROOT="results/compare_two_models"

EXP_A="experiments/converged_tomo_3d_nview"
EXP_B="experiments/converged_tomo_2d_nview"

SEED=123
THINS="2,3,4,5"
DIFF_PCT="99.9"

BM3D_WORKERS=8
BM3D_SIGMA=1.0

OSEM_ITERS=10
OSEM_TIMEOUT=600
SWEEP_ITERS="5,10,15,20,25,30"

# MP4 params (keep turns=1 for speed; bump to 2 if you want 120 frames)
TURNS=1
FPS=10
CRF=18

mkdir -p "${ROOT}/${OUT_ROOT}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 PATIENT1 [PATIENT2 ...]"
  exit 2
fi

cd "${ROOT}"

for PATIENT in "$@"; do
  echo "================================================================================"
  echo "[PIPELINE] patient=${PATIENT}"
  echo "================================================================================"

  PDIR="${OUT_ROOT}/${PATIENT}"
  mkdir -p "${PDIR}"

  echo "[1/3] Stage1: projections + 8-col MP4 + export .dat"
  PYTHONPATH="./:${PYTHONPATH}" "${PYTHON_BIN}" scripts/compare_two_models_mp4.py \
    --patient "${PATIENT}" \
    --spect229-dir datasets/SPECT229 \
    --seed ${SEED} \
    --thin-factors ${THINS} \
    --allow-on-the-fly-cache \
    --bm3d-workers ${BM3D_WORKERS} \
    --bm3d-sigma ${BM3D_SIGMA} \
    --exp-a ${EXP_A} \
    --exp-b ${EXP_B} \
    --label-a "60view3d(ema)" \
    --label-b "60view2d(ema)" \
    --diff-percentile ${DIFF_PCT} \
    --turns ${TURNS} \
    --fps ${FPS} \
    --crf ${CRF} \
    --save-proj-dat-dir ${PDIR}/projections \
    --out ${PDIR}/two_models_8col_p${DIFF_PCT}.mp4 \
    > "${PDIR}/stage1.log" 2>&1

  echo "[2/3] MP4#1: 5x6 (3 proj + 3 recon), auto poisson recon"
  PYTHONPATH="./:${PYTHONPATH}" "${PYTHON_BIN}" scripts/compare_two_models_mp4.py \
    --patient "${PATIENT}" \
    --spect229-dir datasets/SPECT229 \
    --thin-factors ${THINS} \
    --load-proj-dat-dir ${PDIR}/projections \
    --label-a "60view3d(ema)" \
    --label-b "60view2d(ema)" \
    --turns ${TURNS} \
    --fps ${FPS} \
    --crf ${CRF} \
    --recon-dir ${PDIR}/reconstructions \
    --osem-iterations ${OSEM_ITERS} \
    --osem-timeout-sec ${OSEM_TIMEOUT} \
    --proj3recon-mp4-out ${PDIR}/proj3recon_5x6.mp4 \
    > "${PDIR}/proj3recon.log" 2>&1

  echo "[2.5/3] MP4(recon): 5x4 (orig / bm3d / modelB / modelA) reconstruction-domain MP4"
  PYTHONPATH="./:${PYTHONPATH}" "${PYTHON_BIN}" scripts/compare_two_models_mp4.py \
    --patient "${PATIENT}" \
    --spect229-dir datasets/SPECT229 \
    --thin-factors ${THINS} \
    --load-proj-dat-dir ${PDIR}/projections \
    --label-a "60view3d(ema)" \
    --label-b "60view2d(ema)" \
    --recon-dir ${PDIR}/reconstructions \
    --osem-iterations ${OSEM_ITERS} \
    --osem-timeout-sec ${OSEM_TIMEOUT} \
    --recon-turns ${TURNS} \
    --recon-fps ${FPS} \
    --recon-crf ${CRF} \
    --recon-mp4-out ${PDIR}/two_models_recon_4col.mp4 \
    > "${PDIR}/recon.log" 2>&1

  echo "[3/3] MP4#2: OSEM sweep 2x6 (orig vs ema_a) on 20s"
  PYTHONPATH="./:${PYTHONPATH}" "${PYTHON_BIN}" scripts/compare_two_models_mp4.py \
    --patient "${PATIENT}" \
    --spect229-dir datasets/SPECT229 \
    --load-proj-dat-dir ${PDIR}/projections \
    --label-a "60view3d(ema)" \
    --recon-turns ${TURNS} \
    --recon-fps ${FPS} \
    --recon-crf ${CRF} \
    --osem-timeout-sec ${OSEM_TIMEOUT} \
    --osem-sweep-iters ${SWEEP_ITERS} \
    --osem-sweep-mp4-out ${PDIR}/osem_sweep_2x6.mp4 \
    > "${PDIR}/osem_sweep.log" 2>&1

  echo "[DONE] ${PATIENT}"
done

echo "================================================================================"
echo "[SUMMARY] updating aggregated plots/csv"
echo "================================================================================"
PYTHONPATH="./:${PYTHONPATH}" "${PYTHON_BIN}" scripts/plot_existing_patients_proj_recon_summary.py \
  --root ${OUT_ROOT} \
  --osem-iters ${OSEM_ITERS} \
  --methods bm3d,ema_a,ema_b,poi \
  --doses 20s,x2,x3,x4,x5 \
  --out-png ${OUT_ROOT}/summary_proj_recon_change.png \
  --out-csv ${OUT_ROOT}/summary_proj_recon_change.csv \
  > "${OUT_ROOT}/summary.log" 2>&1

echo "[OK] pipeline finished."
