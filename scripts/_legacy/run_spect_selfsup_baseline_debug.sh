#!/bin/bash
# 运行 baseline 配置的 debug 模式脚本
# 用法: ./scripts/run_spect_selfsup_baseline_debug.sh [n2n|n2v|n2b|fpn|all]

set -e

cd "$(dirname "$0")/.."
export PYTHONPATH="./:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

BASELINE_DIR="options/train/spect_selfsup/baseline"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}============================================================${NC}"
    echo ""
}

TOTAL_ITER=${TOTAL_ITER:-200}
VAL_FREQ=${VAL_FREQ:-50}

run_experiment() {
    local config=$1
    local name=$2

    if [ ! -f "$config" ]; then
        echo -e "${RED}Error: Config file not found: $config${NC}"
        return 1
    fi

    print_header "Running: $name (debug mode, ${TOTAL_ITER} iters)"
    echo -e "${YELLOW}Config: $config${NC}"
    echo -e "${YELLOW}GPU: $CUDA_VISIBLE_DEVICES${NC}"
    echo -e "${YELLOW}Total iterations: $TOTAL_ITER${NC}"
    echo -e "${YELLOW}Val frequency: $VAL_FREQ${NC}"
    echo ""

    # 使用 --debug 参数和 --force_yml 来覆盖参数
    python -m prime.train \
        -opt "$config" \
        --debug \
        --force_yml "name=debug_${name}" "train:total_iter=${TOTAL_ITER}" "val:val_freq=${VAL_FREQ}" "logger:print_freq=10" "logger:save_checkpoint_freq=${TOTAL_ITER}"

    echo ""
    echo -e "${GREEN}Completed: $name${NC}"
}

case "${1:-all}" in
    n2n)
        run_experiment "${BASELINE_DIR}/n2n_baseline.yml" "n2n_baseline"
        ;;
    n2v)
        run_experiment "${BASELINE_DIR}/n2v_baseline.yml" "n2v_baseline"
        ;;
    n2b)
        run_experiment "${BASELINE_DIR}/n2b_baseline.yml" "n2b_baseline"
        ;;
    fpn)
        run_experiment "${BASELINE_DIR}/n2n_fpn_fusion.yml" "n2n_fpn_fusion"
        ;;
    all)
        print_header "Running ALL baseline experiments in debug mode"
        echo "Order: n2n -> n2v -> n2b -> fpn"
        echo ""

        run_experiment "${BASELINE_DIR}/n2n_baseline.yml" "n2n_baseline"
        run_experiment "${BASELINE_DIR}/n2v_baseline.yml" "n2v_baseline"
        run_experiment "${BASELINE_DIR}/n2b_baseline.yml" "n2b_baseline"
        run_experiment "${BASELINE_DIR}/n2n_fpn_fusion.yml" "n2n_fpn_fusion"

        print_header "ALL experiments completed!"
        ;;
    *)
        echo "Usage: $0 [n2n|n2v|n2b|fpn|all]"
        echo ""
        echo "Options:"
        echo "  n2n  - Run N2N baseline (UNetRes)"
        echo "  n2v  - Run N2V baseline (DBSNl blind-spot network)"
        echo "  n2b  - Run N2B baseline (Neighbor2Neighbor)"
        echo "  fpn  - Run N2N FPN fusion (UNetFPNFusion)"
        echo "  all  - Run all experiments sequentially (default)"
        echo ""
        echo "Environment variables:"
        echo "  CUDA_VISIBLE_DEVICES  - GPU to use (default: 0)"
        echo "  TOTAL_ITER            - Total iterations (default: 200)"
        echo "  VAL_FREQ              - Validation frequency (default: 50)"
        echo ""
        echo "Example:"
        echo "  ./scripts/run_spect_selfsup_baseline_debug.sh n2n                    # 200 iters"
        echo "  TOTAL_ITER=500 ./scripts/run_spect_selfsup_baseline_debug.sh all     # 500 iters"
        exit 1
        ;;
esac
