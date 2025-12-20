#!/bin/bash
# ============================================================
# SPECT Self-Supervised Denoising 全部消融实验运行脚本
# ============================================================
# 用法:
#   ./scripts/run_all_experiments.sh                  # 运行所有实验
#   ./scripts/run_all_experiments.sh baseline         # 只运行 baseline
#   ./scripts/run_all_experiments.sh tier1            # 只运行 tier1
#   ./scripts/run_all_experiments.sh tier2            # 只运行 tier2 全部
#   ./scripts/run_all_experiments.sh tier2_norm       # 只运行 tier2 norm 消融
#   ./scripts/run_all_experiments.sh tier3            # 只运行 tier3
#   ./scripts/run_all_experiments.sh list             # 列出所有实验
#
# 环境变量:
#   CUDA_VISIBLE_DEVICES  - 指定 GPU (默认: 0)
#   DRY_RUN=1             - 只打印命令不执行
#   SKIP_EXISTING=1       - 跳过已有检查点的实验
# ============================================================

set -e

cd "$(dirname "$0")/.."
export PYTHONPATH="./:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# 配置目录
CONFIG_BASE="options/train/spect_selfsup"

# ============================================================
# 实验配置列表
# ============================================================

# Baseline (4个) - 验证代码修复
BASELINE_CONFIGS=(
    "baseline/n2n_baseline.yml"
    "baseline/n2v_baseline.yml"
    "baseline/n2b_baseline.yml"
    "baseline/n2n_restormer.yml"   # SOTA Transformer (~26M params, 需要16GB显存)
    # "baseline/n2n_fpn_fusion.yml"  # FPN EMA 有问题，暂时跳过
)

# Tier 1: 方法对比 (5个)
TIER1_CONFIGS=(
    "tier1_method_comparison/n2n_baseline.yml"
    "tier1_method_comparison/n2v_baseline.yml"
    "tier1_method_comparison/n2b_baseline.yml"
    "tier1_method_comparison/noiser2noise_baseline.yml"
    "tier1_method_comparison/s2s_baseline.yml"
)

# Tier 2: 消融实验
TIER2_NORM_CONFIGS=(
    "tier2_ablation/norm/n2n_linear_poisson.yml"
    "tier2_ablation/norm/n2n_linear_mse.yml"
    "tier2_ablation/norm/n2n_linear_l1.yml"
    "tier2_ablation/norm/n2n_linear_charb.yml"
    "tier2_ablation/norm/n2n_anscombe_charb.yml"
    "tier2_ablation/norm/n2n_anscombe_poisson.yml"
)

TIER2_DATA_CONFIGS=(
    "tier2_ablation/data_strategy/n2n_patch64_batch32.yml"  # baseline
    "tier2_ablation/data_strategy/n2n_patch64_batch16.yml"
    "tier2_ablation/data_strategy/n2n_patch64_batch8.yml"
    "tier2_ablation/data_strategy/n2n_patch128_batch16.yml"
    "tier2_ablation/data_strategy/n2n_patch256_batch8.yml"
    "tier2_ablation/data_strategy/n2n_patch32_batch64.yml"
    "tier2_ablation/data_strategy/n2n_patch64_batch16_aug.yml"
)

TIER2_REG_CONFIGS=(
    "tier2_ablation/regularization/n2n_wd_1e4.yml"  # baseline
    "tier2_ablation/regularization/n2n_wd_0.yml"
    "tier2_ablation/regularization/n2n_wd_1e5.yml"
    "tier2_ablation/regularization/n2n_wd_5e4.yml"
    "tier2_ablation/regularization/n2n_wd_1e3.yml"
)

TIER2_OPT_CONFIGS=(
    "tier2_ablation/optimization/n2n_adamw_1e4.yml"  # baseline
    "tier2_ablation/optimization/n2n_adam_1e4.yml"
    "tier2_ablation/optimization/n2n_adamw_5e5.yml"
    "tier2_ablation/optimization/n2n_adamw_5e4.yml"
    "tier2_ablation/optimization/n2n_etamin_1e6.yml"
    "tier2_ablation/optimization/n2n_etamin_1e7.yml"
    "tier2_ablation/optimization/n2n_etamin_1e4.yml"
)

TIER2_NET_CONFIGS=(
    "tier2_ablation/network/n2n_unetres_nb4.yml"  # baseline
    "tier2_ablation/network/n2n_unetres_nb2.yml"
    "tier2_ablation/network/n2n_unetres_nb6.yml"
    "tier2_ablation/network/n2n_nc_32_64_128_256.yml"
    "tier2_ablation/network/n2n_nc_96_192_384_768.yml"
)

TIER2_VIEW_CONFIGS=(
    "tier2_ablation/view_mode/n2n_single_view_both.yml"  # baseline
    "tier2_ablation/view_mode/n2n_single_view_anterior.yml"
    "tier2_ablation/view_mode/n2n_single_view_posterior.yml"
    # "tier2_ablation/view_mode/n2n_late_fusion.yml"      # 架构有问题
    # "tier2_ablation/view_mode/n2n_attention_fusion.yml" # 架构有问题
)

TIER2_COMBINED_CONFIGS=(
    "tier2_ablation/combined/n2n_best_config.yml"
    "tier2_ablation/combined/n2n_original_kair.yml"
    "tier2_ablation/combined/n2n_worst_config.yml"
)

# Tier 3: 其他方法优化
TIER3_N2V_CONFIGS=(
    "tier3_other_methods/n2v/n2v_linear_poisson.yml"
    "tier3_other_methods/n2v/n2v_linear_charb.yml"
    "tier3_other_methods/n2v/n2v_anscombe_charb.yml"
    "tier3_other_methods/n2v/n2v_anscombe_poisson.yml"
)

TIER3_N2B_CONFIGS=(
    "tier3_other_methods/n2b/n2b_linear_lambda_1_1.yml"
    "tier3_other_methods/n2b/n2b_anscombe_lambda_1_1.yml"
    "tier3_other_methods/n2b/n2b_anscombe_lambda_1_2.yml"
    "tier3_other_methods/n2b/n2b_anscombe_lambda_2_1.yml"
)

TIER3_S2S_CONFIGS=(
    "tier3_other_methods/s2s/s2s_dropout_0.3_mc_100.yml"
    "tier3_other_methods/s2s/s2s_dropout_0.2_mc_100.yml"
    "tier3_other_methods/s2s/s2s_dropout_0.4_mc_100.yml"
    "tier3_other_methods/s2s/s2s_dropout_0.3_mc_50.yml"
)

# ============================================================
# 函数定义
# ============================================================

print_header() {
    echo ""
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}============================================================${NC}"
    echo ""
}

print_subheader() {
    echo ""
    echo -e "${CYAN}--- $1 ---${NC}"
    echo ""
}

get_exp_name() {
    local config=$1
    # 从配置文件提取实验名
    grep "^name:" "${CONFIG_BASE}/${config}" 2>/dev/null | head -1 | awk '{print $2}' || basename "$config" .yml
}

check_existing() {
    local exp_name=$1
    local state_file="experiments/${exp_name}/training_states/latest.state"
    if [ -f "$state_file" ]; then
        return 0  # 存在
    fi
    return 1  # 不存在
}

run_experiment() {
    local config=$1
    local full_config="${CONFIG_BASE}/${config}"

    if [ ! -f "$full_config" ]; then
        echo -e "${RED}Error: Config not found: $full_config${NC}"
        return 1
    fi

    local exp_name=$(get_exp_name "$config")

    # 检查是否跳过已存在的实验
    if [ "${SKIP_EXISTING:-0}" = "1" ]; then
        if check_existing "$exp_name"; then
            echo -e "${YELLOW}[SKIP] $exp_name - checkpoint exists${NC}"
            return 0
        fi
    fi

    echo -e "${GREEN}[RUN] $exp_name${NC}"
    echo -e "      Config: $config"

    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo -e "      ${YELLOW}(dry run)${NC}"
        return 0
    fi

    python basicsr/train.py -opt "$full_config" --auto_resume

    echo -e "${GREEN}[DONE] $exp_name${NC}"
    echo ""
}

run_config_list() {
    local name=$1
    shift
    local configs=("$@")

    print_subheader "$name (${#configs[@]} experiments)"

    for config in "${configs[@]}"; do
        run_experiment "$config"
    done
}

count_experiments() {
    local total=0
    total=$((total + ${#BASELINE_CONFIGS[@]}))
    total=$((total + ${#TIER1_CONFIGS[@]}))
    total=$((total + ${#TIER2_NORM_CONFIGS[@]}))
    total=$((total + ${#TIER2_DATA_CONFIGS[@]}))
    total=$((total + ${#TIER2_REG_CONFIGS[@]}))
    total=$((total + ${#TIER2_OPT_CONFIGS[@]}))
    total=$((total + ${#TIER2_NET_CONFIGS[@]}))
    total=$((total + ${#TIER2_VIEW_CONFIGS[@]}))
    total=$((total + ${#TIER2_COMBINED_CONFIGS[@]}))
    total=$((total + ${#TIER3_N2V_CONFIGS[@]}))
    total=$((total + ${#TIER3_N2B_CONFIGS[@]}))
    total=$((total + ${#TIER3_S2S_CONFIGS[@]}))
    echo $total
}

list_experiments() {
    echo ""
    echo "SPECT Self-Supervised Denoising 实验列表"
    echo "========================================="
    echo ""
    echo "Baseline (${#BASELINE_CONFIGS[@]}):"
    for c in "${BASELINE_CONFIGS[@]}"; do echo "  - $c"; done
    echo ""
    echo "Tier 1 - Method Comparison (${#TIER1_CONFIGS[@]}):"
    for c in "${TIER1_CONFIGS[@]}"; do echo "  - $c"; done
    echo ""
    echo "Tier 2 - Norm Ablation (${#TIER2_NORM_CONFIGS[@]}):"
    for c in "${TIER2_NORM_CONFIGS[@]}"; do echo "  - $c"; done
    echo ""
    echo "Tier 2 - Data Strategy (${#TIER2_DATA_CONFIGS[@]}):"
    for c in "${TIER2_DATA_CONFIGS[@]}"; do echo "  - $c"; done
    echo ""
    echo "Tier 2 - Regularization (${#TIER2_REG_CONFIGS[@]}):"
    for c in "${TIER2_REG_CONFIGS[@]}"; do echo "  - $c"; done
    echo ""
    echo "Tier 2 - Optimization (${#TIER2_OPT_CONFIGS[@]}):"
    for c in "${TIER2_OPT_CONFIGS[@]}"; do echo "  - $c"; done
    echo ""
    echo "Tier 2 - Network (${#TIER2_NET_CONFIGS[@]}):"
    for c in "${TIER2_NET_CONFIGS[@]}"; do echo "  - $c"; done
    echo ""
    echo "Tier 2 - View Mode (${#TIER2_VIEW_CONFIGS[@]}):"
    for c in "${TIER2_VIEW_CONFIGS[@]}"; do echo "  - $c"; done
    echo ""
    echo "Tier 2 - Combined (${#TIER2_COMBINED_CONFIGS[@]}):"
    for c in "${TIER2_COMBINED_CONFIGS[@]}"; do echo "  - $c"; done
    echo ""
    echo "Tier 3 - N2V Variants (${#TIER3_N2V_CONFIGS[@]}):"
    for c in "${TIER3_N2V_CONFIGS[@]}"; do echo "  - $c"; done
    echo ""
    echo "Tier 3 - N2B Variants (${#TIER3_N2B_CONFIGS[@]}):"
    for c in "${TIER3_N2B_CONFIGS[@]}"; do echo "  - $c"; done
    echo ""
    echo "Tier 3 - S2S Variants (${#TIER3_S2S_CONFIGS[@]}):"
    for c in "${TIER3_S2S_CONFIGS[@]}"; do echo "  - $c"; done
    echo ""
    echo "========================================="
    echo "Total: $(count_experiments) experiments"
    echo ""
}

# ============================================================
# 主程序
# ============================================================

case "${1:-all}" in
    list)
        list_experiments
        ;;
    baseline)
        print_header "Running Baseline Experiments"
        run_config_list "Baseline" "${BASELINE_CONFIGS[@]}"
        ;;
    tier1)
        print_header "Running Tier 1: Method Comparison"
        run_config_list "Tier 1" "${TIER1_CONFIGS[@]}"
        ;;
    tier2_norm)
        print_header "Running Tier 2: Norm Ablation"
        run_config_list "Norm" "${TIER2_NORM_CONFIGS[@]}"
        ;;
    tier2_data)
        print_header "Running Tier 2: Data Strategy Ablation"
        run_config_list "Data Strategy" "${TIER2_DATA_CONFIGS[@]}"
        ;;
    tier2_reg)
        print_header "Running Tier 2: Regularization Ablation"
        run_config_list "Regularization" "${TIER2_REG_CONFIGS[@]}"
        ;;
    tier2_opt)
        print_header "Running Tier 2: Optimization Ablation"
        run_config_list "Optimization" "${TIER2_OPT_CONFIGS[@]}"
        ;;
    tier2_net)
        print_header "Running Tier 2: Network Ablation"
        run_config_list "Network" "${TIER2_NET_CONFIGS[@]}"
        ;;
    tier2_view)
        print_header "Running Tier 2: View Mode Ablation"
        run_config_list "View Mode" "${TIER2_VIEW_CONFIGS[@]}"
        ;;
    tier2_combined)
        print_header "Running Tier 2: Combined Configs"
        run_config_list "Combined" "${TIER2_COMBINED_CONFIGS[@]}"
        ;;
    tier2)
        print_header "Running ALL Tier 2 Ablations"
        run_config_list "Norm" "${TIER2_NORM_CONFIGS[@]}"
        run_config_list "Data Strategy" "${TIER2_DATA_CONFIGS[@]}"
        run_config_list "Regularization" "${TIER2_REG_CONFIGS[@]}"
        run_config_list "Optimization" "${TIER2_OPT_CONFIGS[@]}"
        run_config_list "Network" "${TIER2_NET_CONFIGS[@]}"
        run_config_list "View Mode" "${TIER2_VIEW_CONFIGS[@]}"
        run_config_list "Combined" "${TIER2_COMBINED_CONFIGS[@]}"
        ;;
    tier3_n2v)
        print_header "Running Tier 3: N2V Variants"
        run_config_list "N2V" "${TIER3_N2V_CONFIGS[@]}"
        ;;
    tier3_n2b)
        print_header "Running Tier 3: N2B Variants"
        run_config_list "N2B" "${TIER3_N2B_CONFIGS[@]}"
        ;;
    tier3_s2s)
        print_header "Running Tier 3: S2S Variants"
        run_config_list "S2S" "${TIER3_S2S_CONFIGS[@]}"
        ;;
    tier3)
        print_header "Running ALL Tier 3 Experiments"
        run_config_list "N2V" "${TIER3_N2V_CONFIGS[@]}"
        run_config_list "N2B" "${TIER3_N2B_CONFIGS[@]}"
        run_config_list "S2S" "${TIER3_S2S_CONFIGS[@]}"
        ;;
    all)
        print_header "Running ALL Experiments ($(count_experiments) total)"
        echo -e "${YELLOW}GPU: $CUDA_VISIBLE_DEVICES${NC}"
        echo -e "${YELLOW}Auto Resume: Enabled${NC}"
        echo ""

        run_config_list "Baseline" "${BASELINE_CONFIGS[@]}"
        run_config_list "Tier 1" "${TIER1_CONFIGS[@]}"
        run_config_list "Tier 2 - Norm" "${TIER2_NORM_CONFIGS[@]}"
        run_config_list "Tier 2 - Data" "${TIER2_DATA_CONFIGS[@]}"
        run_config_list "Tier 2 - Reg" "${TIER2_REG_CONFIGS[@]}"
        run_config_list "Tier 2 - Opt" "${TIER2_OPT_CONFIGS[@]}"
        run_config_list "Tier 2 - Net" "${TIER2_NET_CONFIGS[@]}"
        run_config_list "Tier 2 - View" "${TIER2_VIEW_CONFIGS[@]}"
        run_config_list "Tier 2 - Combined" "${TIER2_COMBINED_CONFIGS[@]}"
        run_config_list "Tier 3 - N2V" "${TIER3_N2V_CONFIGS[@]}"
        run_config_list "Tier 3 - N2B" "${TIER3_N2B_CONFIGS[@]}"
        run_config_list "Tier 3 - S2S" "${TIER3_S2S_CONFIGS[@]}"

        print_header "ALL EXPERIMENTS COMPLETED!"
        ;;
    *)
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  list          - 列出所有实验"
        echo "  all           - 运行所有实验 (默认)"
        echo "  baseline      - 运行 baseline 验证"
        echo "  tier1         - 运行 Tier 1 方法对比"
        echo "  tier2         - 运行 Tier 2 所有消融"
        echo "  tier2_norm    - 运行 Tier 2 归一化消融"
        echo "  tier2_data    - 运行 Tier 2 数据策略消融"
        echo "  tier2_reg     - 运行 Tier 2 正则化消融"
        echo "  tier2_opt     - 运行 Tier 2 优化器消融"
        echo "  tier2_net     - 运行 Tier 2 网络结构消融"
        echo "  tier2_view    - 运行 Tier 2 视角模式消融"
        echo "  tier2_combined - 运行 Tier 2 组合配置"
        echo "  tier3         - 运行 Tier 3 所有实验"
        echo "  tier3_n2v     - 运行 Tier 3 N2V 变体"
        echo "  tier3_n2b     - 运行 Tier 3 N2B 变体"
        echo "  tier3_s2s     - 运行 Tier 3 S2S 变体"
        echo ""
        echo "环境变量:"
        echo "  CUDA_VISIBLE_DEVICES=0,1  指定 GPU"
        echo "  DRY_RUN=1                 只打印命令不执行"
        echo "  SKIP_EXISTING=1           跳过已有检查点的实验"
        echo ""
        echo "示例:"
        echo "  ./scripts/run_all_experiments.sh list"
        echo "  ./scripts/run_all_experiments.sh tier2_norm"
        echo "  CUDA_VISIBLE_DEVICES=1 ./scripts/run_all_experiments.sh all"
        echo "  DRY_RUN=1 ./scripts/run_all_experiments.sh all"
        exit 1
        ;;
esac

