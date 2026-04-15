#!/bin/bash
# ============================================================================
# SPECT Self-Supervised Denoising - Experiment Runner with Auto-Resume
# ============================================================================
# Features:
# - Auto-resume from checkpoint if interrupted (Ctrl+C safe)
# - Skip completed experiments
# - Wandb ID persisted in training state for reliable resume
# - Continue to next experiment even if one fails
# ============================================================================

cd "$(dirname "$0")/.."

TIER=${1:-tier1}
MODE=${2:-all}
GPU=${CUDA_VISIBLE_DEVICES:-0}

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ============================================================================
# Helper Functions
# ============================================================================
is_completed() {
    local exp_name=$1
    local log_file="experiments/${exp_name}/train_${exp_name}.log"

    if [ -f "$log_file" ] && grep -q "End of training" "$log_file"; then
        return 0  # True - completed
    fi
    return 1  # False - not completed
}

get_resume_state() {
    local exp_name=$1
    local state_dir="experiments/${exp_name}/training_states"

    if [ ! -d "$state_dir" ]; then
        echo ""
        return
    fi

    # Find all .state files and get the latest one (by iteration number)
    local latest_state=$(find "$state_dir" -maxdepth 1 -name "*.state" -type f -printf '%f\n' | \
                        grep -E '^[0-9]+\.state$' | \
                        sed 's/\.state$//' | \
                        sort -n | \
                        tail -1)

    if [ -z "$latest_state" ]; then
        echo ""
        return
    fi

    echo "${state_dir}/${latest_state}.state"
}

run_single() {
    local config=$1
    local exp_name=$(basename "$config" .yml)

    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Experiment: ${exp_name}${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    # Check if already completed
    if is_completed "$exp_name"; then
        echo -e "${GREEN}✓ Already completed, skipping...${NC}"
        echo ""
        return 0
    fi

    # Check if can resume
    local resume_state=$(get_resume_state "$exp_name")
    if [ -n "$resume_state" ]; then
        echo -e "${YELLOW}⚠ Resuming from: $resume_state${NC}"
    else
        echo -e "${GREEN}→ Starting fresh training...${NC}"
    fi

    # Create temp config with auto_resume ALWAYS enabled
    local temp_config="/tmp/${exp_name}_resume_$$.yml"
    python3 - <<EOF
import yaml

with open('$config', 'r') as f:
    cfg = yaml.safe_load(f)

# ALWAYS enable auto_resume - BasicSR will:
# - If training_states/ exists: automatically find and load latest state + wandb_id
# - If training_states/ doesn't exist: start fresh training
cfg['auto_resume'] = True

with open('$temp_config', 'w') as f:
    yaml.dump(cfg, f, sort_keys=False, allow_unicode=True, default_flow_style=False)
EOF

    echo ""

    # Run training (capture exit code)
    # IMPORTANT: --auto_resume flag is required because parse_options() overrides YAML config
    PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=$GPU \
        python3 -m prime.train -opt "$temp_config" --auto_resume
    local exit_code=$?

    # Cleanup
    rm -f "$temp_config"

    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✓ Completed successfully${NC}"
        echo ""
        return 0
    else
        echo -e "${RED}✗ Failed with exit code $exit_code${NC}"
        echo ""
        return $exit_code
    fi
}

# ============================================================================
# Tier 1: Method Comparison
# ============================================================================
run_tier1() {
    local dir="options/train/spect_selfsup/tier1_method_comparison"

    case $MODE in
        all)
            # Note: noiser2noise excluded due to theoretical issues (compound Poisson distribution)
            local methods=(n2n n2b n2v s2s)
            local failed=0

            for method in "${methods[@]}"; do
                run_single "${dir}/${method}_baseline.yml" || failed=$((failed+1))
            done

            echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
            if [ $failed -eq 0 ]; then
                echo -e "${GREEN}✓ All ${#methods[@]} experiments completed!${NC}"
            else
                echo -e "${YELLOW}⚠ Completed with $failed failures${NC}"
            fi
            echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

            return $failed
            ;;
        n2n|n2b|n2v|s2s)
            run_single "${dir}/${MODE}_baseline.yml"
            ;;
        *)
            echo -e "${RED}Error: Unknown method '$MODE'${NC}"
            echo "Usage: $0 tier1 [all|n2n|n2b|n2v|s2s]"
            return 1
            ;;
    esac
}

# ============================================================================
# Tier 2: Ablation Studies
# ============================================================================
run_tier2() {
    local category=$MODE

    case $category in
        all)
            # Run all Tier 2 ablation categories
            local categories=(norm regularization optimization data_strategy network combined view_mode)
            local total_failed=0
            local total_experiments=0

            echo -e "${BLUE}Running ALL Tier 2 ablations (${#categories[@]} categories)${NC}"
            echo ""

            for cat in "${categories[@]}"; do
                local dir="options/train/spect_selfsup/tier2_ablation/${cat}"
                local configs=("${dir}"/*.yml)
                local count=${#configs[@]}
                total_experiments=$((total_experiments + count))

                echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
                echo -e "${BLUE}Category: ${cat} (${count} configs)${NC}"
                echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
                echo ""

                for config in "${configs[@]}"; do
                    [ -f "$config" ] && run_single "$config" || total_failed=$((total_failed+1))
                done
            done

            echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
            if [ $total_failed -eq 0 ]; then
                echo -e "${GREEN}✓ All ${total_experiments} Tier 2 experiments completed!${NC}"
            else
                echo -e "${YELLOW}⚠ Completed ${total_experiments} experiments with $total_failed failures${NC}"
            fi
            echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

            return $total_failed
            ;;
        norm|regularization|optimization|data_strategy|network|combined|view_mode)
            # Run a specific category
            local dir="options/train/spect_selfsup/tier2_ablation/${category}"

            if [ ! -d "$dir" ]; then
                echo -e "${RED}Error: Category '$category' not found${NC}"
                return 1
            fi

            local configs=("${dir}"/*.yml)
            local total=${#configs[@]}
            local failed=0

            echo -e "${BLUE}Running Tier 2 ablation: ${category} (${total} configs)${NC}"
            echo ""

            for config in "${configs[@]}"; do
                [ -f "$config" ] && run_single "$config" || failed=$((failed+1))
            done

            echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
            if [ $failed -eq 0 ]; then
                echo -e "${GREEN}✓ All ${total} experiments completed!${NC}"
            else
                echo -e "${YELLOW}⚠ Completed with $failed failures${NC}"
            fi
            echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

            return $failed
            ;;
        *)
            echo -e "${RED}Error: Unknown category '$category'${NC}"
            echo "Usage: $0 tier2 [all|norm|regularization|optimization|data_strategy|network|combined|view_mode]"
            return 1
            ;;
    esac
}

# ============================================================================
# Tier 3: Method-Specific
# ============================================================================
run_tier3() {
    local method=$MODE
    local dir="options/train/spect_selfsup/tier3_other_methods/${method}"

    if [ ! -d "$dir" ]; then
        echo -e "${RED}Error: Method '$method' not found${NC}"
        echo "Available: n2v, n2b, s2s"
        return 1
    fi

    local configs=("${dir}"/*.yml)
    local total=${#configs[@]}
    local failed=0

    echo -e "${BLUE}Running Tier 3 experiments: ${method} (${total} configs)${NC}"
    echo ""

    for config in "${configs[@]}"; do
        [ -f "$config" ] && run_single "$config" || failed=$((failed+1))
    done

    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    if [ $failed -eq 0 ]; then
        echo -e "${GREEN}✓ All ${total} experiments completed!${NC}"
    else
        echo -e "${YELLOW}⚠ Completed with $failed failures${NC}"
    fi
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    return $failed
}

# ============================================================================
# Main
# ============================================================================
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  SPECT Self-Supervised Denoising${NC}"
echo -e "${BLUE}  Tier: $TIER | Mode: $MODE | GPU: $GPU${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

case $TIER in
    tier1) run_tier1 ;;
    tier2) run_tier2 ;;
    tier3) run_tier3 ;;
    *)
        echo -e "${RED}Error: Unknown tier '$TIER'${NC}"
        echo "Usage: $0 [tier1|tier2|tier3] [mode]"
        exit 1
        ;;
esac

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ All experiments completed successfully!${NC}"
else
    echo ""
    echo -e "${YELLOW}⚠ Some experiments failed or were interrupted${NC}"
    echo -e "${YELLOW}  You can re-run the script - completed experiments will be skipped${NC}"
fi

exit $exit_code
