# Experiment Runner Usage Guide

## Overview

The `run_experiments.sh` script provides a unified interface to run all SPECT self-supervised denoising experiments with automatic resume support.

## Features

- ✅ **Auto-resume**: Automatically resumes from the latest checkpoint if interrupted (Ctrl+C safe)
- ✅ **Skip completed**: Already completed experiments are automatically skipped
- ✅ **Wandb persistence**: Wandb run ID is saved in training state for reliable resume
- ✅ **Failure resilient**: Continues to next experiment even if one fails

---

## Basic Usage

```bash
bash scripts/run_experiments.sh <tier> <mode>
```

---

## Tier 1: Method Comparison

### Run all Tier 1 methods (N2N, N2B, N2V, S2S)
```bash
bash scripts/run_experiments.sh tier1 all
```
**Note**: Noiser2Noise is excluded due to theoretical issues (compound Poisson distribution)

### Run a specific method
```bash
bash scripts/run_experiments.sh tier1 n2n          # Noise2Noise only
bash scripts/run_experiments.sh tier1 n2b          # Neighbor2Neighbor only
bash scripts/run_experiments.sh tier1 n2v          # Noise2Void only
bash scripts/run_experiments.sh tier1 s2s          # Self2Self only
```

**Total experiments**: 4 methods (N2N already completed, 3 need retraining)

---

## Tier 2: Ablation Studies

### Run ALL ablation experiments (33 configs)
```bash
bash scripts/run_experiments.sh tier2 all
```
**Execution order**: norm → regularization → optimization → data_strategy → network → combined

**Total experiments**: 33 configs across 6 categories

### Run a specific ablation category

#### Normalization + Loss Functions (6 configs)
```bash
bash scripts/run_experiments.sh tier2 norm
```
Configs: `linear_poisson`, `linear_mse`, `linear_l1`, `linear_charb`, `anscombe_poisson`, `anscombe_charb`

#### Regularization / Weight Decay (5 configs)
```bash
bash scripts/run_experiments.sh tier2 regularization
```
Configs: `wd_0`, `wd_1e5`, `wd_1e4`, `wd_5e4`, `wd_1e3`

#### Optimization / Learning Rate (7 configs)
```bash
bash scripts/run_experiments.sh tier2 optimization
```
Configs: `adam_1e4`, `adamw_1e4`, `adamw_5e5`, `adamw_5e4`, `etamin_1e7`, `etamin_1e6`, `etamin_1e4`

#### Data Strategy / Patch & Batch (7 configs)
```bash
bash scripts/run_experiments.sh tier2 data_strategy
```
Configs: `patch256_batch8`, `patch128_batch16`, `patch64_batch32`, `patch64_batch16`, `patch64_batch16_aug`, `patch32_batch64`, `patch64_batch8`

#### Network Architecture (5 configs)
```bash
bash scripts/run_experiments.sh tier2 network
```
Configs: `nb2`, `nb4`, `nb6`, `nc_32_64_128_256`, `nc_96_192_384_768`

#### Combined / Final Comparison (3 configs)
```bash
bash scripts/run_experiments.sh tier2 combined
```
Configs: `best_config`, `original_kair`, `worst_config`

---

## Tier 3: Method-Specific Experiments

### Run experiments for a specific method
```bash
bash scripts/run_experiments.sh tier3 n2v          # Noise2Void experiments
bash scripts/run_experiments.sh tier3 n2b          # Neighbor2Neighbor experiments
bash scripts/run_experiments.sh tier3 s2s          # Self2Self experiments
```

---

## Advanced Usage

### Specify GPU
```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_experiments.sh tier1 all
CUDA_VISIBLE_DEVICES=1 bash scripts/run_experiments.sh tier2 norm
```

### Run in background with logging
```bash
nohup bash scripts/run_experiments.sh tier2 all > tier2_all.log 2>&1 &
```

### Resume after interruption
Simply re-run the same command:
```bash
bash scripts/run_experiments.sh tier2 all
```
The script will:
- Skip completed experiments
- Resume interrupted experiments from the latest checkpoint
- Continue with remaining experiments

---

## Expected Execution Time

| Tier | Mode | Configs | Time/Config | Total Time |
|------|------|---------|-------------|------------|
| Tier 1 | all | 3 (N2N skipped) | 8-10 hours | ~24-30 hours |
| Tier 2 | all | 33 | 8-10 hours | ~264-330 hours |
| Tier 2 | norm | 6 | 8-10 hours | ~48-60 hours |
| Tier 2 | regularization | 5 | 8-10 hours | ~40-50 hours |
| Tier 2 | optimization | 7 | 8-10 hours | ~56-70 hours |
| Tier 2 | data_strategy | 7 | 8-10 hours | ~56-70 hours |
| Tier 2 | network | 5 | 8-10 hours | ~40-50 hours |
| Tier 2 | combined | 3 | 8-10 hours | ~24-30 hours |

---

## Output & Logs

### Training logs
```
experiments/<exp_name>/train_<exp_name>.log
```

### Checkpoints
```
experiments/<exp_name>/models/
experiments/<exp_name>/training_states/
```

### Validation images
```
experiments/<exp_name>/visualization/
```

### TensorBoard logs
```
experiments/<exp_name>/tb_logger/
```

---

## Examples

### Scenario 1: First-time full Tier 1 run
```bash
bash scripts/run_experiments.sh tier1 all
```
**Result**: Runs N2B, N2V, S2S (skips completed N2N)

### Scenario 2: Run Tier 2 ablations one category at a time
```bash
bash scripts/run_experiments.sh tier2 norm
# Wait for completion...
bash scripts/run_experiments.sh tier2 regularization
# Wait for completion...
bash scripts/run_experiments.sh tier2 optimization
# ... and so on
```

### Scenario 3: Run all Tier 2 ablations overnight
```bash
nohup bash scripts/run_experiments.sh tier2 all > tier2_full.log 2>&1 &
tail -f tier2_full.log  # Monitor progress
```

### Scenario 4: Resume after power outage
```bash
# Just re-run the same command
bash scripts/run_experiments.sh tier2 all
```
**Result**:
- Completed experiments are skipped
- Interrupted experiments resume from latest checkpoint
- Remaining experiments start fresh

---

## Troubleshooting

### Check if an experiment is completed
```bash
grep "End of training" experiments/<exp_name>/train_<exp_name>.log
```

### Check latest checkpoint
```bash
ls -lt experiments/<exp_name>/training_states/*.state | head -1
```

### Manually set resume point
Edit the config file and set:
```yaml
path:
  resume_state: experiments/<exp_name>/training_states/10000.state
```

### Force restart an experiment
```bash
rm -rf experiments/<exp_name>/
bash scripts/run_experiments.sh tier1 <method>
```

---

## Summary

| Command | Description |
|---------|-------------|
| `tier1 all` | Run all Tier 1 methods (N2N, N2B, N2V, S2S) |
| `tier1 <method>` | Run specific Tier 1 method |
| `tier2 all` | Run all 33 Tier 2 ablation configs |
| `tier2 <category>` | Run specific ablation category (norm/regularization/optimization/data_strategy/network/combined) |
| `tier3 <method>` | Run Tier 3 method-specific experiments |

**Pro tip**: Use `tier2 all` to run everything unattended!
