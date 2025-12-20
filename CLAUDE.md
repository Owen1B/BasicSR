# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

BasicSR is an open-source image and video restoration toolbox based on PyTorch for tasks like super-resolution, denoising, deblurring, and JPEG artifact removal. It's designed as both a package and a development framework.

## Installation

From the repository root:

```bash
# Install dependencies
pip install -r requirements.txt

# Install BasicSR in development mode (without C++ extensions)
python setup.py develop

# OR with C++ extensions compiled during installation
BASICSR_EXT=True python setup.py develop
```

**C++ Extensions Notes:**
- Extensions include deformable convolution (DCN for EDVR) and StyleGAN operators (upfirdn2d, fused_act)
- Set `BASICSR_EXT=True` during installation to compile extensions (requires gcc/g++ >= 5)
- Set `BASICSR_JIT=True` during runtime to load extensions just-in-time without compilation
- If you don't need these extensions, skip setting these variables

## Training Commands

All commands should be run from the repository root.

### Single GPU Training

```bash
PYTHONPATH="./:${PYTHONPATH}" \
CUDA_VISIBLE_DEVICES=0 \
python basicsr/train.py -opt options/train/SRResNet_SRGAN/train_MSRResNet_x4.yml
```

### Multi-GPU Training (4 GPUs example)

```bash
PYTHONPATH="./:${PYTHONPATH}" \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 \
basicsr/train.py -opt options/train/EDVR/train_EDVR_M_x4_SR_REDS_woTSA.yml --launcher pytorch
```

Or use the convenience script:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
./scripts/dist_train.sh 4 options/train/EDVR/train_EDVR_M_x4_SR_REDS_woTSA.yml
```

## Testing Commands

### Single GPU Testing

```bash
PYTHONPATH="./:${PYTHONPATH}" \
CUDA_VISIBLE_DEVICES=0 \
python basicsr/test.py -opt options/test/SRResNet_SRGAN/test_MSRResNet_x4.yml
```

### Multi-GPU Testing

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
./scripts/dist_test.sh 4 options/test/EDVR/test_EDVR_M_x4_SR_REDS.yml
```

## Running Tests

Unit tests require a GPU CUDA environment. Test files are organized in the `tests/` directory:

```bash
# Example: Run a specific test
python tests/test_models/test_sr_model.py

# Run tests for a specific module
python tests/test_archs/test_srresnet_arch.py
```

## Code Architecture

### Dynamic Instantiation System

BasicSR uses **dynamic instantiation** via `importlib` and `getattr`. When adding new classes, they're automatically discovered based on file naming conventions:

| Module | File Suffix | Example |
|--------|------------|---------|
| Data | `_dataset.py` | `basicsr/data/paired_image_dataset.py` |
| Models | `_model.py` | `basicsr/models/sr_model.py` |
| Architectures | `_arch.py` | `basicsr/archs/srresnet_arch.py` |

Classes are automatically scanned and instantiated based on the type specified in YAML config files. **Do not use these suffixes for other files.**

For **losses** and **metrics**, you must manually register new classes in their respective `__init__.py` files:
- Losses: `basicsr/models/losses/__init__.py`
- Metrics: `basicsr/metrics/__init__.py`

### Core Module Structure

```
basicsr/
├── train.py              # Main training script
├── test.py               # Main testing script
├── archs/                # Network architectures (RRDBNet, EDVR, SwinIR, etc.)
├── data/                 # Dataset classes (paired images, video, LMDB, etc.)
├── models/               # Training models (SR, GAN, video recurrent, etc.)
│   └── losses/           # Loss functions (L1, perceptual, GAN losses)
├── metrics/              # Evaluation metrics (PSNR, SSIM, NIQE, etc.)
├── ops/                  # Custom CUDA operators (DCN, StyleGAN ops)
└── utils/                # Utilities (logging, options parsing, etc.)
```

### Configuration System

All experiments use YAML config files in `options/train/` and `options/test/`. Key sections:

- **name**: Experiment name (include "debug" to enter debug mode with more frequent logging)
- **model_type**: Model class name (e.g., `SRModel`, `ESRGANModel`, `VideoRecurrentGANModel`)
- **datasets**: Train/validation dataset configurations with dataloaders
- **network_g/network_d**: Generator/discriminator architecture definitions
- **train**: Training parameters (optimizers, schedulers, total iterations)
- **val**: Validation settings (metrics, save images)
- **logger**: TensorBoard, wandb, and file logging configuration
- **path**: Paths for pretrained models, results, and experiments

The framework automatically determines the root experiment path (default: `./experiments`). Users can override this via the `path` section in config files.

### Experiment Naming Convention

Example: `001_MSRResNet_x4_f64b16_DIV2K_1000k_B16G1_wandb`

- `001`: Experiment index
- `MSRResNet`: Model name
- `x4_f64b16`: Key parameters (upsampling ratio, channels, blocks)
- `DIV2K`: Training dataset
- `1000k`: Total iterations
- `B16G1`: Batch size 16, 1 GPU
- `wandb`: Uses wandb logger

### Supported Models and Architectures

**Image SR:** EDSR, RCAN, SRResNet, ESRGAN, RealESRGAN, SwinIR, ECBSR
**Video SR:** EDVR, BasicVSR, BasicVSR++, TOF, DUF
**GANs:** StyleGAN2, SRGAN, HiFaceGAN
**Denoising/Deblur:** RIDNet, CBDNet, DeblurGANv2

Each model has training/testing configs in `options/train/<ModelName>/` and `options/test/<ModelName>/`.

## Important Conventions

1. **Loss Item Naming**: Prefix loss items with `l_` (e.g., `l_g_pix`, `l_g_percep`, `l_d_real`) so they're grouped in TensorBoard under `losses/`.

2. **File Naming**: Strictly follow suffixes for dynamic instantiation:
   - Dataset classes: `*_dataset.py`
   - Model classes: `*_model.py`
   - Architecture classes: `*_arch.py`

3. **Class/Function Names**: Cannot be duplicated across the codebase due to dynamic instantiation.

4. **PYTHONPATH**: Always set `PYTHONPATH="./:${PYTHONPATH}"` when running scripts to ensure proper module imports.

5. **Debug Mode**: Include "debug" in experiment name to enable verbose logging and disable remote loggers.

## Data Preparation

Dataset preparation scripts are in `scripts/data_preparation/`. Common datasets include DIV2K, REDS, Vimeo90K. Refer to `docs/DatasetPreparation.md` for detailed instructions.

Datasets can be stored as:
- Disk images (most common)
- LMDB databases (faster I/O for large datasets)
- Meta info files (for specific workflows)

## Pretrained Models

Download pretrained models using:

```bash
python scripts/download_pretrained_models.py
```

Models are stored in `experiments/pretrained_models/` by default. See `docs/ModelZoo.md` for available models.

## Inference

Inference scripts are in the `inference/` directory. Examples:

```bash
# StyleGAN2 inference
python inference/inference_stylegan2.py

# SR model inference
python inference/inference_esrgan.py
```

## Logging

- Training logs: `experiments/<exp_name>/`
- TensorBoard logs: `experiments/<exp_name>/tb_logger/`
- Validation images: `experiments/<exp_name>/visualization/`
- Model checkpoints: `experiments/<exp_name>/models/`

## Additional Resources

- Documentation: See `docs/` folder for detailed guides
- Model Zoo: `docs/ModelZoo.md`
- Design Conventions: `docs/DesignConvention.md`
- Configuration Guide: `docs/Config.md`
- HOWTOs: `docs/HOWTOs.md` for specific model training/inference examples

---

# SPECT Self-Supervised Denoising Experiments

## Project Overview

**Goal**: Develop SOTA self-supervised denoising methods for SPECT images with Poisson noise using only 1x noisy data.

**Research Question**: Which self-supervised method achieves the best performance on SPECT Poisson noise?

## Experiment Design (3-Tier Structure)

### Tier 1: Method Comparison (5 experiments)
**Location**: `options/train/spect_selfsup/tier1_method_comparison/`

Compare 5 self-supervised methods on equal footing:

1. **N2N (Noise2Noise)** - Split 1x → 2×0.5x, paired training
   - Config: Linear + PoissonNLLLoss (optimal for Poisson)
   - Expected: 52-53 dB (reference, but using split 0.5x pairs)
   - **Note**: May not be absolute best since using split 0.5x pairs (not real 1x pairs)

2. **Noiser2Noise** (NEW) - Single 1x, Poisson sampling for augmentation
   - Config: Linear + PoissonNLLLoss (theoretically optimal)
   - Expected: 51-52 dB (close to N2N, theoretically sound)
   - **Key**: Online Poisson sampling in count domain after crop
   - **Advantage**: Single image + PoissonNLL + sampling diversity

3. **N2V (Noise2Void)** - Single image, blind-spot network
   - Config: Anscombe + CharbonnierLoss (original)
   - Expected: 50.5-51.5 dB (blind-spot loss ~1-2 dB)

4. **N2B (Neighbor2Neighbor)** - Single image, spatial downsampling
   - Config: Anscombe + CharbonnierLoss (original)
   - Expected: 51.5-52.0 dB (close to N2N due to SPECT smoothness)

5. **S2S (Self2Self)** - Single image, dropout + MC sampling
   - Config: Anscombe + CharbonnierLoss (original)
   - Expected: 50.0-51.0 dB (MC variance)

**Key Settings** (Unified across all methods):
- `max_value: 150.0` (for consistent evaluation)
- `data_range: 150.0` (for PSNR/SSIM metrics)
- Manual seed: 20020113
- Val split: 96-100 (same for fair comparison)
- Resume support enabled
- Wandb project: `spect_selfsup_denoising`

### Tier 2: Deep Ablation on Best Method (~30 experiments)
**Location**: `options/train/spect_selfsup/tier2_ablation/`

Systematic ablation on the best-performing method from Tier 1 to understand key factors:

1. **norm/** (6 configs) - Normalization & Loss Function ✅ COMPLETED
   - Linear + PoissonNLL (optimal)
   - Anscombe + Charbonnier
   - Anscombe + PoissonNLL (fails - domain mismatch)
   - Linear + Charbonnier
   - Linear + MSE
   - Linear + L1

2. **regularization/** (5 configs) - Weight Decay
   - wd=1e-4 (baseline)
   - wd=0 (severe overfitting)
   - wd=1e-3, 5e-4, 1e-5

3. **optimization/** (7 configs) - LR Schedule
   - AdamW vs Adam
   - Different learning rates (5e-5, 1e-4, 5e-4)
   - Different eta_min (1e-7, 1e-6, 1e-5, 1e-4)

4. **data_strategy/** (7 configs) - Patch & Batch Size ✅ COMPLETED
   - patch=64/batch=16 (baseline)
   - patch=128/256, batch=8/32
   - With/without augmentation
   - **Best**: patch64_batch32 (52.82 dB)

5. **network/** (5 configs) - Architecture
   - UNetRes nb=2/4/6
   - Channel widths: narrow/baseline/wide

6. **view_mode/** (7 configs) - View Fusion Strategies
   - Single-view vs dual-view
   - Late fusion architecture
   - Attention fusion architecture

7. **combined/** (3 configs) - Extreme Cases
   - Best config (all optimal)
   - Worst config (all negative factors)
   - Original KAIR alignment

### Tier 3: Method-Specific Improvements (~12 experiments)
**Location**: `options/train/spect_selfsup/tier3_other_methods/`

Explore if PoissonNLL can improve other methods:

1. **n2v/** (4 configs)
   - Anscombe + Charbonnier (baseline)
   - Linear + Charbonnier
   - Anscombe + PoissonNLL (experimental)
   - Linear + PoissonNLL (optimal if theory holds)

2. **n2b/** (4 configs)
   - Lambda ablation (λ1/λ2 weights)
   - Normalization comparison

3. **s2s/** (4 configs)
   - Dropout rate (0.2, 0.3, 0.4)
   - MC samples (50, 100)

## Experimental Results

### Tier 2 Ablation Results

#### 1. Normalization & Loss Function (norm/) ✅ COMPLETED

**Summary**: Linear normalization + PoissonNLLLoss is the clear winner, achieving 53.91 dB PSNR.

| Rank | Config | Normalization | Loss Function | Best PSNR (EMA) | Iter | Relative Gap |
|------|--------|---------------|---------------|-----------------|------|--------------|
| 🥇 | n2n_linear_poisson | Linear | PoissonNLL | **53.91 dB** | 48k | - |
| 🥈 | n2n_linear_mse | Linear | MSE | **53.72 dB** | 40k | -0.19 dB |
| 🥉 | n2n_linear_l1 | Linear | L1 | 52.04 dB | 12k | -1.87 dB |
| 4 | n2n_linear_charb | Linear | Charbonnier | 52.01 dB | 13k | -1.90 dB |
| 5 | n2n_anscombe_charb | Anscombe | Charbonnier | 51.89 dB | 13k | -2.02 dB |
| ❌ | n2n_anscombe_poisson | Anscombe | PoissonNLL | FAILED | - | Domain mismatch |

**Key Findings**:
1. **Linear + PoissonNLL is optimal** (+1.87 dB vs L1/Charbonnier)
   - Theoretically correct: directly models Poisson negative log-likelihood
   - Most stable training: continues improving to 48k iterations
   - Must use with `norm_type: linear` in loss config

2. **Linear + MSE performs surprisingly well** (53.72 dB, -0.19 dB gap)
   - Converges faster (40k vs 48k)
   - Simpler implementation, still very effective
   - Good alternative if PoissonNLL unavailable

3. **Anscombe normalization is suboptimal**
   - Best Anscombe result: 51.89 dB (-2.02 dB gap)
   - Variance stabilization theory doesn't translate to better performance
   - **NEVER combine Anscombe + PoissonNLL**: domain mismatch causes failure

4. **L1 vs Charbonnier: minimal difference**
   - Both converge early (12-13k) with similar performance
   - Both significantly worse than PoissonNLL/MSE

5. **Critical Bug Confirmed**: `Anscombe + PoissonNLL` combination fails
   - PoissonNLLLoss expects count domain data
   - Anscombe transforms to Anscombe domain (variance-stabilized)
   - Domain mismatch causes training failure
   - **Action**: This config should be removed or marked invalid

**Recommendation**: Use `norm: linear` + `loss: PoissonNLLLoss` for all SPECT denoising experiments.

---

#### 2. Data Strategy (data_strategy/) ✅ COMPLETED

**Summary**: patch64_batch32 achieves best balance of diversity and stability.

| Rank | Config | Patch Size | Batch Size | Iters/Epoch | Best PSNR (EMA) | Relative Gap |
|------|--------|-----------|-----------|-------------|-----------------|--------------|
| 🥇 | n2n_patch64_batch32 | 64 | 32 | 12 | **52.82 dB** | - |
| 🥈 | n2n_patch64_batch16 | 64 | 16 | 24 | 52.74 dB | -0.08 dB |
| 🥉 | n2n_patch64_batch8 | 64 | 8 | 48 | 52.71 dB | -0.11 dB |
| 4 | n2n_patch128_batch16 | 128 | 16 | 6 | 52.64 dB | -0.18 dB |
| 5 | n2n_patch256_batch8 | 256 | 8 | 1.5 | 52.31 dB | -0.51 dB |
| 6 | n2n_patch32_batch64 | 32 | 64 | 24 | 52.25 dB | -0.57 dB |
| 7 | n2n_patch64_batch16_aug | 64 | 16 | 24 | 51.96 dB | -0.86 dB |

**Key Findings**:
1. **patch64_batch32 is optimal** (12 iters/epoch)
   - Good balance: diversity from small patches + stability from large batch
   - All subsequent ablations should use this configuration

2. **Larger patches hurt performance**
   - patch256: -0.51 dB (too little diversity per epoch)
   - patch128: -0.18 dB (moderate degradation)

3. **Data augmentation (flip/rotate) hurts performance** (-0.86 dB)
   - SPECT has anatomical orientation that should be preserved
   - Random augmentation destroys structural information

**Recommendation**: Use `gt_size: 64` + `batch_size_per_gpu: 32` + `dataset_enlarge_ratio: 4`.

---

## Key Implementation Details

### Overfitting Prevention (Critical for N2N)

**Problem Identified**: N2N with incorrect hyperparameters shows PSNR rise-then-fall pattern (overfitting).

**Solution** (from KAIR alignment analysis):
1. ⭐⭐⭐⭐⭐ `weight_decay: 1e-4` (most critical)
2. ⭐⭐⭐⭐ `eta_min: 1e-5` (not too small, e.g., 1e-7 causes late-stage damage)
3. ⭐⭐⭐⭐ Small patch size (64) for diversity
4. ⭐⭐⭐ Large batch size (16) for stability

**Reference**: `options/train/demo/KAIR_vs_BasicSR_ANALYSIS.md`

### Poisson Noise Characteristics

- **Distribution**: y ~ Poisson(λ)
- **Mean**: E[y] = λ (unbiased)
- **Variance**: Var[y] = λ (heteroscedastic)
- **Low counts**: Highly asymmetric, noise can exceed signal
- **Optimal loss**: PoissonNLLLoss (negative log-likelihood)

### Normalization Strategies

**EXPERIMENTAL EVIDENCE** (from norm ablation):

1. **Linear** (`max_value: 150.0`) ⭐⭐⭐⭐⭐ **BEST**
   - Preserves Poisson characteristics
   - **Must use with PoissonNLLLoss**: 53.91 dB (optimal)
   - Also works well with MSE: 53.72 dB (-0.19 dB)
   - Significantly outperforms Anscombe: +2.02 dB improvement

2. **Anscombe** (variance stabilization) ⭐⭐
   - Transform: f(y) = 2√(y + 3/8)
   - Makes variance approximately constant
   - Best result: 51.89 dB with Charbonnier (-2.02 dB vs Linear+PoissonNLL)
   - ❌ **NEVER use with PoissonNLLLoss**: domain mismatch causes failure
   - Traditional approach but experimentally suboptimal

**Domain Mismatch Bug** ⚠️:
- PoissonNLLLoss expects data in **count domain**
- Anscombe transform converts to **Anscombe domain** (variance-stabilized)
- Combining them causes training failure
- Config `n2n_anscombe_poisson` is invalid and should not be used

## Running Experiments

### Quick Start

```bash
# Tier 1: Method comparison
./scripts/run_experiments.sh tier1 all          # All 4 methods
./scripts/run_experiments.sh tier1 n2n          # Single method

# Tier 2: Ablation studies (after Tier 1 completes)
./scripts/run_experiments.sh tier2 norm         # Norm/loss ablation
./scripts/run_experiments.sh tier2 regularization

# Tier 3: Method-specific
./scripts/run_experiments.sh tier3 n2v
```

### Resume Support

All experiments support automatic resumption:
- Script checks for `experiments/<name>/training_states/latest.state`
- If exists, automatically resumes from checkpoint
- Wandb runs are also resumed with saved run IDs

### Monitoring

- **Logs**: `experiments/<exp_name>/train_*.log`
- **TensorBoard**: `experiments/<exp_name>/tb_logger/`
- **Wandb**: Project `spect_selfsup_denoising`
- **Checkpoints**: Only keep 2 latest (save space)

## Expected Results & Timeline

### Performance Targets

| Method | Expected PSNR | Ranking | Notes |
|--------|--------------|---------|-------|
| N2N | 52-53 dB | Reference | May not be absolute best (0.5x pairs) |
| Noiser2Noise | 51-52 dB | High | Single 1x + Poisson sampling + PoissonNLL |
| N2B | 51.5-52.0 dB | High | SPECT smoothness helps downsampling |
| N2V | 50.5-51.5 dB | Medium | Blind-spot loss offset by correlation |
| S2S | 50.0-51.0 dB | Lower | MC variance, slower inference |

**Important**:
- N2N uses split 0.5x pairs, not true 1x pairs, so may not achieve absolute best performance.
- **Noiser2Noise** is theoretically sound and may compete with or exceed N2N.
- Other single-image methods may also compete or exceed N2N.

### Training Time (Single RTX 3090)

- Tier 1: ~26 hours (5 methods)
- Tier 2: ~60 hours (30 ablations)
- Tier 3: ~24 hours (12 experiments)
- **Total**: ~110 hours (~4-5 days)

## Critical Implementation Notes

### For Claude Code / Future Developers

**⭐ ALWAYS USE OPTIMAL BASELINE CONFIGURATION** (unless specifically testing alternatives):
- Normalization: `norm: linear` (NOT anscombe)
- Loss: `type: PoissonNLLLoss` with `norm_type: linear`
- Data: `gt_size: 64`, `batch_size_per_gpu: 32`, `dataset_enlarge_ratio: 4`
- Regularization: `weight_decay: 1e-4`
- LR Schedule: `eta_min: 1e-5`
- No augmentation: `use_hflip: false`, `use_rot: false`

This baseline achieves ~53.9 dB PSNR. Any deviations should be explicitly justified.

---

1. **Always use unified max_value=150.0**
   - Training: `max_value: 150.0`
   - Validation: `data_range: 150.0`
   - Visualization: `vmax_cnt: 150.0`

2. **Metric calculation**
   - Use `calculate_psnr_range` (not `calculate_psnr`)
   - Use `calculate_ssim_range` (not `calculate_ssim`)
   - Always specify `data_range=150.0`

3. **Anscombe inverse**
   - Use `dataset.inverse_from_norm()` for proper handling
   - Function automatically handles Anscombe/Linear normalization

4. **Resume implementation**
   - Check `experiments/<name>/training_states/latest.state`
   - Update config `resume_state` field dynamically
   - Extract wandb run ID from directory if exists

5. **Loss function compatibility** ⚠️ CRITICAL
   - **PoissonNLLLoss**: MUST use with `norm_type: linear` only
     - Expects data in count domain (not Anscombe domain)
     - Best performance: 53.91 dB (experimentally verified)
     - ❌ NEVER combine with Anscombe normalization (domain mismatch → failure)
   - **MSE**: Works well with linear normalization (53.72 dB)
     - Good alternative to PoissonNLL, slightly faster convergence
   - **CharbonnierLoss/L1**: Can use with any normalization
     - Lower performance (~52 dB) regardless of normalization
     - Early convergence (12-13k iterations)

   **Summary from experiments**:
   - Best: Linear + PoissonNLL (53.91 dB)
   - Alternative: Linear + MSE (53.72 dB, -0.19 dB)
   - Avoid: Anscombe + PoissonNLL (FAILS)
   - Suboptimal: Any combination with L1/Charbonnier (~52 dB)

6. **Model types required**
   - `SRModel`: Standard training (N2N, N2V)
   - `Neighbor2NeighborModel`: Dual-loss training (N2B)
   - `Self2SelfModel`: MC Dropout inference (S2S)

7. **Dataset types required**
   - `SPECTDatPairedDataset`: Standard paired data
   - `Neighbor2NeighborDataset`: Spatial downsampling
   - `Self2SelfDataset`: Dropout augmentation

## File Organization

```
options/train/spect_selfsup/
├── README.md                          # Experiment overview
├── TIER1_RESULTS.md                   # Tier 1 method comparison results
├── tier1_method_comparison/
│   ├── QUICK_START.md                 # User guide
│   ├── n2n_baseline.yml               # Noise2Noise
│   ├── n2v_baseline.yml               # Noise2Void
│   ├── n2b_baseline.yml               # Neighbor2Neighbor
│   └── s2s_baseline.yml               # Self2Self
├── tier2_ablation/
│   ├── norm/                          # 6 configs ✅ COMPLETED
│   ├── regularization/                # 5 configs
│   ├── optimization/                  # 7 configs
│   ├── data_strategy/                 # 7 configs ✅ COMPLETED
│   ├── network/                       # 5 configs
│   ├── view_mode/                     # 7 configs
│   └── combined/                      # 3 configs
└── tier3_other_methods/
    ├── n2v/                           # 4 configs
    ├── n2b/                           # 4 configs
    └── s2s/                           # 4 configs

options/train/demo/                    # Legacy experiments
├── KAIR_vs_BasicSR_ANALYSIS.md        # Overfitting analysis
├── n2n_base_aligned_kair.yml          # Verified optimal config
└── n2n_ablation_*.yml                 # Initial ablation studies

scripts/
├── run_experiments.sh                 # Unified experiment runner
└── EXPERIMENT_RUNNER_GUIDE.md         # Runner documentation

basicsr/archs/
├── unet_latefusion_arch.py            # Late fusion dual-view architecture
└── unet_attentionfusion_arch.py       # Attention fusion dual-view architecture
```

## Publication Target

**Goal**: Top-tier conference (CVPR/MICCAI/TMI)

**Contributions**:
1. First systematic comparison of self-supervised methods on SPECT Poisson noise
2. **Experimental validation of PoissonNLLLoss optimality**: 53.91 dB vs 52.01 dB (Charbonnier)
3. **Discovery of Anscombe+PoissonNLL domain mismatch bug**: prevents training failure
4. Identification of key factors preventing overfitting (weight decay, eta_min, patch size)
5. **Data strategy optimization**: patch64_batch32 achieves +0.57 dB vs naive patch32_batch64
6. Quantification of SPECT-specific advantages (smoothness → less performance gap)
7. Practical guidelines for method selection with experimental evidence

**Novel Findings**:
- Linear normalization consistently outperforms Anscombe (+2.02 dB)
- MSE loss nearly matches PoissonNLL performance (53.72 vs 53.91 dB)
- Data augmentation hurts SPECT denoising performance (-0.86 dB)
- Loss function choice (+1.87 dB gain) more important than data strategy (+0.57 dB gain)

---

**Last Updated**: 2025-12-16
**Contact**: Owen
**Status**:
- ✅ Tier 2 norm ablation completed (5/6 successful, 1 failed as expected)
- ✅ Tier 2 data_strategy ablation completed (7/7 successful)
- 🔄 Tier 2 remaining ablations: regularization, optimization, network, view_mode, combined
- 📋 Tier 1 method comparison: ready to run
- 📋 Tier 3 method-specific improvements: pending
