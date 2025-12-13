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
