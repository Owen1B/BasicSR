# NEMA Single Projection Fine-tuning

## Overview

Fine-tune the SPECT229 pre-trained model on a single NEMA projection data using mixed-dose N2N strategy.

### Key Features

- ✅ **Single sample training**: Uses only `ProjectionImage_Original.dat`
- 🎲 **In-batch diversity**: Each batch item uses different k ∈ {2,3,4,5} for Poisson sampling
- 🔄 **Data augmentation**: Horizontal flip (optional rotation)
- 🧠 **Transfer learning**: Initializes from SPECT229 pre-trained weights
- 💾 **Memory efficient**: Patch mode for limited GPU memory

## Data Format

**Input file**: `datasets/NEMA/osem/ProjectionImage_Original.dat`
- Shape: 120 views × 256 × 256
- Type: uint16
- Order: Fortran (column-major)
- Size: ~15 MB

## Training Strategies

### Strategy 1: Patch-based (Recommended for <24GB GPU)

**Config**: `finetune_nema_single_projection.yml`

- Extracts random patches: 60×128×128
- Batch size: 2
- Gradient accumulation: 4 (effective batch = 8)
- Training time: ~3-4 hours (10k iterations)
- Memory: ~12-16 GB

```bash
# Train
PYTHONPATH="./:${PYTHONPATH}" \
CUDA_VISIBLE_DEVICES=0 \
python basicsr/train.py -opt options/train/nema/finetune_nema_single_projection.yml
```

### Strategy 2: Full Volume (For ≥24GB GPU)

**Config**: `finetune_nema_single_projection_fullvolume.yml`

- Full volume: 120×256×256
- Batch size: 1
- Gradient accumulation: 8 (effective batch = 8)
- Training time: ~2-3 hours (5k iterations)
- Memory: ~20-24 GB

```bash
# Train
PYTHONPATH="./:${PYTHONPATH}" \
CUDA_VISIBLE_DEVICES=0 \
python basicsr/train.py -opt options/train/nema/finetune_nema_single_projection_fullvolume.yml
```

## Mixed-Dose N2N Mechanism

For each training sample, the algorithm:

1. **Randomly selects k** ∈ {2, 3, 4, 5}
2. **Generates k Poisson samples** from the original projection
3. **Creates two groups**:
   - Split A: k//2 samples
   - Split B: remaining samples
4. **Trains N2N**: Split A → predict Split B

This creates diverse training pairs within a single batch, preventing overfitting.

### Example (k=4):

```
Original: [100, 100, 100, ...]  (counts)
         ↓ Split into 4 Poisson samples
Sample 1: [24, 26, 23, ...]  (dose = 25%)
Sample 2: [23, 25, 24, ...]  (dose = 25%)
Sample 3: [26, 24, 25, ...]  (dose = 25%)
Sample 4: [25, 27, 26, ...]  (dose = 25%)
         ↓ Group randomly
Split A:  [48, 50, 47, ...]  (samples 1+2, dose = 50%)
Split B:  [51, 51, 51, ...]  (samples 3+4, dose = 50%)
         ↓ N2N training
Loss: PoissonNLL(Model(Split A), Split B)
```

## Pre-trained Model Setup

1. **Download the pre-trained model** (or copy from your existing experiments):

```bash
mkdir -p experiments/pretrained_models
cp /root/shared-nvme/BasicSR/experiments/n2n_spect229_60view3d_unet_projonly_full_multidose_L_k2_pretrain_from_Lscratch33k/models/net_g_91000.pth \
   experiments/pretrained_models/n2n_spect229_net_g_91000.pth
```

2. **Verify the model**:

```python
import torch
ckpt = torch.load('experiments/pretrained_models/n2n_spect229_net_g_91000.pth')
print(ckpt.keys())  # Should have 'params_ema'
```

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir experiments/finetune_nema_single_projection/tb_logger
```

### Wandb (Optional)

Set your wandb API key and the training will automatically log to wandb project `nema_finetune`.

## Expected Results

After fine-tuning:

- **Convergence**: Loss should decrease in first 1-2k iterations
- **Validation PSNR**: ~45-50 dB (on validation split with k=2)
- **Reconstruction quality**: Should see noise reduction in OSEM reconstructions

## Inference

After training, use the fine-tuned model for inference:

```python
import torch
import numpy as np
from basicsr.archs import build_network

# Load fine-tuned model
model = build_network({
    'type': 'UNet3DRes',
    'in_nc': 1, 'out_nc': 1,
    'nc': [32, 64, 128, 256], 'nb': 4,
    'bias': False, 'global_residual': True,
    'circular_depth': True, 'act': 'relu'
})

ckpt = torch.load('experiments/finetune_nema_single_projection/models/net_g_best.pth')
model.load_state_dict(ckpt['params_ema'])
model.eval().cuda()

# Load data
proj = np.fromfile('datasets/NEMA/osem/ProjectionImage_Original.dat', dtype=np.uint16)
proj = proj.reshape(120, 256, 256, order='F').astype(np.float32)

# Normalize
proj_norm = proj / 150.0

# Inference
with torch.no_grad():
    proj_tensor = torch.from_numpy(proj_norm).unsqueeze(0).unsqueeze(0).cuda()
    output = model(proj_tensor)
    output_np = output.squeeze().cpu().numpy() * 150.0

# Save
output_np.astype(np.float32).tofile('NEMA_finetuned_denoised.dat')
```

## Hyperparameter Tuning

If results are not satisfactory, try adjusting:

1. **Learning rate**: 5e-7 to 5e-6
2. **Weight decay**: 1e-6 to 1e-4
3. **Virtual dataset size**: 200 to 1000
4. **Patch size**: 60×128×128 to 80×192×192
5. **K choices**: [2, 3] (less diversity) or [2,3,4,5,6] (more diversity)

## Troubleshooting

### Out of Memory (OOM)

- Use patch mode
- Reduce `batch_size_per_gpu` to 1
- Increase `accumulation_steps`
- Enable `amp: enable: true` (mixed precision)

### Model diverges (loss increases)

- Reduce learning rate by 10×
- Increase warmup iterations
- Check data loading (visualize input/output pairs)

### No improvement after fine-tuning

- Pre-trained model may already be optimal
- Try different k_choices or augmentation
- Verify data quality (check for artifacts)

## Citation

If you use this code, please cite:

```
@inproceedings{basicsr,
  author = {Xintao Wang and Liangbin Xie and Ke Yu and Kelvin C.K. Chan and Chen Change Loy and Chao Dong},
  title = {BasicSR: Open Source Image and Video Restoration Toolbox},
  booktitle = {ACM Multimedia},
  year = {2022}
}
```








