# Tier 1 Method Comparison - Quick Start

## Overview

Compare 4 self-supervised denoising methods on SPECT Poisson noise (Noiser2Noise removed due to failure).

| Method | Input | Expected PSNR | Note |
|--------|-------|--------------|------|
| N2N | Split 1x→2×0.5x | 52-53 dB | Reference (may not be absolute best) |
| N2B | Single 1x | 48-50 dB | Spatial downsampling |
| N2V | Single 1x | 50-51 dB | Blind-spot network |
| S2S | Single 1x | 48-50 dB | Dropout + MC |

**Note**: All methods now use AdamW + weight_decay=1e-4 and 50k iterations.

## Running Experiments

```bash
# Run all 4 methods (Noiser2Noise excluded)
./scripts/run_spect_selfsup_experiments.sh tier1 all

# Run specific method
./scripts/run_spect_selfsup_experiments.sh tier1 n2n
./scripts/run_spect_selfsup_experiments.sh tier1 n2b
./scripts/run_spect_selfsup_experiments.sh tier1 n2v
./scripts/run_spect_selfsup_experiments.sh tier1 s2s

# Specify GPU
CUDA_VISIBLE_DEVICES=1 ./scripts/run_spect_selfsup_experiments.sh tier1 all
```

## Resume Support

All experiments auto-resume from checkpoint if interrupted. Just re-run the same command.

## Results

- **Logs**: `experiments/<method>_baseline/train_*.log`
- **Wandb**: Project `spect_selfsup_denoising`
- **Checkpoints**: `experiments/<method>_baseline/models/`

## Expected Time

~26 hours for all 5 methods on RTX 3090 (5 methods × ~5 hours each).

## Next Steps

After Tier 1 completes, run Tier 2 ablation studies:

```bash
./scripts/run_spect_selfsup_experiments.sh tier2 norm
./scripts/run_spect_selfsup_experiments.sh tier2 regularization
# ... etc
```

See `../README.md` for full experimental design.
