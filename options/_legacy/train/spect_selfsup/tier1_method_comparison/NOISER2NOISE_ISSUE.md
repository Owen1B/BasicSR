# Noiser2Noise Method - Critical Issue Analysis

**Status**: ❌ **Method Failed - Theoretical Problem**

**Result**: PSNR = 36.05 dB (Expected: 51-52 dB)

**Training Loss**: `-1.0479e+10` (Abnormally large negative value)

---

## Problem Description

Noiser2Noise method shows **catastrophic failure** with:
- Performance 16 dB worse than expected
- Loss magnitude 10 billion times larger than N2N (-1e10 vs -2 to -5)
- Unable to learn meaningful denoising

---

## Root Cause Analysis

### Theoretical Issue: Compound Poisson Process

**Original Noiser2Noise Theory** (works for additive Gaussian noise):
- Input: x + n1 (signal + noise1)
- GT: x + n2 (signal + noise2)
- Works because: E[x + n2 | x + n1] = x

**Our Implementation** (for Poisson noise):
- GT: 1x ~ Poisson(λ_clean)  [1x is already noisy!]
- Input: y ~ Poisson(1x)  [Sampling from noisy 1x]

**The Problem**:
1. 1x itself is a Poisson sample: 1x ~ Poisson(λ_clean)
2. Sampling from 1x creates: y ~ Poisson(1x) ~ Poisson(Poisson(λ_clean))
3. This is a **compound Poisson process**, NOT simple Poisson!
4. PoissonNLLLoss assumes: y ~ Poisson(λ), but we have y ~ Poisson(Poisson(λ))

### Numerical Instability

**PoissonNLLLoss computation**:
```
NLL = pred_count - target_count * log(pred_count + eps)
```

**With our data** (max_value=150, patch size 64x64x2):
- If pred_count ≈ 150, target_count ≈ 150
- NLL per pixel ≈ 150 - 150*log(150) ≈ 150 - 750 = -600
- NLL per patch ≈ -600 * 8192 = -4,915,200
- NLL per batch (16 patches) ≈ -78,643,200

This explains the -1e10 magnitude!

---

## Why It Fails

### 1. Distribution Mismatch
- **Loss expects**: y ~ Poisson(λ), where λ is predicted by network
- **Reality**: y ~ Poisson(1x), where 1x ~ Poisson(λ_clean)
- Network cannot learn the correct inverse mapping

### 2. Information Loss
- 1x has noise variance = λ_clean
- y has additional noise from Poisson(1x)
- Total noise is NOT Poisson distributed
- Cannot be inverted using simple Poisson model

### 3. Comparison with N2N
**N2N (works)**:
- y1, y2 ~ Poisson(λ_clean) (independent samples from clean signal)
- E[y2|y1] ≈ λ_clean (denoising objective is well-defined)

**Noiser2Noise (fails)**:
- 1x ~ Poisson(λ_clean)
- y ~ Poisson(1x)
- E[1x|y] ≠ λ_clean (cannot denoise to clean signal)

---

## Possible Fixes (Not Implemented)

### Option 1: Use Lower Count as Input (Noise2Less)
Instead of Poisson sampling, artificially reduce counts:
```python
# Don't: y ~ Poisson(1x)
# Do: y = 0.5x as input, 1x as GT
lq = gt * 0.5  # Deterministic downscaling
```
- Becomes "Noise2Less-Noise" instead of "Noiser2Noise"
- May work but less theoretically motivated

### Option 2: Different Loss Function
Use a loss that accounts for compound Poisson:
- Not straightforward to derive
- Would require extensive theoretical work

### Option 3: Abandon Method
- Noiser2Noise fundamentally doesn't work for Poisson noise
- Use N2B/N2V/S2S for single-image denoising instead

---

## Recommendation

**Do NOT use Noiser2Noise for this task**

**Better alternatives**:
1. **N2N** (52.29 dB) - if paired data available (split)
2. **N2B** (47.52 dB) - best single-image method
3. **N2V/S2S** (need fixes) - alternative single-image methods

---

## Lessons Learned

1. **Poisson noise ≠ Gaussian noise**: Methods that work for Gaussian don't directly transfer
2. **Sampling from noisy data**: Creates compound distributions, not simple Poisson
3. **Theory matters**: Must match data generation process with loss function assumptions

---

**Date**: 2025-12-15
**Conclusion**: Method abandoned due to fundamental theoretical incompatibility.
