# Tier 1 Methods - Critical Fixes Required

## Summary

Only **N2N** (52.29 dB) trained successfully. All other methods failed or underperformed:
- **N2B** (47.52 dB @ 3.4k): Overfitting + reg loss explosion
- **N2V** (36.10 dB): Used wrong model type
- **S2S** (36.08 dB): Self-prediction collapsed to identity

---

## Fix 1: N2V - Change Model Type ✅ DONE

**Problem**: Used `SRModel` which doesn't support masked loss
**Result**: Network learned output = input (identity mapping)

**Fix**:
```yaml
# options/train/spect_selfsup/tier1_method_comparison/n2v_baseline.yml
model_type: Noise2VoidModel  # Changed from SRModel
```

**Why**: N2V requires computing loss only on masked pixels. `SRModel` computes loss on all pixels, so network learns to copy input.

---

## Fix 2: N2B - Reduce Reg Loss Growth

**Problem**: Regularization loss exploded from 4e-7 → 2e-3 (5000x growth)

**Root cause**:
```python
Lambda = current_iter / total_iter * increase_ratio
# At iter 20k: Lambda = 20000/20000 * 2.0 = 2.0
# Reg loss dominates reconstruction loss
```

**Proposed fixes** (choose one):

### Option A: Reduce increase_ratio (Recommended)
```yaml
train:
  increase_ratio: 0.5  # Changed from 2.0
  # Lambda will grow from 0 → 0.5 instead of 0 → 2.0
```

### Option B: Increase total_iter (if training longer)
```yaml
train:
  total_iter: 50000
  # Lambda grows slower: 50000/50000 * 2.0 = 2.0 but over longer time
```

### Option C: Add early stopping
- Best PSNR was at iter 3400
- Could add `early_stopping_patience` or manually stop at peak

**Recommendation**: Try Option A first (increase_ratio: 0.5)

---

## Fix 3: S2S - Strengthen Dropout or Abandon

**Problem**: Network collapsed to identity (output = input)
**Loss evolution**: 3.5e-3 → 4e-6 (converged to perfect reconstruction)

**Root cause**:
- S2S trains to predict noisy from noisy (self-prediction)
- This is inherently an identity task
- Dropout (0.3) not strong enough to prevent collapse

**Proposed fixes** (in order of likelihood to help):

### Option A: Increase dropout rate
```yaml
network_g:
  dropout_rate: 0.5  # Changed from 0.3
```

### Option B: Reduce network capacity
```yaml
network_g:
  nc: [32, 64, 128, 256]  # Changed from [64, 128, 256, 512]
```
Smaller network = harder to memorize identity

### Option C: Add stronger regularization
```yaml
train:
  optim_g:
    weight_decay: !!float 5e-4  # Changed from 1e-4
```

### Option D: Abandon S2S for this task
- S2S inherently difficult for Poisson noise
- MC sampling (100 forward passes) very slow
- Performance ceiling likely lower than N2B/N2V

**Recommendation**: Try dropout_rate=0.5 first, but S2S may not be suitable for this data.

---

## Priority

1. **N2V**: ✅ Fixed (model_type change)
2. **N2B**: High priority (reduce increase_ratio to 0.5)
3. **S2S**: Low priority (may abandon if fix doesn't work)

---

## Expected Results After Fixes

| Method | Current | Expected After Fix |
|--------|---------|-------------------|
| N2N | 52.29 dB | - (reference) |
| N2B | 47.52 dB @ 3.4k | **49-50 dB @ 50k** (no overfitting) |
| N2V | 36.10 dB | **49-51 dB @ 50k** (with correct model) |
| S2S | 36.08 dB | **45-48 dB** (if dropout fix works) or abandon |

---

**Date**: 2025-12-15
**Recommendation**: Fix N2V + N2B first, re-run. Decide on S2S later.
