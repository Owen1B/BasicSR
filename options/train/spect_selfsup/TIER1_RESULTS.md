# Tier 1: Self-Supervised Denoising Methods - Results Analysis

**Date**: 2025-12-15
**Dataset**: SPECT XCAT Poisson 1x

---

## 📊 Final Results Summary

| Method | PSNR (dB) | SSIM | LPIPS ↓ | Poisson Fit ↓ | Iters | Status |
|--------|-----------|------|---------|---------------|-------|--------|
| **N2N** | **52.29** | **0.9980** | **0.0044** | **0.0136** | 50k | ✅ Best |
| **N2B** | **47.52** | **0.9922** | **0.0262** | 0.0275 | 20k | ✅ Good |
| Noiser2Noise | 36.05 | 0.8765 | 0.4232 | 0.9476 | 50k | ❌ Failed |
| N2V | 36.10 | 0.8775 | 0.4226 | 0.8246 | 20k | ⚠️ Poor |
| S2S | 36.08 | 0.8762 | 0.4239 | 0.6162 | 20k | ⚠️ Poor |

---

## 🔍 Key Findings

### 1. N2N (Noise2Noise) - Reference Method ✅

**Config**: Linear norm + PoissonNLLLoss
**Input**: Split 1x → 2×0.5x pairs
**Iterations**: 50,000

**Results**:
- PSNR: 52.29 dB (Best)
- SSIM: 0.9980
- LPIPS: 0.0044
- Poisson Fit: 0.0136

**Conclusion**: ✅ **Works as expected**. Sets the upper bound for performance.

---

### 2. N2B (Neighbor2Neighbor) - Strong Second ✅

**Config**: Anscombe norm + Charbonnier loss
**Input**: Single 1x image (spatial downsampling)
**Iterations**: 20,000

**Results**:
- PSNR: 47.52 dB (-4.77 dB vs N2N)
- SSIM: 0.9922
- LPIPS: 0.0262
- Poisson Fit: 0.0275

**Conclusion**: ✅ **Excellent performance for single-image method**. Only 4.77 dB gap from N2N despite using no paired data. **Best practical method for single 1x images**.

---

### 3. Noiser2Noise - CRITICAL FAILURE ❌

**Config**: Linear norm + PoissonNLLLoss
**Input**: Single 1x as GT, Poisson sample as input
**Iterations**: 50,000

**Results**:
- PSNR: 36.05 dB (❌ -16.24 dB vs N2N!)
- SSIM: 0.8765
- LPIPS: 0.4232
- Poisson Fit: 0.9476

**CRITICAL ISSUE**:
- Training loss: `-1.0479e+10` (abnormally large negative value)
- Expected loss: `-2` to `-5` (like N2N)
- Loss magnitude is **10 billion times** larger than expected!

**Suspected Root Causes**:
1. **Online Poisson sampling bug**: May generate extreme values
2. **PoissonNLLLoss numerical instability**: Loss computation overflow
3. **Normalization mismatch**: Count domain → normalize order issue
4. **Lambda parameter issue**: Poisson sampling from noisy 1x instead of clean signal

**Action Required**: 🚨 **Debug Noiser2Noise implementation immediately**

---

### 4. N2V (Noise2Void) - Underperforming ⚠️

**Config**: Anscombe norm + Charbonnier loss
**Input**: Single 1x image (blind-spot network)
**Iterations**: 20,000

**Results**:
- PSNR: 36.10 dB (❌ -16.19 dB vs N2N!)
- SSIM: 0.8775
- LPIPS: 0.4226
- Poisson Fit: 0.8246

**Issues**:
- Performance much worse than expected (should be ~50-51 dB)
- Similar to Noiser2Noise (36 dB), suggesting systematic issue
- Only 20k iterations (may need 50k like N2N/Noiser2Noise)

**Hypotheses**:
1. **Insufficient training**: 20k iter not enough (N2B got 47.52 dB with 20k)
2. **Loss function**: Charbonnier may not be optimal for Poisson noise
3. **Network architecture**: Blind-spot network may need tuning

---

### 5. S2S (Self2Self) - Underperforming ⚠️

**Config**: Anscombe norm + Charbonnier loss
**Input**: Single 1x image (Dropout + MC sampling)
**Iterations**: 20,000

**Results**:
- PSNR: 36.08 dB (❌ -16.21 dB vs N2N!)
- SSIM: 0.8762
- LPIPS: 0.4239
- Poisson Fit: 0.6162

**Issues**: Same as N2V

---

## 📈 Performance Ranking

**By PSNR**:
1. N2N: 52.29 dB (needs paired data)
2. **N2B: 47.52 dB** (single image) ⭐ **Best single-image method**
3. N2V: 36.10 dB ⚠️
4. S2S: 36.08 dB ⚠️
5. Noiser2Noise: 36.05 dB ❌

**Performance Gaps**:
- N2N → N2B: -4.77 dB (acceptable for single-image)
- N2B → N2V: -11.42 dB (❌ too large!)
- N2B → Noiser2Noise: -11.47 dB (❌ critical failure!)

---

## 🎯 Conclusions

### What Worked ✅

1. **N2N**: Baseline works perfectly (52.29 dB)
2. **N2B**: Excellent single-image method (47.52 dB with only 20k iter)
3. **Training infrastructure**: Resume, wandb logging, metrics all working

### Critical Issues ❌

1. **Noiser2Noise**: Complete failure (36.05 dB, loss = -1e10)
   - Theory is sound, implementation has severe bug
   - Must debug before proceeding

2. **N2V/S2S**: Severe underperformance (36 dB vs expected 50-51 dB)
   - May need more iterations (50k instead of 20k)
   - Or fundamental configuration issue

### Theory vs Reality

**Expected**:
- N2N: 52-53 dB ✓ (got 52.29)
- Noiser2Noise: 51-52 dB ✗ (got 36.05)
- N2B: 51.5-52.0 dB ✗ (got 47.52, but still good)
- N2V: 50.5-51.5 dB ✗ (got 36.10)
- S2S: 50.0-51.0 dB ✗ (got 36.08)

---

## 🚨 Action Items

### Priority 1: Fix Noiser2Noise (Critical)

1. Debug PoissonNLLLoss computation
2. Check Poisson sampling implementation
3. Verify normalization order (sample → normalize vs normalize → sample)
4. Add gradient clipping if needed

### Priority 2: Investigate N2V/S2S

1. Extend training to 50k iterations
2. Try Linear + PoissonNLLLoss instead of Anscombe + Charbonnier
3. Check if blind-spot network/dropout is properly implemented

### Priority 3: Tier 2 Ablations

**Should we proceed with Tier 2?**
- ✅ Yes for N2N/N2B (working well)
- ❌ No for Noiser2Noise/N2V/S2S (need fixes first)

**Recommendation**:
- Run Tier 2 ablations for **N2N only** (norm, loss, regularization, etc.)
- Fix Noiser2Noise/N2V/S2S before their Tier 3 ablations

---

## 📝 Notes

- All experiments completed without crashes
- Resume functionality works correctly
- Wandb logging successful
- **Tier 2 configs ready** (32 configs generated for N2N ablations)

**Next steps**: Debug Noiser2Noise, then decide on N2V/S2S strategy.
