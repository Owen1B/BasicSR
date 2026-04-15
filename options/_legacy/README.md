# Legacy Options Archive

This directory contains historical experiment configs that are **not part of the active mainline**.

## Why archived
- They depend on legacy model types (`SPECT3DModel`, `SPECT3DConsistencyModel`, `Noise2VoidModel`, `Neighbor2NeighborModel`, etc.).
- Mainline has been converged to `SPECTModel` only.
- Archived configs are kept for reference/reproducibility, not for routine training/testing.

## Mainline options
- Train: `options/train/converged/*.yml`
- Test: `options/test/converged/*.yml`
- Debug smoke: `options/train/debug/*.yml`, `options/test/debug/*.yml`

