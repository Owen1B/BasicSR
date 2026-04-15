# Migration from Forked BasicSR

This package is designed to replace direct edits in `basicsr/` with a standalone recipe layer.

## What moved
- Custom `arch/data/model/loss/metric` modules are under `prime/*`.
- Domain pipelines are under `prime/pipeline/*`.
- Utility scripts are under project-root `scripts/*`.
- Train/test/inference use dedicated entrypoints (no unified compatibility CLI).

## Recommended usage
1. Keep upstream BasicSR unmodified.
2. Install this recipe package in editable mode.
3. Use explicit commands for each workflow (`train`, `test`, `inference`, utility scripts).
4. Use converged mainline options under `options/train|test/converged/`.
5. Treat `options/_legacy/` and `prime/models/_legacy/` as archive-only.

## Command mapping
- `python basicsr/train.py ...` -> `python -m prime.train ...`
- `python basicsr/test.py ...` -> `python -m prime.test ...`
- `python spect_ct/scripts/compare_two_models_mp4.py ...` -> `python scripts/compare_two_models_mp4.py ...`
- `python spect_ct/scripts/convert_proj60_to_ap.py ...` -> `python scripts/convert_proj60_to_ap.py ...`
- `python spect_ct/scripts/viz_sinogram_rows.py ...` -> `python scripts/viz_sinogram_rows.py ...`
