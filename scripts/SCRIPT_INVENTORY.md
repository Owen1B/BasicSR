# Script Inventory

## Mainline (actively maintained)
- `async_eval_checkpoints.py`: asynchronous heavy evaluation, non-blocking training.
- `check_mainline_integrity.py`: guard script to prevent legacy model types/imports/dataset types leaking into mainline.
- `compare_two_models_mp4.py`: two-model visualization entrypoint.
- `run_compare_two_models_pipeline.sh`: batch wrapper for two-model comparison.
- `precompute_validation_bm3d.py`: precompute validation BM3D caches.
- `precompute_umap_aux.py`: precompute μ-map auxiliary caches.
- `convert_proj60_to_ap.py`: convert 60-view projections to AP format.
- `eval_poisson_calibration.py`: Poisson calibration analysis.
- `plot_counts_change_boxplot.py`: counts-change plot for comparison runs.
- `plot_existing_patients_proj_recon_summary.py`: patient-level summary plots/CSV.
- `run_patient_report.py`: patient-level report pipeline.
- `gen_3proj3recon_gif.py`: quick GIF wrapper.
- `viz_recon_atten_align_gif.py`: reconstruction/attenuation alignment visualization.
- `viz_sinogram_rows.py`: sinogram row diagnostics.
- `sync_spect_experiment_to_remote.sh`, `sync_basicsr_to_remote.sh`: sync helpers.
- `toolbox.py`: interactive orchestrator.

## Legacy (archived)
- `scripts/_legacy/*`: historical `spect_selfsup` experiment runners and checks.
- `scripts/_legacy/nema_reconstruct_and_visualize.py`: NEMA reconstruction pipeline (archived).
- `scripts/_legacy/monitor_nema.sh`: NEMA monitor script (archived).
- Archived scripts are not part of converged mainline and may depend on `options/_legacy/*`.
