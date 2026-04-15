# Script I/O Reference

说明：以下均为 `PRIME/scripts/` 下脚本；公共实现位于 `prime/pipeline/*`。

## async_eval_checkpoints.py
- 作用：异步轮询新 checkpoint，并独立调用 `prime.test` 做重评估。
- 输入：`--test-opt`、`--models-dir`、轮询参数、可选 `--param-key-g`。
- 触发策略：`--every-n-iters`（例如每 1000 step 评测一次）。
- 防堆积：`--backlog-policy latest|window|all`（推荐 `latest`）与 `--max-pending`。
- 失败控制：`--max-retries-per-ckpt` + `--retry-exhausted-policy`。
- 并发控制：单飞锁（默认 `<models-dir>/.async_eval.lock`）；上一个未完成时，新触发自动无效退出。
- 输出：评估日志、状态文件（默认 `<models-dir>/.async_eval_state.json`）。

## check_mainline_integrity.py
- 作用：检查主线配置/注册是否引入 legacy 模型类型、旧导入或非主线 dataset type。
- 输入：仓库内固定路径（`options/train|test/{converged,debug}` 与 `prime/bootstrap.py`）。
- 输出：终端检查报告；发现违规时返回非零退出码。

## compare_two_models_mp4.py
- 作用：两模型投影/重建对比，生成 MP4 与可选中间结果。
- 输入：两组模型配置与 ckpt、病人数据或 `--input-proj`。
- 输出：对比 MP4、可选投影 dat、重建结果与汇总图。

## convert_proj60_to_ap.py
- 作用：将 `(views,H,W)` 投影转为 `(2,H,W)` 的 A/P 通道。
- 输入：`--input` dat、`--input-shape`、`--input-dtype`。
- 输出：`--output` dat（`int16` 或 `float32`）。

## eval_poisson_calibration.py
- 作用：评估模型输出与泊松噪声假设的一致性。
- 输入：配置、ckpt、投影样本集合。
- 输出：分桶统计 CSV、校准曲线图、摘要文本。

## gen_3proj3recon_gif.py
- 作用：生成 2x3（投影+重建）可视化 GIF/MP4。
- 输入：患者数据、模型配置、ckpt、重建参数。
- 输出：GIF/MP4 与对应中间数据文件。

## _legacy/nema_reconstruct_and_visualize.py
- 作用：NEMA 数据重建与可视化（已归档）。
- 输入：NEMA 数据目录、par/orbit/atten、迭代参数。
- 输出：重建 dat、旋转 GIF、过程日志。

## plot_counts_change_boxplot.py
- 作用：单病例多剂量计数变化箱线图。
- 输入：模型 A/B、thin 因子、病例投影。
- 输出：箱线图 PNG、可选缓存数据。

## plot_existing_patients_proj_recon_summary.py
- 作用：汇总多病例投影/重建变化并导出统计。
- 输入：已有结果根目录。
- 输出：汇总 PNG 与 CSV。

## precompute_umap_aux.py
- 作用：预计算 μ-map 辅助缓存。
- 输入：μ-map 数据目录与角度/模式参数。
- 输出：`npz/npy` 缓存文件。

## precompute_validation_bm3d.py
- 作用：预计算验证集 thinning + BM3D 缓存。
- 输入：病例目录、seed、thin 因子、BM3D 参数。
- 输出：`bm3d_cache/` 下缓存文件。

## run_patient_report.py
- 作用：病人级报告总流程（多模型、图表、汇总）。
- 输入：病人集合、模型组、路径参数。
- 输出：病例报告目录（图、表、可选 GIF）。

## toolbox.py
- 作用：交互/命令行一体化入口（list/run/batch/recon-results/gif-results）。
- 输入：实验目录、ckpt、病人范围、重建与可视化参数。
- 输出：`outputs/<exp>/...` 下投影、重建、GIF、CSV 汇总。

## viz_sinogram_rows.py
- 作用：按行生成 sinogram 可视化。
- 输入：投影 dat 或病人名、行索引参数。
- 输出：sinogram PNG，或可选 `sinograms_all.npy`。
