# SPECT Recipe Scripts

本目录对齐 Real-ESRGAN 的组织方式：
- `prime/` 只放可复用库代码。
- `scripts/` 放一次一用或分析型入口脚本。
- 训练/测试/推理主入口不在此目录，而是：
  - `python -m prime.train`
  - `python -m prime.test`
  - `python -m prime.infer`

## 脚本分层
- 训练后分析与对比：`compare_two_models_mp4.py`、`plot_counts_change_boxplot.py`、`plot_existing_patients_proj_recon_summary.py`
- 数据处理：`convert_proj60_to_ap.py`、`precompute_validation_bm3d.py`、`precompute_umap_aux.py`
- 可视化与报告：`viz_sinogram_rows.py`、`gen_3proj3recon_gif.py`、`run_patient_report.py`
- 异步验证：`async_eval_checkpoints.py`（支持按步触发、防堆积、单飞锁；上一个没结束时新触发无效）
- 主线守卫：`check_mainline_integrity.py`（检查主线配置/注册中没有 legacy 模型与 dataset 类型）
- 交互式总控：`toolbox.py`
- 统计评估：`eval_poisson_calibration.py`

## 约定
- 脚本应尽量“薄”：只做参数解析与调度，核心逻辑放 `prime/pipeline/*`。
- 读写 dat、推理加载、MP4/GIF 渲染等公共能力统一复用 `pipeline` 模块，不重复实现。
- 默认输出目录使用 `outputs/`（可通过参数覆盖）。
- 历史实验脚本/配置统一归档到 `_legacy`，不作为主线入口。
- NEMA 专项脚本已归档到 `scripts/_legacy/`。

详细 I/O 见 `SCRIPT_IO_REFERENCE.md`。
