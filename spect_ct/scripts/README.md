## spect_ct/scripts 脚本清单（推荐入口）

这一目录历史上积累了不少一次性脚本。下面是**建议保留/主要使用**的“主入口”，其余脚本多为分析/调试/历史可视化。

### 1) 3proj + 3recon（2x3 GIF + counts.txt）【主入口】
- **脚本**: `generate_3projections_3recons_gif.py`（单病人 wrapper；核心实现复用 `spect_ct/pipeline/threeproj3recon.py`）
- **功能**:
  - 原始投影 / 降噪投影 / 降噪+泊松投影（第一行）
  - 三者 OSEM 重建 MIP（第二行）
  - 每个病人输出 `counts.txt`（投影域&重建域总计数 + 百分比变化）
  - 支持断点续跑：`--skip-existing`（若该病人的最终 GIF + counts.txt 已存在则跳过）
  - 批处理推荐用 `run_patient_report.py --patients-all ... --start-index/--max-patients ...`（已不再需要 `.sh`）

### 2) recon vs atten（2x2 GIF）
已移除（不再维护该逻辑）

### 3) 衰减 PostAtten 单独 GIF（1x1）
已移除（不再维护该逻辑）

### 4) 训练/损失相关 Debug
- **`debug_lowdose_loss_scale.py`**: 快速验证 low-dose loss 数值量级与 factor
- **`debug_projection_background_stats.py`**: 背景 ROI 统计（原始/降噪/泊松采样）

### 5) 病人级“报告/对比”一站式入口（新增）
- **`run_patient_report.py`**:
  - 指定 ckpt/config + 病人列表
  - 生成 2x3 GIF（含 counts.txt）
  - 生成重建中间切片图（original/denoised/denoised_poisson）
  - 生成衰减中间切片图
  - 支持多模型对比（输出“多模型 denoised 重建切片对比图”）
  - 可选：`--merge-2x3-gif` 把多个模型的 2x3 GIF 拼成一个大对比 GIF
  - 可选：`--patients-all` 扫描全量病人；`--patients-file` 从文件读取；`--start-index/--max-patients` 分段跑
  - **统一输出路径**：默认写到 `spect_ct/results/<exp-name>/...`（见 `--results-root/--exp-name`）

### 6) 泊松拟合评测（新增）
- **`eval_poisson_fit.py`**:
  - 评测 Var(y-λ̂) vs mean(λ̂)（分桶 + 拟合）
  - 输出 plot + csv + summary（默认写到 `spect_ct/results/<exp-name>/poisson_fit/`）

### 7) 从 counts.txt 汇总统计并出图（新增）
- 已合并到 `run_patient_report.py`：
  - 在同一次选择的病人范围内汇总：`--summary`
  - 输出位置：`<out-root>/_summary/<tag>/counts_activity_analysis.csv + PNG`

### 8) 数据处理（新增）
- **`convert_60views_to_ap.py`**:
  - 把 (60,128,128) 转换成前/后位两通道 (2,128,128)
  - 默认输出到 `spect_ct/results/<exp-name>/converted_ap/`

（已清理：历史分析/调试/遗留 wrapper 已删除，只保留推荐入口。）

