# SPECT/CT 工具与结果归档（BasicSR）

这个目录用于**集中存放所有 SPECT/CT 相关脚本与产出结果**，避免散落在 `scripts/`、`datasets/` 或 `~` 目录中。

## 目录结构

- **`spect_ct/scripts/`**：SPECT/CT 相关脚本（后续新增脚本也放这里）
- **`spect_ct/docs/`**：分析文档/说明文档
- **`spect_ct/results/`**：脚本运行生成的图片/统计结果（png/csv/json 等）
- **`spect_ct/web/`**：网页可视化（Three.js 查看器等）
  - **`spect_ct/web/viewer_8080/`**：8080 版本（你当前主用的）
  - **`spect_ct/web/viewer_8081/`**：专业版（可选）

## 兼容性说明（旧路径仍可用）

为了不影响你之前的使用方式，我在原位置保留了**软链接**：

- `scripts/visualize_bianchengming.py` → `spect_ct/scripts/visualize_bianchengming.py`
- `scripts/prepare_web_data.py` → `spect_ct/scripts/prepare_web_data.py`
- `scripts/generate_viewer.py` → `spect_ct/scripts/generate_viewer.py`
- `scripts/attenuation_correction_demo.py` → `spect_ct/scripts/attenuation_correction_demo.py`
- `scripts/verify_poisson_projections.py` → `spect_ct/scripts/verify_poisson_projections.py`
- `scripts/explain_sinogram_vs_projection.py` → `spect_ct/scripts/explain_sinogram_vs_projection.py`
- `scripts/generate_professional_viewer.py` → `spect_ct/scripts/generate_professional_viewer.py`

文档也同样保留软链接：

- `datasets/BianChengMing_ANALYSIS.md` → `spect_ct/docs/BianChengMing_ANALYSIS.md`
- `datasets/BianChengMing_SINOGRAM_POISSON.md` → `spect_ct/docs/BianChengMing_SINOGRAM_POISSON.md`

## 网页查看器（8080 版本）

当前网页文件已归档在：

- `spect_ct/web/viewer_8080/`

启动方式：

```bash
cd /home/owen/code/BasicSR/spect_ct/web/viewer_8080
python3 -m http.server 8080
```

然后浏览器打开：`http://localhost:8080`

## 结果文件

现有结果图已统一拷贝到：

- `spect_ct/results/`

例如：
- `projection_comparison.png`
- `noise_realizations.png`
- `attenuation_correction_analysis.png`
- `projection_poisson_verification.png`
- `sinogram_vs_projection_explanation.png`

## 后续约定（重要）

从现在开始：

- 新脚本放 **`spect_ct/scripts/`**
- 任何生成的图/表/中间数据默认输出到 **`spect_ct/results/`**
- 网页相关输出放 **`spect_ct/web/`**





