# Data Modules (Mainline)

主线收敛为 1 个训练数据集 + 1 个验证/推理数据集：

1. `SPECTTrainDataset`
- 统一输入：先读成 `(V,H,W)`。
- 统一职责：在 dataset 内完成 view 采样、Poisson thinning 分裂、归一化、打包。
- 采样模式：
  - `single_view`：逐视角
  - `opposite_pair`：前后位对视角
  - `n_views`：任意 n 个视角
- 打包模式：
  - `2d_channel`：输出 `(C,H,W)`
  - `3d_depth`：输出 `(1,D,H,W)`

2. `SPECTProjectionEvalDataset`
- 用于 3D 路径轻量验证/推理加载，直接返回原始 `(V,H,W)` 投影序列。

## 主线映射
- `planar_2d_1view`：`input_layout=ap2` + `single_view + 2d_channel(C=1)`
- `planar_2d_2view`：`input_layout=ap2` + `opposite_pair + 2d_channel(C=2)`
- `tomo_2d_1view`：`input_layout=volume` + `single_view + 2d_channel(C=1)`
- `tomo_2d_2view`：`input_layout=volume` + `opposite_pair(offset=V/2) + 2d_channel(C=2)`
- `tomo_2d_nview`：`input_layout=volume` + `n_views + 2d_channel(C=n)`
- `tomo_3d_nview`：`input_layout=volume` + `n_views + 3d_depth(D=n)`

历史实验数据加载器已归档到 `prime/data/_legacy/`，不在主线注册。

数据格式与 YAML 字段契约见：
- `prime/data/DATA_CONTRACT.md`
