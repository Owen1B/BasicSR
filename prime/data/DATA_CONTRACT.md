# Data Contract (Mainline)

本项目主线数据是“无头二进制 `.dat`”，统一按三维体解析：`(V, H, W)`。

## 1. 文件级格式
- 文件类型：raw binary `.dat`（无 header）。
- 元素个数必须等于：`views * height * width`。
- 元素类型：`uint16 | int16 | float32`（由 YAML `dtype` 指定）。
- 读入后统一转 `float32` 进入后续流程。

如果元素个数不匹配，数据集会直接报错并终止。

## 2. YAML 是唯一配置入口
训练/测试数据组织全部由 YAML 驱动，关键字段：
- `type`: `SPECTTrainDataset` 或 `SPECTProjectionEvalDataset`
- `dataroot_*` / `pattern_*`: 文件定位
- `views/height/width/dtype`: 二进制解析契约
- `input_layout`: `ap2 | volume`
- `view_sampler.mode`: `single_view | opposite_pair | n_views`
- `pack_mode`: `2d_channel | 3d_depth`
- `mode`: `paired | poisson | poisson_thinning`

## 3. 张量输出契约
`SPECTTrainDataset`：
- `2d_channel` -> `lq/gt` 形状 `(C,H,W)`
- `3d_depth` -> `lq/gt` 形状 `(1,D,H,W)`

`SPECTProjectionEvalDataset`：
- 返回完整 `proj_sequence`，形状 `(V,H,W)`，用于轻量验证/推理。

## 4. 6 模式映射
- `planar_2d_1view`: `ap2 + single_view + 2d_channel`
- `planar_2d_2view`: `ap2 + opposite_pair + 2d_channel`
- `tomo_2d_1view`: `volume + single_view + 2d_channel`
- `tomo_2d_2view`: `volume + opposite_pair(offset=V/2) + 2d_channel`
- `tomo_2d_nview`: `volume + n_views + 2d_channel`
- `tomo_3d_nview`: `volume + n_views + 3d_depth`

## 5. 约束与建议
- `opposite_pair` 语义要求 `views` 为偶数（常见 60）。
- U-Net 路径建议 `H/W` 可被 8 整除。
- 同一 dataloader/batch 内 shape 必须一致。
