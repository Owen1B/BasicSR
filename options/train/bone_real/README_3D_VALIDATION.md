# 3D 数据降噪模型验证 GIF 生成

## 概述

`SPECT3DModel` 是专门为 3D 数据降噪（如 SPECT 投影序列）设计的模型类，在验证时自动生成 5 视图对比 GIF，方便视觉评估。

## 使用方法

### 1. 修改配置文件

将 `model_type` 从 `SRModel` 改为 `SPECT3DModel`：

```yaml
model_type: SPECT3DModel  # 使用 SPECT3DModel 而不是 SRModel
```

### 2. 配置验证数据集

在 `val` 部分添加 GIF 生成选项：

```yaml
val:
  name: Bone_val_n2n_single
  type: SPECTDatPairedDataset
  # ... 其他数据集配置 ...

  # 启用 3D 数据的 GIF 生成
  generate_gif: true
  gif_output_dir: experiments/{name}/visualization/gifs
  max_gifs_per_val: 3  # 每次验证最多生成 3 个 GIF（避免 I/O 过多）
```

### 3. 自动检测 3D 数据

模型会自动检测以下情况为 3D 数据：
- 文件路径包含 `ProjectionImage` 或 `projection`
- 数据集类型为 `SPECTDatPairedDataset` 且模式为 `n2n` 或 `poisson`

对于 2D 数据，会使用标准的 `SRModel` 验证流程（不生成 GIF）。

## GIF 内容

生成的 GIF 包含 6 个视图：
1. **Original** - 原始投影
2. **BM3D** - BM3D 降噪结果（Anscombe + BM3D + 逆 Anscombe）
3. **Denoised (g)** - 深度学习降噪结果（net_g）
4. **Denoised (ema)** - 深度学习降噪结果（net_g_ema）
5. **Residual** - 残差图（红=正，蓝=负）
6. **Residual Anscombe** - Anscombe 域的残差图

### BM3D 缓存机制

BM3D 降噪结果会自动缓存到：
```
experiments/{exp_name}/cache/bm3d/{ProjectionImage*}_bm3d.npy
```

- **首次运行**：计算 BM3D 降噪并保存到缓存（可能需要几分钟）
- **后续运行**：直接从缓存加载，无需重复计算
- **缓存键**：基于输入文件名（如 `ProjectionImage1_bm3d.npy`）

如果输入数据未改变，BM3D 结果只需计算一次，后续验证会直接使用缓存。

每帧底部显示：
- **Step**: 训练步数（如 "Step: 28,000"）
- **View**: 视角编号（如 "View 00/59"）
- **Angle**: 角度（如 "Angle -180°"）

## 配置示例

完整示例见：`n2n_bone_proj_20s_singleview_sota_tv_val_example.yml`

## 注意事项

1. **性能**: GIF 生成会增加验证时间，建议设置 `max_gifs_per_val` 限制数量
2. **存储**: GIF 文件较大，注意磁盘空间
3. **数据格式**: 目前支持 `ProjectionImage*.dat` 格式（uint16, 60×128×128）
4. **2D vs 3D**: 2D 数据会自动跳过 GIF 生成，使用标准验证流程

## 与 SRModel 的区别

| 特性 | SRModel | SPECT3DModel |
|------|---------|--------------|
| 2D 数据验证 | ✅ 标准流程 | ✅ 标准流程 |
| 3D 数据验证 | ❌ 无 GIF | ✅ 自动生成 GIF |
| 模型类型 | `SRModel` | `SPECT3DModel` |

## 输出位置

GIF 文件保存在：
```
experiments/{exp_name}/visualization/gifs/{img_name}_iter{current_iter}_{net_tag}.gif
```

例如：
```
experiments/n2n_bone_proj_20s_singleview_sota_tv/visualization/gifs/ProjectionImage1_iter28000_ema.gif
```

