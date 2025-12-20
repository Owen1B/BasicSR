# 真实骨 SPECT 投影数据去噪实验

## 数据集

### 来源
- **数据集**: bone_20230327
- **病人数**: 30 人
- **采集时间**: 20s (高质量参考)
- **投影格式**: 60 视角 × 128×128 像素

### 数据提取
```bash
python3 spect_ct/scripts/prepare_bone_projection_dataset.py \
  --input datasets/bone_20230327 \
  --output datasets/bone_projection_20s \
  --time 20s-1 \
  --max-value 60
```

### 提取后的数据格式
- **总数据量**: 900 对
- **数据形状**: (2, 128, 128) - 双通道 (anterior + posterior)
- **数据类型**: float32
- **值范围**: 0 ~ 60 (P99.9)
- **平均值**: 2.33

#### 视角分离
- **Anterior (前视角)**: Detector 1, angles 0° to 174° (30 个视角)
- **Posterior (后视角)**: Detector 0, angles -180° to -6° (30 个视角)

每个病人的 60 个视角被分离为 30 对双通道图像。

### 数据分割
- **训练集**: 810 对 (90%, 27 个病人)
- **验证集**: 90 对 (10%, 3 个病人)

## 实验配置

### ⚠️ 重要更新：所有配置已适配 SPECT3DModel

所有配置文件现在使用 `SPECT3DModel` 而不是 `SRModel`，支持：
- **自动 GIF 生成**：验证时自动生成 6 视图对比 GIF（原始 | BM3D | Denoised(g) | Denoised(ema) | Residual | Residual Anscombe）
- **BM3D 缓存**：BM3D 降噪结果自动缓存，避免重复计算
- **自动检测**：自动识别 3D 数据（投影序列）并生成 GIF，2D 数据使用标准验证流程

详见：`README_3D_VALIDATION.md`

### ✅ 推荐：n2n_bone_proj_20s_singleview_sota.yml（单通道，SOTA-like）

**核心设计**：
- 在线 Poisson 采样：将 20s (1x) 分裂为两个 0.5x 进行配对训练
- **单通道输入/输出**：每次只训练一个视角（anterior 或 posterior）
- 128×128 单视角去噪（更贴近你现有的单视角最佳实验设置）

**最佳配置**（基于 XCAT 实验结果）：
- **归一化**: Linear (max_value=60)
- **损失函数**: PoissonNLLLoss
- **网络**: UNetRes (in_nc=1, out_nc=1, nc=[64,128,256,512], nb=4)
- **优化器**: AdamW (lr=1e-4, weight_decay=1e-4)
- **学习率**: CosineAnnealing (eta_min=1e-5)
- **数据策略**: patch64, batch32, enlarge_ratio=4
- **无数据增强**: 解剖结构有方向性

#### 单通道样本数量
我们保留每个文件内部的 (anterior, posterior) 两视角存储，但在数据加载时用 `view_mode=split1` 拆成单通道样本：
- 900 个 `.dat` 文件 → **1800 个单通道样本**

## 运行训练

### 单 GPU 训练
```bash
# 推荐用 conda KAIR 环境（已包含 torch + CUDA）
cd /home/owen/code/BasicSR
conda run -n KAIR bash -lc 'PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/bone_real/n2n_bone_proj_20s_singleview_sota.yml'
```

### 恢复训练
```bash
PYTHONPATH="./:${PYTHONPATH}" \
CUDA_VISIBLE_DEVICES=0 \
python basicsr/train.py -opt options/train/bone_real/n2n_bone_proj_20s.yml \
  --auto_resume
```

## 实验结果

### 预期性能
基于 XCAT 实验（Linear + PoissonNLL）:
- **PSNR**: ~53.9 dB (XCAT 理想数据)
- **SSIM**: ~0.98

真实数据可能略低：
- **PSNR**: ~48-52 dB (预估)
- **SSIM**: ~0.92-0.96

### 训练时间
- **每次迭代**: ~0.1s (RTX 3090)
- **每个 epoch**: ~10s (101 iters)
- **总训练时间**: ~14 小时 (50k iters)

## 数据特点

### 优势
✅ **真实临床数据**：实际病人扫描
✅ **真实 Poisson 噪声**：物理噪声分布
✅ **在线采样**：更多训练样本多样性
✅ **双通道设计**：与 XCAT 实验一致

### 挑战
⚠️ **无绝对真值**：只能用 N2N 自身验证
⚠️ **病人间差异**：解剖结构和注射剂量不同
⚠️ **稀疏数据**：中位数为 0，65% 像素为背景

## 后续实验

### 1. 不同采集时间
```bash
# 2s 数据（极低计数，高噪声）
python3 spect_ct/scripts/prepare_bone_projection_dataset.py \
  --input datasets/bone_20230327 \
  --output datasets/bone_projection_2s \
  --time 2s-10 \
  --max-value 15

# 4s 数据（低计数）
python3 spect_ct/scripts/prepare_bone_projection_dataset.py \
  --input datasets/bone_20230327 \
  --output datasets/bone_projection_4s \
  --time 4s-5 \
  --max-value 25
```

### 2. 其他自监督方法
- **Noise2Void**: 单图像盲点网络
- **Neighbor2Neighbor**: 空间下采样
- **Self2Self**: Dropout + MC 采样

### 3. 重建后去噪
对比投影域去噪 vs 图像域去噪效果

## 参考

- XCAT 实验结果: `experiments/n2n_linear_poisson/`
- 数据集分析: `spect_ct/docs/BONE_DATASET_ANALYSIS.md`
- 提取脚本: `spect_ct/scripts/prepare_bone_projection_dataset.py`

## 备注：双通道版本（可选）
如果你将来想尝试把 anterior+posterior 作为 2 通道联合输入，可以使用：
- `options/train/bone_real/n2n_bone_proj_20s.yml`

---

**创建时间**: 2025-12-18
**数据版本**: bone_20230327



