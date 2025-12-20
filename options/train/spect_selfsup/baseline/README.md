# SPECT Self-Supervised Denoising Baseline Configurations

本目录包含所有自监督去噪方法的 baseline 配置文件。

## 📁 配置文件

| 文件 | 方法 | 网络 | 说明 |
|------|------|------|------|
| `n2n_baseline.yml` | Noise2Noise | UNetRes | N2N baseline，使用 binomial split |
| `n2v_baseline.yml` | Noise2Void | DBSNl | N2V 使用 blind-spot 网络 |
| `n2b_baseline.yml` | Neighbor2Neighbor | UNetRes | N2B 使用空间下采样 |
| `n2n_fpn_fusion.yml` | N2N + FPN Fusion | UNetFPNFusion | 多尺度特征融合网络 |

## 🎯 Baseline 配置参数

所有 baseline 使用统一的最优参数：

| 参数 | 值 | 说明 |
|------|-----|------|
| `patch_size` | 64 | 训练 patch 大小 |
| `batch_size` | 32 | 批大小 |
| `norm_type` | linear | 归一化方式 |
| `loss` | PoissonNLLLoss | 损失函数 |
| `max_value` | 150.0 | 最大计数值 |
| `lr` | 1e-4 | 学习率 |
| `weight_decay` | 1e-4 | 权重衰减（防止过拟合） |
| `eta_min` | 1e-5 | 学习率调度器最小值 |
| `total_iter` | 50000 | 总迭代次数 |
| `augmentation` | 关闭 | SPECT 有解剖方向 |

## 🚀 运行实验

### 运行所有 baseline

```bash
cd /home/owen/code/BasicSR

# 先运行测试确保代码正确
python scripts/test_code_fixes.py

# 运行 N2N baseline
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 \
  python basicsr/train.py \
  -opt options/train/spect_selfsup/baseline/n2n_baseline.yml \
  --auto_resume

# 运行 N2V baseline
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 \
  python basicsr/train.py \
  -opt options/train/spect_selfsup/baseline/n2v_baseline.yml \
  --auto_resume

# 运行 N2B baseline
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 \
  python basicsr/train.py \
  -opt options/train/spect_selfsup/baseline/n2b_baseline.yml \
  --auto_resume

# 运行 FPN Fusion
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 \
  python basicsr/train.py \
  -opt options/train/spect_selfsup/baseline/n2n_fpn_fusion.yml \
  --auto_resume
```

## 📊 预期性能

| 方法 | 预期 PSNR | 说明 |
|------|-----------|------|
| N2N (binomial split) | 52-54 dB | 参考基准，使用 0.5x pairs |
| N2V (DBSNl) | 50-52 dB | Blind-spot 网络，理论上限较低 |
| N2B | 51-52 dB | 利用空间相关性 |
| FPN Fusion | 52-54 dB | 多尺度特征融合 |

## 🔧 代码修复说明

### 1. DBSNl Blind-Spot 网络 (新增)

- 文件: `basicsr/archs/dbsnl_arch.py`
- 基于 AP-BSN 论文的 blind-spot 网络实现
- 使用 `CentralMaskedConv2d` 确保网络看不到中心像素
- 使用 dilated convolution 扩大感受野

### 2. UNetFPNFusion 网络 (新增)

- 文件: `basicsr/archs/unet_fpn_fusion_arch.py`
- FPN 风格的多尺度特征融合
- 双视角独立编码 + 多尺度融合 + 共享解码

### 3. PoissonNLLLoss (修复)

- 文件: `basicsr/losses/poisson_loss.py`
- 支持 Anscombe 归一化的计数域损失计算
- 移除不必要的裁剪，保持梯度流
- GPU 友好的 PyTorch 实现

### 4. Noise2VoidBlindSpotModel (新增)

- 文件: `basicsr/models/noise2void_model.py`
- 专门为 blind-spot 网络设计的 N2V 模型
- 不需要数据 mask，更简洁高效

## ⚠️ 注意事项

1. **运行前先测试**：`python scripts/test_code_fixes.py`
2. **不要修改 baseline 参数**：消融实验应该只改变一个参数
3. **使用 linear 归一化**：实验证明 linear + PoissonNLL 最优
4. **禁用数据增强**：SPECT 有解剖方向，增强会损害性能

## 📈 消融实验规划

详见 `../tier2_ablation/` 目录中的各个消融实验配置。

所有消融实验都基于 `n2n_baseline.yml`，每个实验只改变一个参数。

---

**最后更新**: 2024-12-20

