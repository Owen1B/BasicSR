# ============================================================================
# KAIR vs BasicSR 配置对比分析 - N2N 实验
# ============================================================================
# 问题: BasicSR 实验中 PSNR/SSIM 先涨后降 (过拟合现象)
# 目标: 找出关键差异，创建对齐配置
# ============================================================================

## 配置对比表

| 参数 | KAIR (正常) | BasicSR (过拟合) | 影响程度 | 分析 |
|------|------------|-----------------|---------|------|
| **归一化** | Linear (max: 150) | Anscombe (max: 100) | ⭐⭐⭐ | Anscombe可能引入高计数失真 |
| **Loss函数** | PoissonNLL | CharbonnierLoss | ⭐⭐⭐⭐ | Charb在Anscombe域不匹配泊松 |
| **Optimizer** | **AdamW** | Adam | ⭐⭐⭐⭐⭐ | **缺少weight decay!** |
| **Weight Decay** | **1e-4** | **0** | ⭐⭐⭐⭐⭐ | **过拟合主因!** |
| **Scheduler** | CosineAnnealingLR | CosineAnnealingRestartLR | ⭐⭐ | 相似，但... |
| **eta_min** | **1e-5** | **1e-7** | ⭐⭐⭐⭐ | **太小破坏已学模式!** |
| **Patch Size** | 64 | 256 | ⭐⭐⭐⭐ | 大patch易过拟合 |
| **Batch Size** | 16 | 8 | ⭐⭐⭐ | 小batch梯度噪声大 |
| **Total Iter** | 50000 | 20000 | ⭐⭐ | 20k可能已过拟合 |
| **LR** | 1e-4 | 1e-4 | ✓ | 相同 |
| **EMA** | 0.999 | 0.999 | ✓ | 相同 |
| **Init** | Orthogonal (0.2) | Default | ⭐⭐ | 可能影响稳定性 |
| **AMP** | bfloat16 | Disabled | ⭐ | 速度差异 |
| **Seed** | 20020113 | 0 | - | 可复现性 |

---

## 🚨 过拟合主要原因 (按重要性排序)

### 1. **缺少 Weight Decay** ⭐⭐⭐⭐⭐ (最关键!)
- **KAIR**: `AdamW` + `wd: 1e-4`
- **BasicSR**: `Adam` + `wd: 0`
- **影响**: 没有权重正则化，网络参数不受约束，容易记忆训练数据

### 2. **eta_min 过小** ⭐⭐⭐⭐
- **KAIR**: `1e-5`
- **BasicSR**: `1e-7` (小100倍!)
- **影响**: 训练后期学习率过低但仍在更新，微小更新可能破坏已学到的平滑模式

### 3. **Patch Size 过大** ⭐⭐⭐⭐
- **KAIR**: `64×64`
- **BasicSR**: `256×256` (16倍面积!)
- **影响**:
  - 大patch包含更多全局信息，更容易记忆特定模式
  - 每个epoch实际看到的patch数量更少（更少的数据多样性）
  - 缺少patch-level的随机性正则化

### 4. **Loss + Norm 组合不匹配** ⭐⭐⭐
- **KAIR**: `Linear + PoissonNLL` (完美匹配泊松统计)
- **BasicSR**: `Anscombe + Charbonnier` (假设高斯噪声)
- **影响**:
  - Anscombe变换目的是稳定方差，让泊松噪声→高斯噪声
  - 但Charbonnier是robust L1，不是真正的高斯likelihood
  - PoissonNLL直接优化泊松log-likelihood，理论更优

### 5. **Batch Size 较小** ⭐⭐⭐
- **KAIR**: `16`
- **BasicSR**: `8`
- **影响**: 梯度估计噪声更大，训练不稳定

### 6. **网络初始化** ⭐⭐
- **KAIR**: Orthogonal (gain: 0.2)
- **BasicSR**: Default (可能是Xavier)
- **影响**: 初始化影响训练初期动态，可能影响最终收敛点

---

## 📊 训练曲线分析

### PSNR/SSIM 先涨后降的典型模式:
```
Iter    PSNR    SSIM    现象
--------------------------------
0-5k    ↑↑      ↑↑      快速学习降噪
5k-10k  ↑       ↑       持续改进
10k-15k ↑(慢)   ↑(慢)   接近最优
15k-20k ↓       ↓       开始过拟合!
```

### 原因:
1. **前期 (0-10k)**: 学习通用降噪模式，泛化良好
2. **中期 (10k-15k)**: 优化细节，逐渐记忆训练集特征
3. **后期 (15k-20k)**:
   - 没有weight decay，参数无约束增长
   - eta_min=1e-7还在微调，破坏已学模式
   - 大patch + 小batch，持续记忆训练集

---

## 🔧 修复方案

### 方案A: 完全对齐 KAIR (推荐用于验证BasicSR框架)
```yaml
# 完全模仿KAIR配置
normalization: linear (max: 150)
loss: PoissonNLLLoss
optimizer: AdamW (lr: 1e-4, wd: 1e-4)
scheduler: CosineAnnealingLR (T_max: 50000, eta_min: 1e-5)
patch_size: 64
batch_size: 16
total_iter: 50000
init: orthogonal (gain: 0.2)
```

### 方案B: 保持Anscombe，修复关键问题
```yaml
# 保留Anscombe变换，但修复过拟合
normalization: anscombe
loss: CharbonnierLoss  # 或改为PoissonNLL
optimizer: AdamW (lr: 1e-4, wd: 1e-4)  # 添加weight decay!
scheduler: CosineAnnealingRestartLR (eta_min: 1e-5)  # 提高eta_min
patch_size: 64  # 减小patch
batch_size: 16  # 增大batch
total_iter: 50000
```

### 方案C: 增强正则化
```yaml
# 激进的正则化
optimizer: AdamW (lr: 1e-4, wd: 1e-3)  # 更强的weight decay
dropout: 0.1  # 添加dropout
patch_size: 64
batch_size: 32  # 更大batch
data_augmentation: true  # 启用augmentation
```

---

## 📋 消融实验建议 (Ablation Study)

基于对齐配置 (方案A)，逐个测试每个因素的影响：

```
1. base_aligned          - 完全对齐KAIR的配置 (baseline)
2. ablation_wd_0         - 去掉weight decay (wd: 0)
3. ablation_etamin_1e7   - 降低eta_min (1e-7)
4. ablation_patch_256    - 增大patch (256)
5. ablation_batch_8      - 减小batch (8)
6. ablation_anscombe     - 改用anscombe归一化
7. ablation_charbonnier  - 改用Charbonnier loss
8. ablation_combined     - 组合多个负面因素
```

每个实验运行 50000 步，对比验证集 PSNR/SSIM 曲线。

---

## 🎯 预期结果

如果诊断正确，应该看到：
1. **base_aligned**: PSNR/SSIM 持续增长，不过拟合
2. **ablation_wd_0**: 后期明显下降 (验证weight decay的作用)
3. **ablation_etamin_1e7**: 轻微下降或震荡
4. **ablation_patch_256**: 中等程度过拟合
5. **ablation_combined**: 严重过拟合 (复现当前问题)

---

## 💡 长期建议

1. **标准化配置模板**: 为所有SPECT实验创建统一的base配置
2. **Early Stopping**: 添加基于验证集的early stopping
3. **正则化策略**:
   - 始终使用 weight decay (1e-4 ~ 1e-3)
   - eta_min 不要低于 1e-5
   - 优先使用小patch (64) + 大batch (16+)
4. **监控指标**: 同时监控训练和验证指标，及时发现过拟合

---

## ✅ 实锤证据 (Experimental Evidence)

基于实际实验结果的过拟合证据：

### 1. Loss函数 + 归一化组合的影响 ⚠️ **实锤**

| 实验 | 配置 | 最佳 PSNR | 最佳 Iter | 最终 PSNR | 最终 Iter | 下降 | 状态 |
|------|------|-----------|-----------|-----------|-----------|------|------|
| `n2n_anscombe_charb` | Anscombe + Charbonnier | **51.89 dB** | 13000 | **50.17 dB** | 40000 | **-1.72 dB** | ⚠️ **严重过拟合** |
| `n2n_linear_charb` | Linear + Charbonnier | **52.01 dB** | 13000 | **50.18 dB** | 13000 | **-1.83 dB** | ⚠️ **严重过拟合** |
| `n2n_linear_poisson` | Linear + PoissonNLL | **53.91 dB** | 48000 | **53.81 dB** | 48000 | **-0.10 dB** | ✅ **正常** |

**结论**:
- ✅ **实锤1**: `Anscombe + Charbonnier` 组合导致严重过拟合（下降 **1.72 dB**）
- ✅ **实锤2**: `Linear + Charbonnier` 也导致过拟合（下降 **1.83 dB**）
- ✅ **实锤3**: `Linear + PoissonNLL` 不会过拟合，持续改进到 48k iter

**关键发现**:
- **CharbonnierLoss 是问题根源**：无论配 Anscombe 还是 Linear，都会导致过拟合
- **PoissonNLLLoss 是正确选择**：与 Linear 归一化完美匹配，持续改进
- **Anscombe 归一化本身可能不是问题**，但配 Charbonnier 会失败

### 2. 其他配置因素的影响

| 实验 | 配置 | 最佳 PSNR | 状态 |
|------|------|-----------|------|
| `n2n_patch256_batch8` | Patch256 + Batch8 | 52.50 dB @ 50002 iter | ✅ 无明显过拟合 |
| `n2n_patch64_batch32` | Patch64 + Batch32 | 52.82 dB @ 50002 iter | ✅ 正常 |
| `n2n_wd_0` | No Weight Decay | 52.90 dB @ 50002 iter | ✅ 无明显过拟合 |

**注意**: 这些实验使用了 `Linear + PoissonNLL`，所以即使没有 weight decay 或使用大 patch，也没有明显过拟合。这说明 **Loss 函数的选择比正则化更重要**。

### 3. 综合结论

**导致过拟合的配置（实锤）**:
1. ⚠️ **Anscombe + Charbonnier** → 下降 1.72 dB
2. ⚠️ **Linear + Charbonnier** → 下降 1.83 dB

**不会过拟合的配置（实锤）**:
1. ✅ **Linear + PoissonNLL** → 持续改进到 48k iter（最佳 53.91 dB）

**建议**:
- ❌ **避免使用 CharbonnierLoss**（无论配什么归一化）
- ✅ **优先使用 Linear + PoissonNLLLoss 组合**
- ✅ 如果必须使用 Anscombe，应该配 PoissonNLLLoss（但需要验证）

### 4. 损失函数消融实验（完整对比）

**实验位置**: `options/train/spect_selfsup/tier2_ablation/norm/`

| 损失函数 | 归一化 | eps | 最佳 PSNR | 最佳 Iter | 最终 PSNR | 性能排名 | 状态 |
|---------|--------|-----|-----------|-----------|-----------|---------|------|
| **PoissonNLLLoss** | Linear | 1e-9 | **53.91 dB** | 48000 | 53.81 dB | 🥇 1 | ✅ 最佳 |
| **MSELoss** | Linear | N/A | **53.72 dB** | 40000 | 53.66 dB | 🥈 2 | ✅ 接近最佳 |
| **L1Loss** | Linear | N/A | **52.04 dB** | 12000 | 50.15 dB | 🥉 3 | ⚠️ 早期收敛 |
| **CharbonnierLoss** | Linear | **1e-12** | **52.01 dB** | 13000 | 50.18 dB | 4 | ❌ 严重过拟合 |
| **CharbonnierLoss** | Anscombe | **1e-12** | **51.89 dB** | 13000 | 50.17 dB | 5 | ❌ 严重过拟合 |

**关键发现**:
1. **PoissonNLLLoss 最优**（53.91 dB），持续改进到 48k iter
2. **MSELoss 表现很好**（53.72 dB，仅 -0.19 dB 差距），是很好的替代选择
3. **L1Loss 早期收敛**（12k iter），但最终也下降（-1.89 dB）
4. **CharbonnierLoss 无论配什么归一化都过拟合**

**关于 eps 参数**:
- **BasicSR CharbonnierLoss 默认**: `eps=1e-12`
- **KAIR PoissonNLLLoss 使用**: `eps=1e-9`（这是 PoissonNLLLoss 的 eps，不是 CharbonnierLoss）
- **当前 CharbonnierLoss 实验**: `eps=1e-12`

**⚠️ eps 参数可能不是问题根源**:
- CharbonnierLoss 的 eps 主要影响数值稳定性（防止 sqrt(0)）
- 过拟合的根本原因是 **CharbonnierLoss 不适合泊松噪声**（假设高斯噪声）
- 即使调整 eps，CharbonnierLoss 仍然不匹配泊松统计特性

**建议的进一步实验**（可选）:
- 测试 `CharbonnierLoss + eps=1e-9` 或 `eps=1e-6` 看是否改善
- 但预期改善有限，因为根本问题是损失函数与噪声分布不匹配

---
