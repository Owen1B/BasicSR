# SPECT 自监督降噪实验

## 📋 研究目标

**仅使用 1x 噪声 SPECT 图像实现最优自监督降噪**

### 核心问题
1. 哪种自监督方法最适合 SPECT 泊松噪声？
2. 如何针对泊松噪声优化配置？
3. 关键超参数如何影响性能？

---

## 🗂️ 实验组织结构

### Tier 1: 方法主对比 (Main Comparison)
**目录**: `tier1_method_comparison/`
**目的**: 确定最佳自监督降噪方法
**实验数**: 5 个

| 配置文件 | 方法 | 输入需求 | 配置 | 预期PSNR | 理论依据 |
|---------|------|---------|------|----------|---------|
| `n2n_baseline.yml` | Noise2Noise | 配对 (split 1x → 2×0.5x) | Linear + PoissonNLL | **52-53 dB** | 参考（split 0.5x pairs） |
| `noiser2noise_baseline.yml` | **Noiser2Noise** | 单图 (1x) | Linear + PoissonNLL | **51-52 dB** | 泊松采样 + PoissonNLL |
| `n2b_baseline.yml` | Neighbor2Neighbor | 单图 (1x) | Anscombe + Charb | **51.5-52.0 dB** | 下采样近似 (-0.5~-1 dB) |
| `n2v_baseline.yml` | Noise2Void | 单图 (1x) | Anscombe + Charb | **50.5-51.5 dB** | 盲点损失 (-1~-2 dB) |
| `s2s_baseline.yml` | Self2Self | 单图 (1x) | Anscombe + Charb | **50.0-51.0 dB** | MC方差 (-1.5~-2.5 dB) |

**性能排序 (预期)**: N2N ≈ Noiser2Noise > N2B > N2V ≈ S2S

#### 理论分析

**0. Noiser2Noise** (新增方法): 泊松采样 + 理论最优损失 ⭐⭐⭐⭐⭐
- **原理**: 1x作为GT，泊松采样生成更噪版本作为输入，y ~ Poisson(λ=1x)
- **优势**: (1) 只需1x单图 (2) 可用PoissonNLL (3) 在线采样增加多样性
- **劣势**: 1x本身有噪声，不是真正的clean GT
- **预期**: 51-52 dB (可能略低于N2N split，但接近理论最优)
- **关键**: **在线采样** - 每次epoch采样不同，增加训练多样性
- **实现**: 先crop patch（计数域），再泊松采样，最后归一化

**1. N2N (Noise2Noise)**: 理论上界（但用split 0.5x） ⭐⭐⭐⭐⭐
- **原理**: 配对噪声样本 y1, y2 ~ Poisson(λ), 学习 E[y2|y1] = λ
- **优势**: 直接优化条件期望，无偏估计，充分利用配对信息
- **配置**: Linear normalization (保留Poisson特性) + PoissonNLLLoss (最大似然)
- **已验证**: PSNR=52.80 dB, SSIM=0.9983, LPIPS=0.0031 @ 50k iter
- **文献**: Lehtinen et al., ICML 2018

**2. N2B (Neighbor2Neighbor)**: 接近上界 ⭐⭐⭐⭐
- **原理**: 空间2×2下采样生成伪配对 (y_even, y_odd)，双loss策略
- **优势**: 单图训练，SPECT空间相关性强，下采样假设合理
- **劣势**: 下采样引入轻微信息损失 (~0.5-1 dB)
- **预期**: 51.5-52.0 dB (接近N2N，因SPECT平滑性强)
- **文献**: Huang et al., CVPR 2021

**3. N2V (Noise2Void)**: 次优但稳健 ⭐⭐⭐⭐
- **原理**: 盲点网络，用邻域像素预测中心，完全自监督
- **优势**: 单图训练，无需任何配对或假设
- **劣势**: 缺失中心像素信息，性能略低 (~1-2 dB)
- **预期**: 50.5-51.5 dB (SPECT邻域相关性强，损失相对小)
- **改进潜力**: 使用Linear + PoissonNLL可能接近51.5 dB (Tier 3探索)
- **文献**: Krull et al., CVPR 2019

**4. S2S (Self2Self)**: 灵活但方差大 ⭐⭐⭐
- **原理**: Dropout作为隐式mask，Monte Carlo采样 (100次)
- **优势**: 单图训练，适用于无法生成配对的场景
- **劣势**: (1) MC采样引入额外方差 (2) dropout假设独立性 (3) 推理慢
- **预期**: 50.0-51.0 dB (MC方差 + dropout不完美匹配Poisson)
- **文献**: Quan et al., NeurIPS 2020

#### 关键Insights

1. **配对 vs 单图权衡**:
   - N2N需要配对 (0.5x split) 但性能最优 (52.80 dB)
   - N2B单图训练但接近N2N (预期51.5-52.0 dB)
   - N2V/S2S单图但性能略低 (50-51.5 dB)

2. **SPECT特性优势**:
   - 空间平滑性强 → N2B下采样假设更合理
   - 邻域相关性高 → N2V盲点损失相对较小
   - 泊松噪声特性 → PoissonNLL理论最优

3. **实用性考虑**:
   - 若有配对数据 (split) → N2N (52.80 dB)
   - 若只有单图 + 追求性能 → N2B (预期51.5-52.0 dB)
   - 若只有单图 + 追求鲁棒 → N2V (预期50.5-51.5 dB)
   - 若无法生成配对/下采样 → S2S (预期50.0-51.0 dB)

---

### Tier 2: 深度消融研究 (In-depth Ablation)
**目录**: `tier2_ablation/`
**目的**: 理解关键因素，优化配置
**针对**: N2N (最佳方法)
**实验数**: ~30 个

#### 2.1 归一化 & 损失函数 (`norm/`)
**关键问题**: Linear vs Anscombe? PoissonNLL vs Charbonnier?

| 配置 | Norm | Loss | 理论匹配 | 预期性能 |
|------|------|------|----------|----------|
| `n2n_linear_poisson.yml` | Linear | PoissonNLL | ✅✅ 最优 | **51.08 dB** (baseline) |
| `n2n_anscombe_charb.yml` | Anscombe | Charbonnier | ✅ 合理 | ~50 dB |
| `n2n_anscombe_poisson.yml` | Anscombe | PoissonNLL | ⚠️ 矛盾 | ~50.5 dB |
| `n2n_linear_charb.yml` | Linear | Charbonnier | ⚠️ 次优 | ~50.2 dB |
| `n2n_linear_mse.yml` | Linear | MSE | ⚠️ 次优 | ~49.8 dB |
| `n2n_linear_l1.yml` | Linear | L1 | ⚠️ 次优 | ~49.5 dB |

**实验数**: 6

#### 2.2 正则化策略 (`regularization/`)
**关键问题**: weight decay 如何防止过拟合？

| 配置 | weight_decay | 预期 | 说明 |
|------|--------------|------|------|
| `n2n_wd_1e4.yml` | 1e-4 | 最优 | Baseline |
| `n2n_wd_0.yml` | 0 | **严重过拟合** | 验证WD重要性 |
| `n2n_wd_1e3.yml` | 1e-3 | 略差 | 正则化过强 |
| `n2n_wd_5e4.yml` | 5e-4 | 良好 | 中等正则 |
| `n2n_wd_1e5.yml` | 1e-5 | 轻微过拟合 | 正则化过弱 |

**实验数**: 5

#### 2.3 优化器配置 (`optimization/`)
**关键问题**: 学习率调度如何影响收敛？

| 配置 | Optimizer | LR | eta_min | 说明 |
|------|-----------|----|---------|----- |
| `n2n_adamw_1e4.yml` | AdamW | 1e-4 | 1e-5 | Baseline |
| `n2n_adam_1e4.yml` | Adam | 1e-4 | 1e-5 | 无内置WD |
| `n2n_adamw_5e5.yml` | AdamW | 5e-5 | 1e-5 | 更保守 |
| `n2n_adamw_5e4.yml` | AdamW | 5e-4 | 1e-5 | 更激进 |
| `n2n_etamin_1e7.yml` | AdamW | 1e-4 | 1e-7 | eta_min过小 |
| `n2n_etamin_1e6.yml` | AdamW | 1e-4 | 1e-6 | 中间值 |
| `n2n_etamin_1e4.yml` | AdamW | 1e-4 | 1e-4 | eta_min过大 |

**实验数**: 7

#### 2.4 数据策略 (`data_strategy/`)
**关键问题**: patch size & batch size 如何平衡？

| 配置 | Patch | Batch | 说明 |
|------|-------|-------|------|
| `n2n_patch64_batch16.yml` | 64 | 16 | Baseline |
| `n2n_patch128_batch16.yml` | 128 | 16 | 中等patch |
| `n2n_patch256_batch8.yml` | 256 | 8 | 大patch (显存限制) |
| `n2n_patch64_batch8.yml` | 64 | 8 | 小batch |
| `n2n_patch64_batch32.yml` | 64 | 32 | 大batch |
| `n2n_patch64_batch16_aug.yml` | 64 | 16 | + augmentation |

**实验数**: 6

#### 2.5 网络结构 (`network/`)
**关键问题**: 网络深度/宽度如何权衡？

| 配置 | nb (blocks) | nc (channels) | 说明 |
|------|-------------|---------------|------|
| `n2n_unetres_nb4.yml` | 4 | [64,128,256,512] | Baseline |
| `n2n_unetres_nb2.yml` | 2 | [64,128,256,512] | 浅网络 |
| `n2n_unetres_nb6.yml` | 6 | [64,128,256,512] | 深网络 |
| `n2n_nc_32_64_128_256.yml` | 4 | [32,64,128,256] | 窄网络 |
| `n2n_nc_96_192_384_768.yml` | 4 | [96,192,384,768] | 宽网络 |

**实验数**: 5

#### 2.6 组合验证 (`combined/`)
**关键问题**: 最优/最差配置的对比验证

| 配置 | 说明 |
|------|------|
| `n2n_best_config.yml` | 所有最优配置组合 |
| `n2n_worst_config.yml` | 所有最差配置组合 (验证分析) |
| `n2n_original_kair.yml` | 完全对齐KAIR |

**实验数**: 3

---

### Tier 3: 其他方法针对性消融 (Method-specific)
**目录**: `tier3_other_methods/`
**目的**: 验证泛化性，探索改进空间
**实验数**: ~12 个

#### 3.1 Noise2Void (`n2v/`)
**关键问题**: PoissonNLL 是否适用于 N2V？

| 配置 | Norm | Loss | 说明 |
|------|------|------|------|
| `n2v_anscombe_charb.yml` | Anscombe | Charbonnier | Baseline (原版) |
| `n2v_linear_charb.yml` | Linear | Charbonnier | 对比 |
| `n2v_anscombe_poisson.yml` | Anscombe | PoissonNLL | 实验 |
| `n2v_linear_poisson.yml` | Linear | PoissonNLL | 实验 (理论最优) |

**实验数**: 4

#### 3.2 Neighbor2Neighbor (`n2b/`)
**关键问题**: lambda 权重如何调优？

| 配置 | Norm | λ1 | λ2 | increase_ratio |
|------|------|----|----|----- |
| `n2b_anscombe_lambda_1_1.yml` | Anscombe | 1 | 1 | 2.0 (baseline) |
| `n2b_linear_lambda_1_1.yml` | Linear | 1 | 1 | 2.0 |
| `n2b_anscombe_lambda_1_2.yml` | Anscombe | 1 | 2 | 2.0 (强调reg) |
| `n2b_anscombe_lambda_2_1.yml` | Anscombe | 2 | 1 | 2.0 (强调rec) |

**实验数**: 4

#### 3.3 Self2Self (`s2s/`)
**关键问题**: dropout 率 & MC 采样数如何选择？

| 配置 | Dropout | MC Samples | 说明 |
|------|---------|------------|------|
| `s2s_dropout_0.3_mc_100.yml` | 0.3 | 100 | Baseline |
| `s2s_dropout_0.2_mc_100.yml` | 0.2 | 100 | 低dropout |
| `s2s_dropout_0.4_mc_100.yml` | 0.4 | 100 | 高dropout |
| `s2s_dropout_0.3_mc_50.yml` | 0.3 | 50 | 快速推理 |

**实验数**: 4

---

## 📊 实验总览

| Tier | 目的 | 实验数 | 预计时间 |
|------|------|--------|----------|
| Tier 1 | 方法对比 | 4 | 1 周 |
| Tier 2 | 深度消融 | ~30 | 2-3 周 |
| Tier 3 | 其他方法 | ~12 | 1 周 |
| **总计** | | **~46** | **4-5 周** |

---

## 🎯 预期贡献

### 1. 方法对比
- 首次系统对比 4 种自监督方法在 SPECT 上的表现
- 量化单图 vs 配对方法的性能差异

### 2. 泊松适配
- 证明 Linear + PoissonNLL 是理论最优配置
- 分析 Anscombe 变换的优缺点

### 3. 关键因素
- 揭示 weight decay 对防止过拟合的关键作用
- 量化 patch size / batch size 的影响

### 4. 泛化性
- 探索 PoissonNLL 在不同方法中的适用性
- 为其他泊松降噪任务提供参考

---

## 📝 相关文档

- `EXPERIMENT_DESIGN.md` - 详细实验设计
- `BASELINE_CONFIG.md` - 基准配置说明
- `../demo/KAIR_vs_BasicSR_ANALYSIS.md` - 过拟合问题分析

---

## 🚀 快速开始

### 运行单个实验
```bash
cd /home/owen/code/BasicSR
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 \
python basicsr/train.py -opt options/train/spect_selfsup/tier1_method_comparison/n2n_baseline.yml
```

### 运行批量实验
```bash
# Tier 1: 方法对比
./scripts/run_tier1_experiments.sh

# Tier 2: N2N 消融
./scripts/run_tier2_ablation.sh norm
./scripts/run_tier2_ablation.sh regularization
# ...

# Tier 3: 其他方法
./scripts/run_tier3_experiments.sh
```

### 结果分析
```bash
# 生成对比表格
python scripts/analyze_results.py --tier 1

# 可视化训练曲线
python scripts/plot_curves.py --experiments tier2_ablation/norm/*
```

---

**更新日期**: 2024-12-14
**负责人**: Owen
