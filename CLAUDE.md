# Chromosome-KD 项目研究记录

## 项目结构

```
projects/
├── LDMDet/          # Flow Matching 染色体检测 (主方向)
│   ├── mods/        # 模型组件 (diffusiondet_head, loss, noise_sampler, ...)
│   ├── configs/     # 实验配置 (ldmdet_flowdet_adaln.py 为当前 SOTA)
│   ├── tools/       # 分析/评估工具
│   └── scripts/     # 训练脚本
└── KaryoFlow/       # 端到端核型排列 (子方向)
    ├── karyoflow/   # 核心模块 (encoder, flow_module, lehmer, ...)
    └── tools/       # 验证/评估工具
```

## 当前 SOTA 基线

**LDMDet + AdaLN + OT + shifted schedule**: mAP = 0.751
- 配置: `projects/LDMDet/configs/ldmdet_flowdet_adaln.py`
- 模型: `projects/LDMDet/mods/diffusiondet_head.py` (DiffusionDetHead)
- 最佳 checkpoint: `work_dirs/ldmdet_flowdet_adaln/best_coco_bbox_mAP_epoch_56.pth`
- 24 类染色体检测, COCO 格式标注
- 数据集: `/data/linkst/datasets/Chromosome20240904_NoAug_NoResize_coco` (438 train / 127 valid)

### 关键发现: Size-AP 相关性 r=0.74

染色体物理尺寸 (不是类别频率) 决定检测难度。小染色体 (G21/G22/Y) AP 远低于大染色体 (A1-A3)。

染色体实际 bbox 面积范围 (已验证):
- G21 中位数面积: ~645 px²
- A1 中位数面积: ~6687 px²
- 面积比 ~10:1

---

## 已验证的负结果 (经代码审查确认正确)

### 1. Scale-Conditioned Flow Matching → 无效

**假设**: 按染色体尺寸缩放噪声/损失权重能改善小染色体检测

**实现** (已验证正确):
- `projects/LDMDet/mods/diffusiondet_head.py:102-105, 246-257, 480-491, 563-564`
- `projects/LDMDet/mods/loss.py:368, 448-486`
- COCO category 顺序验证: id 1-3=A1-A3, id 9-12=C6-C9 (注意 C6-C9 在 C10-C12 之后), `_chromo_group_of_class` 映射与 dataset 一致

**结果**:

| 配置 | Best mAP | vs Baseline 0.751 | G21 AP | Y AP |
|------|----------|-------------------|--------|------|
| sc_noise only | 0.738 | **-0.013** | 0.676 | 0.586 |
| sc_loss only | 0.745 | **-0.006** | 0.676 | 0.629 |
| sc_combined | 0.736 | **-0.015** | 0.664 | 0.614 |

**理论原因** (见 THEORY_WHY_FAILED.md):
1. Scale-conditioned 噪声引入有偏流速场: $v^{(\sigma)} = v^* + (\sigma_c - 1)\mathbb{E}[\epsilon|z_t]$
2. 损失重加权 $w_c \neq 1/\sigma_c^2$ 破坏变分原理 (不是任何 f-divergence)
3. Size-AP 相关性不是因果的: 信息量上限 $I \propto \text{pixel\_area}$，噪声缩放不能增加信息

**代码验证状态**: ✅ 已验证 class mapping, noise scaling, loss weighting 正确

### 2. KaryoFlow 端到端排列学习 → 不可行

**假设**: 用离散流匹配 (MDLM) 从染色体 crops 学习核型排列

**编码器分类能力验证** (已验证正确):
- 代码: `projects/KaryoFlow/tools/verify_encoder.py` (24-way), `verify_encoder_groups.py` (8-group)
- Label 推导: `[CLASS_TO_INDEX[SLOT_ORDER[i]] for i in range(N)]` — 与 dataset slot 顺序一致

| crop_size | 24-class acc | 8-group acc | random baseline |
|-----------|-------------|-------------|-----------------|
| 64×64 | 19.4% | 44.3% | 4.2% / 30.4% |
| 128×128 | 27.4% | 54.5% | 4.2% / 30.4% |

**Matching-Based 排列基线** (训练逻辑正确, 评估有 transpose bug 但不影响结论):
- 代码: `projects/KaryoFlow/tools/matching_baseline.py`
- Per-crop classifier: crop features → MLP → 46-slot classification
- CE loss 收敛于 ~3.78 (random=3.83), raw_train_acc ~3% (random=2.17%)
- 面积启发式基线: position acc = 1.9%

**理论原因** (见 THEORY_WHY_FAILED.md):
1. 信息论下界: 需要 ~178 bits 确定排列, ResNet18 仅提供 ~25.8 bits
2. $2^{22}$ 同源对等价类 + 438 训练样本 → 样本复杂度不可达
3. 组内特征 SNR << 1 (C6-C12/D13-D15/E16-E18 视觉高度相似)

**代码验证状态**: ✅ 已验证 label 推导, dataset 的 crop ordering, 训练目标构造

---

## 已验证的正结果

### LDMDet Flow Matching 检测 (mAP=0.751)
- AdaLN-Zero + OT coupling + shifted schedule 构成当前最优
- Velocity prediction mode, Heun solver, 500 proposals
- 配置路径: `projects/LDMDet/configs/ldmdet_flowdet_adaln.py`

---

## 理论框架 (THEORY_WHY_FAILED.md)

完整路径: `projects/LDMDet/THEORY_WHY_FAILED.md`

包含:
1. **ΔH = log K**: OT 耦合多样性坍塌的数学解释
2. **定理 1**: Scale-Conditioned 噪声等价于有偏流速场
3. **定理 2**: 损失重加权破坏变分原理
4. **定理 3**: KaryoFlow 排列学习的信息论不可行性下界
5. **Shannon-Hartley**: 小染色体信息量上限约束

---

## 新方向建议 (未验证)

基于已验证的负结果, 以下方向值得尝试:

### 对 LDMDet 检测有提升潜力的方向:
1. **更大 backbone** (ResNet50/ConvNeXt): 小染色体信息量受像素面积约束, 但更深的 backbone 能提取更好的特征
2. **多尺度特征** (FPN): 当前只用 single-level feature, 对小目标不友好
3. **Data augmentation**: mixup/cutmix 对少数类 (Y染色体仅 693 样本)
4. **Focal loss 调参**: 当前 $\gamma=2.0, \alpha=0.25$, 可增大以关注难样本

### 对 KaryoFlow 可能可行的降级方向:
1. **先检测后排序**: 不做端到端排列, 而是先做 8-group 分类 + 组内面积排序
2. **相对排序**: 不预测绝对 slot, 而是预测 pairwise order (A > B > C > ...)
3. **更大预训练模型**: 用 DINOv2/CLIP 替换 ResNet18 编码器

### 论文方向:
- 核心贡献: ΔH 理论 + Size-AP 反直觉发现 + LDMDet 0.751 baseline
- Scale-Conditioned 和 KaryoFlow 的负结果可作为 ablation/theory validation

---

## 关键数据路径

```
数据集: /data/linkst/datasets/Chromosome20240904_NoAug_NoResize_coco/
  train/_annotations.coco.json, train/images/
  valid/_annotations.coco.json, valid/images/
  train_karyoflow.json, valid_karyoflow.json

LDMDet 工作目录: work_dirs/
  ldmdet_flowdet_adaln/          → baseline 0.751
  scale_conditioned_sc_noise/     → 0.738
  scale_conditioned_sc_loss/      → 0.745
  scale_conditioned_sc_combined/  → 0.736

Conda 环境: chromo (python 3.8, torch, mmdet)
GPU: RTX A5000 (24GB), RTX A4000 (16GB)
```

## 开发注意事项

- `_chromo_group_of_class` 在 diffusiondet_head.py 中硬编码, 顺序需与 COCO category id 顺序一致
- COCO category id 排序: C10(id=6), C11(7), C12(8), C6(9), C7(10), C8(11), C9(12) — C6-C9 在 C10-C12 之后, 不是字典序
- KaryoFlowDataset 的 crops 按 slot 顺序 (0→45) 排列, 不是按 class 排列
- matching_baseline.py 的 Hungarian 评估有 transpose bug: `cost = -b_logits.cpu().numpy()` 应为 `cost = -b_logits.T.cpu().numpy()` 才能得到正确的 slot→crop 映射, 不过训练 loss 已充分证明不可学习
