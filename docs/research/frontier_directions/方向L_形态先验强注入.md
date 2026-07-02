# 方向 L：形态先验强注入 (Morphology Prior Injection)

> **目标**: 用自监督预训练的染色体形态编码器, 把着丝粒/臂长/带型等领域先验强注入检测头, 用 metric learning 替代 softmax 分类, 攻克 43.4% 的同组形态混淆误差。
>
> **理论依据**:
> - SimCLR/MoCo: 自监督表示学习
> - Metric Learning: 人脸识别 (ArcFace, CircleLoss)
> - 染色体形态学: 着丝粒位置 (centromeric index), 臂长比
> - 方向 C (ShapeAttention) 的演进: 从局部注意力到全形态编码
>
> **当前代码位置**:
> - [ldmdet/core/shape_attention.py](../../ldmdet/core/shape_attention.py) — 局部形状注意力 (方向 C1)
> - [ldmdet/criterion/contrastive_loss.py](../../ldmdet/criterion/contrastive_loss.py) — 对比损失 (方向 C2)

---

## 1. 背景与动机

### 1.1 当前形态学习的局限

方向 C 已尝试 ShapeAttention + ContrastiveLoss, 但:
- ShapeAttention 只在 RoI 特征上做局部注意力, **未显式建模形态学特征**
- ContrastiveLoss 是辅助损失, 主分类仍是 softmax
- 模型从 CNN 特征隐式学形态, **未利用领域知识**

### 1.2 染色体形态学的强先验

染色体分类的**医学标准**就是形态学:

| 组 | 特征 | centromeric index |
|----|------|-------------------|
| A (1-3) | 大, 中着丝粒 | 0.45-0.50 |
| B (4-5) | 大, 亚中着丝粒 | 0.25-0.40 |
| C (6-12,X) | 中, 中着丝粒 | 0.40-0.50 |
| D (13-15) | 中, 近端着丝粒 | 0.15-0.25 |
| E (16-18) | 小, 中/亚中 | 0.30-0.45 |
| F (19-20) | 短, 中着丝粒 | 0.40-0.50 |
| G (21-22,Y) | 短, 近端着丝粒 | 0.15-0.25 |

**关键洞察**: 同组内分类靠的是**臂长细微差异**, 这是 CNN 难以捕获的。

### 1.3 形态先验强注入的核心思想

```
当前: RoI 特征 → softmax 分类 (隐式学形态)
创新: 形态编码器 (自监督) → metric learning (显式用形态)
```

三步:
1. **自监督预训练**: 在无标注染色体图上学形态表示 (着丝粒、臂长)
2. **强注入检测头**: 形态 embedding 作为检测头的额外输入
3. **Metric 分类**: 用形态度量替代 softmax, 同组内对比

---

## 2. 子方向说明

### L1: 自监督形态编码器预训练

**问题**: CNN 从头学形态, 数据有限 (4680 张), 难以捕获细粒度。

**方案**: 自监督预训练 (SimCLR/MoCo), 数据增强模拟形态变化。

**实现**:
```python
class ChromoMorphologyEncoder(nn.Module):
    """染色体形态自监督编码器"""
    def __init__(self, backbone='resnet18', feat_dim=256):
        super().__init__()
        self.backbone = build_backbone(backbone)
        self.projection = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, feat_dim)
        )

    def forward(self, x):
        feat = self.backbone(x).flatten(1)
        return self.projection(feat)

# SimCLR 训练
def simclr_loss(z1, z2, temperature=0.5):
    """对比损失: 同染色体不同增强应相似"""
    z = torch.cat([z1, z2])
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
    labels = torch.arange(z1.shape[0], device=z.device)
    return F.cross_entropy(sim / temperature, labels)

# 数据增强 (形态保持)
morphology_augs = [
    RandomRotation(180),        # 旋转 (染色体方向无关)
    RandomFlip(),               # 水平/垂直翻转
    ColorJitter(brightness=0.2),# 染色强度变化
    # 不用: 强烈形变 (会破坏形态)
]
```

**预训练数据**: 所有染色体裁剪图 (无标签), ~215k 实例 (4680 图 × 46 条)。

**复杂度**: 中 — 预训练需额外 1-2 天, 但下游收益大。

**预期收益**: 下游分类 mAP +2-4%。

### L2: 形态 Embedding 强注入检测头

**问题**: 检测头只用 RoI 特征, 未用显式形态信息。

**方案**: 冻结预训练编码器, 把 embedding 注入检测头。

**实现**:
```python
class MorphologyInjectedHead(nn.Module):
    """形态注入的检测头"""
    def __init__(self, feat_channels=256, morph_dim=256):
        super().__init__()
        # 冻结的预训练形态编码器
        self.morph_encoder = load_pretrained_encoder()
        for p in self.morph_encoder.parameters():
            p.requires_grad_(False)

        # 融合: RoI 特征 + 形态 embedding
        self.fusion = nn.Sequential(
            nn.Linear(feat_channels + morph_dim, feat_channels),
            nn.ReLU(),
            nn.Linear(feat_channels, feat_channels)
        )

    def forward(self, roi_features, roi_images):
        # roi_features: (N, C, H, W) — 标准 RoI 特征
        # roi_images: (N, 3, h, w) — RoI 对应的原图裁剪
        morph_emb = self.morph_encoder(roi_images)  # (N, morph_dim)
        roi_feat = roi_features.flatten(2).mean(-1)  # (N, C)

        # 强融合
        fused = self.fusion(torch.cat([roi_feat, morph_emb], dim=-1))
        return fused
```

**复杂度**: 低 — 编码器冻结, 只加融合层。

**预期收益**: mAP +1-2% (显式形态信息)。

### L3: Metric Learning 分类 (替代 Softmax)

**问题**: Softmax 分类无组内对比, 同组混淆严重。

**方案**: 用 ArcFace/CircleLoss 的 metric learning, 学类内紧凑、类间分离的 embedding。

**实现**:
```python
class MetricClassifier(nn.Module):
    """Metric Learning 分类头 (ArcFace 风格)"""
    def __init__(self, feat_channels=256, num_classes=24, s=30, m=0.5):
        super().__init__()
        # 类别中心 (可学习)
        self.class_centers = nn.Parameter(torch.randn(num_classes, feat_channels))
        nn.init.normalize_(self.class_centers)
        self.s = s  # 缩放因子
        self.m = m  # 角度 margin

    def forward(self, features, labels=None):
        # 归一化
        features = F.normalize(features, dim=-1)
        centers = F.normalize(self.class_centers, dim=-1)

        # ArcFace: 角度 + margin
        if labels is not None:
            cos = features @ centers.t()  # (N, num_classes)
            cos_target = cos[range(N), labels]
            cos_target = cos_target - self.m  # 角度 margin
            cos[range(N), labels] = cos_target
            logits = cos * self.s
        else:
            logits = (features @ centers.t()) * self.s
        return logits

# Loss: ArcFace + 同组对比
def metric_loss(logits, labels, group_ids):
    loss_arcface = F.cross_entropy(logits, labels)
    # 同组内 contrastive (C9 vs C10 等)
    loss_group = group_contrastive(features, group_ids)
    return loss_arcface + λ * loss_group
```

**复杂度**: 中 — 需调整 margin 和缩放因子。

**预期收益**: 同组分类误差 -30-50%, mAP +2-3%。

### L4: 显式形态学特征工程 (可选)

**问题**: 纯学习可能忽略已知形态学指标。

**方案**: 手工提取 centromeric index 等, 作为额外特征。

**实现**:
```python
def extract_morphology_features(box_image):
    """手工形态学特征 (基于传统 CV)"""
    # 1. 骨架化
    skeleton = cv2.ximgproc.thinning(box_image)
    # 2. 着丝粒检测 (最窄点)
    widths = [cv2.countNonZero(skeleton[y]) for y in range(h)]
    centromere_y = np.argmin(widths)
    # 3. centromeric index = p_arm / total
    p_arm = centromere_y
    q_arm = h - centromere_y
    ci = p_arm / (p_arm + q_arm)
    # 4. 臂长比
    arm_ratio = p_arm / q_arm
    # 5. 总长度, 弯曲度
    total_length = ...
    curvature = ...
    return [ci, arm_ratio, total_length, curvature]
```

**复杂度**: 高 (需传统 CV 工程), 但可解释性强。

**预期收益**: 与 L3 互补, mAP +0.5-1%。

---

## 3. 预训练数据与流程

### 3.1 数据准备

```python
# 从训练集裁剪所有染色体实例
def extract_chromosome_crops(dataset):
    crops = []
    for img, ann in dataset:
        for box in ann.bboxes:
            crop = img[box.y1:box.y2, box.x1:box.x2]
            crops.append(crop)
    return crops  # ~215k 实例
```

### 3.2 预训练流程

```bash
# Step 1: 裁剪实例
python tools/extract_chromosome_crops.py --out data/chromo_crops/

# Step 2: SimCLR 预训练 (8 GPU, 100 epoch)
python tools/pretrain_morphology_encoder.py \
    --data data/chromo_crops/ \
    --method simclr \
    --epochs 100 \
    --backbone resnet18

# Step 3: 下游检测训练 (加载预训练)
python experiments/runners/train.py \
    experiments/configs/ldmdet/direction_l_morphology.py \
    --work-dir work_dirs/direction_l
```

---

## 4. 文件组织

```
ldmdet/
├── models/
│   ├── morphology_encoder.py    # L1: 自监督编码器
│   ├── injected_head.py         # L2: 形态注入头
│   └── metric_classifier.py     # L3: Metric 分类
├── criterion/
│   └── arcface_loss.py          # ArcFace 损失
├── tools/
│   ├── extract_chromosome_crops.py
│   └── pretrain_morphology_encoder.py
└── tests/
    └── test_direction_l.py
```

---

## 5. 测试计划

```python
class TestMorphologyEncoder:
    def test_simclr_pretraining(self):
        """对比损失应使同染色体不同增强相似"""
    def test_feature_invariance(self):
        """旋转/翻转不变性"""

class TestInjectedHead:
    def test_morphology_gradient_flow(self):
        """形态 embedding 应影响分类 (梯度)"""
    def test_encoder_frozen(self):
        """预训练编码器应冻结"""

class TestMetricClassifier:
    def test_intra_class_compact(self):
        """类内 embedding 应紧凑"""
    def test_inter_class_separated(self):
        """类间 (同组) 应分离"""
    def test_arcface_margin(self):
        """ArcFace margin 应增大类间距离"""
```

---

## 6. 实验计划

### 6.1 主实验

| 实验 | 配置 | 预期 |
|------|------|------|
| L1-pretrain | SimCLR 预训练 | 形态表示质量验证 |
| L2-inject | 注入检测头 | mAP +1-2% |
| L3-metric | ArcFace 分类 | 同组误差 -30-50% |
| L-all | 全部组合 | mAP +3-5% |

### 6.2 消融

| 消融 | 验证 |
|------|------|
| 有/无预训练 | L1 贡献 |
| 冻结 vs 微调编码器 | 编码器策略 |
| Softmax vs ArcFace | L3 贡献 |
| margin m = 0.1/0.3/0.5/0.7 | ArcFace 超参 |
| 同组 contrastive 有/无 | 组对比贡献 |

### 6.3 训练命令

```bash
# L1 预训练
python tools/pretrain_morphology_encoder.py --epochs 100

# L-all 下游训练
python experiments/runners/train.py \
    experiments/configs/ldmdet/direction_l_morphology.py \
    --work-dir work_dirs/direction_l --seed 42
```

---

## 7. 风险与缓解

### 7.1 预训练数据不足

**风险**: 215k 实例可能不够 SimCLR。

**缓解**:
- 用 ImageNet 预训练 backbone 初始化
- 强增强增加有效数据
- 或用 MoCo (无需大 batch)

### 7.2 形态编码器与 RoI 特征冗余

**风险**: 形态 embedding 可能与 RoI 特征重叠。

**缓解**: 用正交损失约束两者互补:
```python
loss_orth = (roi_feat @ morph_emb.t()).abs().mean()  # 鼓励正交
```

### 7.3 ArcFace 超参敏感

**风险**: margin m 和缩放 s 难调。

**缓解**:
- 先用 s=30, m=0.5 (人脸识别经验值)
- 网格搜索 m ∈ {0.1, 0.3, 0.5, 0.7}

---

## 8. 成功指标

| 指标 | baseline | 目标 |
|------|---------|------|
| mAP | 0.858 | +0.03~0.05 |
| 同组分类误差 | 43.4% | -30-50% |
| C9/C10 混淆 | Top 1 | 降到 Top 5 外 |
| 预训练表示线性探测 | - | >70% |

---

## 9. 实现路线图

1. **Phase 1 (3 周)**: L1 — 自监督预训练编码器
2. **Phase 2 (2 周)**: L2 — 形态注入检测头
3. **Phase 3 (3 周)**: L3 — Metric learning 分类
4. **Phase 4 (可选, 2 周)**: L4 — 手工形态特征

---

## 10. 参考文献

- SimCLR: https://arxiv.org/abs/2002.05709
- MoCo: https://arxiv.org/abs/1911.05722
- ArcFace: https://arxiv.org/abs/1801.07698
- CircleLoss: https://arxiv.org/abs/2002.10857
- 染色体形态学: ISCN 2020
