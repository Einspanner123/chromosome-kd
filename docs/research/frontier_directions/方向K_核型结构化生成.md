# 方向 K：核型结构化生成 (Karyotype-Structured Generation)

> **目标**: 把染色体检测从"46 个独立框预测"重构为"一个核型结构 → 解码为 46 个框", 用图神经网络建模框间关系, 攻克同组形态混淆 (43.4% 分类误差)。
>
> **理论依据**:
> - Graph R-CNN: Yang et al. (ECCV 2018) — 关系建模
> - DETR 全局注意力: Carion et al. (ECCV 2020)
> - Human Pose Estimation: 结构化输出预测
> - 染色体核型学: ISCN 国际标准 (46 条, 23 对, 7 组 A-G + X/Y)
>
> **当前代码位置**:
> - [ldmdet/core/head.py](../../ldmdet/core/head.py) — 独立框预测
> - [ldmdet/criterion/criterion.py](../../ldmdet/criterion/criterion.py) — 独立分类损失

---

## 1. 背景与动机

### 1.1 染色体检测的特殊性被忽视

当前模型把染色体当**独立目标**检测, 但染色体是**核型结构**:

```
核型 = {
    A组: {1, 2, 3}      — 大, 中着丝粒
    B组: {4, 5}          — 大, 亚中着丝粒
    C组: {6-12, X}       — 中, 中着丝粒 (X 同 C 组)
    D组: {13, 14, 15}    — 中, 近端着丝粒
    E组: {16, 17, 18}    — 小, 中/亚中着丝粒
    F组: {19, 20}        — 短, 中着丝粒
    G组: {21, 22, Y}     — 短, 近端着丝粒 (Y 同 G 组)
}
```

**关键约束**:
- 每图正好 46 条 (23 对)
- 同组形态相似 → 当前模型 C10↔C9, B5→B4 混淆
- 组内有序 (1 < 2 < 3 按长度)

### 1.2 独立预测的问题

当前: `pred = [box_1, box_2, ..., box_46]` (独立预测每个框)

**问题**:
- 无组关系建模 → 同组混淆无法消除
- 无数量约束 → 可能预测 45 或 47 条
- 无全局一致性 → 无法利用"必有 46 条"的强先验

### 1.3 结构化生成的核心思想

创新: 预测一个**核型结构图** → 解码为 46 个框

```
输入图像 → backbone → 全局特征
                      ↓
            核型图 G = (V, E)
            V: 46 个节点 (每个对应一条染色体)
            E: 组内/组间关系边
                      ↓
            GNN 消息传递 (建模组关系)
                      ↓
            解码: 每个节点 → (box, class)
```

**优势**:
- 组内对比: C 组节点共享组 embedding, 天然区分 6-12
- 数量约束: 图固定 46 节点, 不会多/少
- 全局一致: GNN 保证 46 个预测相互协调

---

## 2. 子方向说明

### K1: 核型图构建与 GNN 关系建模

**问题**: 独立预测无组关系, 同组混淆。

**方案**: 用 GNN 在 46 个 proposal 间建模组关系。

**实现**:
```python
class KaryotypeGNN(nn.Module):
    """核型图神经网络"""
    def __init__(self, feat_channels=256, num_groups=7, num_classes=24):
        super().__init__()
        # 组 embedding (A-G + X + Y)
        self.group_embeddings = nn.Embedding(num_groups, feat_channels)
        # 类别 → 组映射 (ISCN 标准)
        self.class_to_group = {
            1,2,3: 0,  # A
            4,5: 1,    # B
            6,7,8,9,10,11,12: 2,  # C (含 X)
            13,14,15: 3,  # D
            16,17,18: 4,  # E
            19,20: 5,     # F
            21,22: 6,     # G (含 Y)
        }

        # GNN 层
        self.gnn_layers = nn.ModuleList([
            GraphTransformerLayer(feat_channels) for _ in range(3)
        ])

    def forward(self, node_features, adjacency):
        """
        node_features: (N, C) — N=46 节点特征
        adjacency: (N, N) — 边权重 (同组高, 异组低)
        """
        for layer in self.gnn_layers:
            node_features = layer(node_features, adjacency)
        return node_features
```

**复杂度**: 中 — GNN 参数量小, 但需构建邻接矩阵。

**预期收益**: 同组混淆降低 30-50%, mAP +1-2%。

### K2: 结构化输出空间 (Set Prediction)

**问题**: 独立预测无法保证"正好 46 条, 各类 2 条 (除性染色体)"。

**方案**: 用匈牙利匹配的 set prediction (类 DETR), 但约束集合结构。

**实现**:
```python
class KaryotypeSetPredictor(nn.Module):
    def __init__(self, num_queries=46):
        super().__init__()
        # 固定 46 个查询 (对应 46 条染色体)
        self.queries = nn.Embedding(num_queries, 256)

    def forward(self, image_features):
        # 查询与图像特征交叉注意力
        # 输出固定 46 个 (box, class) 对
        return boxes, classes  # (46, 4), (46,)

    def loss(self, pred, gt):
        # 匈牙利匹配 + 结构约束
        loss_match = hungarian_match(pred, gt)
        # 数量约束: 预测必须 46 条
        assert pred[0].shape[0] == 46
        # 类别约束: 各类应 2 条 (1-22), X/Y 各 1 条
        loss_count = self._count_constraint(pred_classes)
        return loss_match + λ * loss_count
```

**复杂度**: 中 — DETR 风格, 但查询数固定。

**预期收益**: 数量错误降为 0, mAP +0.5-1%。

### K3: 组条件化分类 (Group-Conditioned Classification)

**问题**: 同组形态相似 (C10↔C9), 独立分类难区分。

**方案**: 分类时注入组上下文, 用组内对比学习。

**实现**:
```python
class GroupConditionedClassifier(nn.Module):
    def __init__(self, feat_channels=256, num_classes=24):
        super().__init__()
        # 组感知注意力: 同组内做对比
        self.group_attention = nn.MultiheadAttention(feat_channels, num_heads=8)

    def forward(self, node_features, group_ids):
        # 按 group_ids 分组
        for group_id in unique(group_ids):
            group_feats = node_features[group_ids == group_id]
            # 组内注意力 (对比)
            enhanced = self.group_attention(group_feats, group_feats, group_feats)
            node_features[group_ids == group_id] = enhanced

        # 分类头
        logits = self.cls_head(node_features)
        return logits
```

**复杂度**: 低 — 只改分类头。

**预期收益**: 同组分类误差 -20-30%。

---

## 3. 核型先验的数学表达

### 3.1 核型结构约束

```
P(boxes, classes | image) = P(structure | image) × P(boxes | structure, image)
```

其中 structure 约束:
- |boxes| = 46
- |{c : c = k}| = 2 for k ∈ {1,...,22}
- |{c : c = X}| = 1 or 2 (性别)
- |{c : c = Y}| = 0 or 1

### 3.2 图结构定义

```python
def build_karyotype_graph(pred_classes):
    """根据预测类别构建核型图"""
    N = len(pred_classes)
    adjacency = torch.zeros(N, N)

    group_map = {1:0, 2:0, 3:0, 4:1, 5:1, ...}  # 类→组

    for i in range(N):
        for j in range(N):
            if group_map[pred_classes[i]] == group_map[pred_classes[j]]:
                adjacency[i, j] = 1.0  # 同组
            else:
                adjacency[i, j] = 0.1  # 异组 (弱连接)
    return adjacency
```

---

## 4. 文件组织

```
ldmdet/
├── models/
│   ├── karyotype_gnn.py        # K1: GNN 关系建模
│   ├── karyotype_predictor.py  # K2: 结构化输出
│   └── group_classifier.py     # K3: 组条件化分类
├── criterion/
│   └── karyotype_loss.py       # 结构约束损失
├── data/
│   └── karyotype_utils.py      # 核型先验工具
└── tests/
    └── test_direction_k.py
```

---

## 5. 测试计划

```python
class TestKaryotypeGNN:
    def test_output_count_fixed(self):
        """输出节点数应固定 46"""
    def test_group_information_flow(self):
        """同组节点应相互影响 (梯度验证)"""

class TestKaryotypeConstraints:
    def test_count_constraint(self):
        """预测必须 46 条"""
    def test_pair_constraint(self):
        """1-22 类各 2 条"""
    def test_sex_chromosome(self):
        """X/Y 各 1 条 (或 X 2 条)"""

class TestGroupClassifier:
    def test_same_group_contrast(self):
        """同组内分类应更准 (对比 C9/C10)"""
```

---

## 6. 实验计划

### 6.1 主实验

| 实验 | 配置 | 预期 |
|------|------|------|
| K1-gnn | GNN 关系建模 | 同组混淆 -30%, mAP +1-2% |
| K2-set | 结构化输出 | 数量错误 0, mAP +0.5-1% |
| K3-group | 组条件化分类 | 同组分类 -20% |
| K-all | 全部组合 | mAP +2-3%, 论文级 |

### 6.2 消融

| 消融 | 验证 |
|------|------|
| 有/无 GNN | K1 贡献 |
| 有/无数量约束 | K2 贡献 |
| 有/无组注意力 | K3 贡献 |
| 同组 vs 全连接图 | 图结构影响 |

---

## 7. 风险与缓解

### 7.1 GNN 训练不稳定

**风险**: 图结构依赖预测类别, 训练时类别未定。

**缓解**:
- 训练用 GT 类别建图, 推理用预测类别 (scheduled sampling)
- 软分配: 用类别概率加权重构图

### 7.2 性别未知

**风险**: X/Y 数量依赖性别 (XX 或 XY), 标注可能未知。

**缓解**:
- 用预测的 X/Y 数量推断性别
- 或不约束性染色体数量

### 7.3 异常核型

**风险**: 临床可能有异常 (三体、缺失), 不满足 46 条约束。

**缓解**: 设异常类 (class=25), 允许数量浮动。

---

## 8. 成功指标

| 指标 | baseline | 目标 |
|------|---------|------|
| mAP | 0.858 | +0.02 |
| 同组混淆率 | 43.4% | -20-30% |
| 数量错误率 | ~5% | 0% (K2) |
| 论文创新性 | - | 高 (领域特化) |

---

## 9. 实现路线图

1. **Phase 1 (4 周)**: K3 — 组条件化分类 (最小可行, 验证组关系收益)
2. **Phase 2 (4 周)**: K1 — GNN 关系建模
3. **Phase 3 (3 周)**: K2 — 结构化输出约束

---

## 10. 参考文献

- Graph R-CNN: https://arxiv.org/abs/1808.07172
- DETR: https://arxiv.org/abs/2005.12872
- ISCN 染色体核型标准
- Relation Networks: https://arxiv.org/abs/1711.08043
