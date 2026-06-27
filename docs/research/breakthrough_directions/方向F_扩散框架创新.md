# 方向 F：扩散框架创新与领域适配 (Diffusion Framework Innovation)

> **目标**：基于 DiffusionDet 对框坐标做高斯扩散的核心思路，探索创新改进与染色体领域适配。当前模型将检测视为"从高斯噪声到 GT 框坐标"的扩散去噪过程，本方向探讨这一思路本身的可改进点。
>
> **理论依据**：
> - DiffusionDet: Du et al., CVPR 2023 — 将扩散应用于检测框坐标
> - Rectified Flow: Liu et al., ICLR 2023 — 直线流匹配
> - Chromosome domain: 染色体核型分析的特殊性 (46 条固定数量、组内形态相似、可能有重叠/杂质)
>
> **当前代码位置**：[ldmdet/diffusion/rectified_flow.py](../../ldmdet/diffusion/rectified_flow.py), [ldmdet/core/head.py](../../ldmdet/core/head.py)

---

## 1. 当前扩散思路回顾

### 1.1 DiffusionDet 的核心机制

[head.py](../../ldmdet/core/head.py) 中:

1. **随机噪声初始化**: `bboxes = randn(N, 4)` (N=500 proposals)
2. **扩散耦合**: 通过 GHSS/random coupling 将噪声框 $x_1$ 与 GT 框 $x_0$ 配对
3. **流匹配训练**: 学习速度场 $v_\theta(x_t, t) \approx x_1 - x_0$，路径 $x_t = (1-t)x_0 + t x_1$
4. **采样去噪**: 从 $x_1$ (噪声) 出发，用 $v_\theta$ 反向积分到 $x_0$ (GT)
5. **Box renewal**: 每步采样后用网络重新预测框
6. **Ensemble**: 多次采样取平均

### 1.2 核心局限

**局限 1: 噪声先验与染色体分布不匹配**

标准高斯噪声 $\mathcal{N}(0, I)$ 与染色体框的真实分布 (固定 46 条、组内数量约束、尺寸有界) 差异巨大。模型需"从纯噪声找到 GT"，但染色体框的先验很强 (数量固定、组内数量约束)，高斯噪声浪费了这些先验。

**局限 2: 框坐标扩散忽略形态信息**

扩散仅作用于 4D 框坐标 $(x, y, w, h)$，但染色体的形态 (臂长比、着丝粒位置、弯曲度) 是分类的关键。框坐标扩散与形态分类脱节。

**局限 3: 46 条固定数量未利用**

染色体核型固定 46 条 (22 对常染色体 + 2 条性染色体)，但当前 N=500 proposals 远超 46，大量 proposals 浪费在背景上。

---

## 2. 创新方向探索

### F1: 结构化噪声先验 (Structured Noise Prior)

**思路**: 用染色体框的统计分布代替高斯噪声作为扩散起点。

**方案**:
- 从训练集统计各类框的位置/尺寸分布 $\mathcal{P}(x | c)$
- 训练时 $x_1 \sim \mathcal{P}(x)$ (结构化噪声)，而非 $\mathcal{N}(0, I)$
- 采样时从 $\mathcal{P}(x)$ 采样初始框，减少"从纯噪声找 GT"的难度

**实现**:
```python
# 预计算各类别框的高斯混合分布
class StructuredPrior:
    def __init__(self, dataset_stats):
        self.gmm = fit_gmm(dataset_stats)  # 按类别拟合 GMM
    def sample(self, n):
        return self.gmm.sample(n)  # 替代 randn(n, 4)
```

**收益**: 模型只需学习"从分布内噪声到 GT"的精化，降低学习难度，可能提升少步采样精度。

**风险**: 采样可能困在统计分布的局部，缺乏探索性。

### F2: 形态感知扩散 (Morphology-Aware Diffusion)

**思路**: 将扩散从 4D 框坐标扩展到高维形态空间，让扩散过程同时建模位置和形态。

**方案**:
- 扩散目标从 $(x, y, w, h)$ 扩展到 $(x, y, w, h, \text{shape\_emb})$
- $\text{shape\_emb}$ 由辅助编码器从 RoI 特征提取 (训练时)
- 采样时联合去噪框坐标和形态嵌入

**实现**: 需修改 [rectified_flow.py](../../ldmdet/diffusion/rectified_flow.py) 的 `x_t` 维度，并增加形态编码器。

**收益**: 扩散过程直接建模形态，分类和定位联合优化。

**风险**: 高维扩散训练困难，可能不稳定。

### F3: 数量约束扩散 (Count-Constrained Diffusion)

**思路**: 利用染色体固定 46 条的先验，约束扩散过程。

**方案**:
- proposals 数从 500 减为 ~100 (覆盖 46 + 冗余)
- 采样时加入数量约束: 每 step 后用计数分支预测目标数，截断多余 proposals
- 已有 [ldmdet/core/counting_branch.py](../../ldmdet/core/counting_branch.py) 和 [ldmdet/inference/count_constrained_nms.py](../../ldmdet/inference/count_constrained_nms.py) 基础

**收益**: 减少背景 proposals 的浪费，提升有效 proposals 利用率。

**风险**: 计数预测不准可能导致漏检。

### F4: 分组扩散 (Group-wise Diffusion)

**思路**: 染色体按组 (A-G, X, Y) 形态差异大，分组扩散可能更有效。

**方案**:
- 每个组独立的扩散头 (A 组用大框扩散, G 组用小框扩散)
- 或条件化扩散: 扩散过程以组标签为条件 $v_\theta(x_t, t, c)$

**收益**: 每组扩散路径适配该组尺度/形态分布。

**风险**: 参数量增加，小样本组 (Y 类 202 个) 训练不足。

### F5: 离散类别扩散 (Categorical Diffusion for Classification)

**思路**: 当前分类是确定性的 (cls_head 输出 logits)，但染色体分类本质有不确定性 (形态相似类)。将分类也建模为扩散过程。

**方案**:
- 类别用离散扩散 (D3PM) 建模，从均匀分布扩散到具体类别
- 与框坐标扩散联合训练

**收益**: 分类不确定性显式建模，可能改善形态相似类的判别。

**风险**: 离散扩散训练复杂，与连续框扩散融合困难。

### F6: 条件化扩散 (Conditional Diffusion with Image Features)

**思路**: 当前扩散以图像特征为条件，但条件化方式可能不足。增强条件信息注入。

**方案**:
- 用更强的图像编码器 (Swin Transformer 替代 ResNet)
- 或用 cross-attention 将图像特征注入扩散过程每一步
- AdaLN-Zero 已实现时间条件化，可扩展为时间+图像双条件

**收益**: 扩散过程更充分利用图像信息，提升定位精度。

**风险**: 计算量增加。

---

## 3. 推荐实验设计

### F1 + F3 组合 (结构化噪声 + 数量约束)

**子方向互相抵消评估**:
- F1 结构化噪声 → 改变扩散起点
- F3 数量约束 → 改变 proposals 数量和后处理
- 二者**正交** (一个改输入分布，一个改输出约束)，可组合

**组合配置**:

| 配置项 | baseline | F1+F3 组合 |
|--------|----------|-----------|
| 噪声先验 | $\mathcal{N}(0, I)$ | 数据集统计分布 GMM (F1) |
| num_proposals | 500 | 100 (F3) |
| 后处理 | NMS | Count-Constrained NMS (F3) |
| 计数分支 | 无 | 启用 (F3) |

**预期**:
- mAP +0.01 ~ +0.03 (结构化噪声降低学习难度)
- 推理速度提升 (proposals 从 500 减到 100)
- 若 mAP 提升 > 0.015，方向 F 确认可行

### 备选: F6 单独实验 (条件化增强)

若 F1+F3 效果不明显，可尝试 F6 (更强 backbone + cross-attention)。

---

## 4. 实现步骤 (F1+F3 组合)

1. 新建 [ldmdet/diffusion/structured_prior.py](../../ldmdet/diffusion/) — 结构化噪声分布
2. 统计训练集框分布，拟合 GMM
3. 修改 [head.py](../../ldmdet/core/head.py) 噪声初始化为 F1 采样
4. 启用 [counting_branch.py](../../ldmdet/core/counting_branch.py) 和 [count_constrained_nms.py](../../ldmdet/inference/count_constrained_nms.py)
5. num_proposals 改为 100
6. 新建配置 `structured_diffusion.py`

---

## 5. 风险与可行性

| 风险 | 评估 |
|------|------|
| F1 结构化噪声可能导致模式崩溃 | 加随机扰动保证多样性 |
| F3 计数不准导致漏检 | 先训练计数分支到合理精度再启用约束 |
| F2/F5 形态扩散/离散扩散复杂度高 | 暂不尝试，优先 F1+F3 |
| F6 更换 backbone 计算量大 | 作为备选，F1+F3 失败后再考虑 |

**结论**：方向 F 是最具创新性的方向 (领域适配 + 扩散框架本身改进)。F1+F3 组合可一次性验证，若有效则可进一步探索 F2/F4/F5。
