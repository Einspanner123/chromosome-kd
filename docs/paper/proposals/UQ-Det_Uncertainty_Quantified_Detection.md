# UQ-Det: Uncertainty-Quantified Detection via Stochastic Diffusion Sampling

> **方向类型**: 辅助应用方向（可独立投稿，也可与 CC-RF 叠加）
> **可叠加方向**: CC-RF（CC-RF 降低方差，UQ-Det 度量剩余方差，两者互补）
> **目标会议**: MICCAI 2026 / IEEE TMI
> **预期价值**: 临床安全性提升，不直接提升 mAP 但提升核型分析可靠率

---

## 1. 问题动机

### 1.1 当前局限性

当前扩散检测器的推理是确定性的：给定初始噪声，ODE 积分产生唯一结果。但扩散模型本质上是生成模型，其多次采样可以给出预测分布。

在医学影像中，**不确定性量化有直接临床价值**：
- 高置信度检测 → 自动接受，节省专家时间
- 低置信度检测 → 标记为"需专家复核"，保证安全

当前框架完全忽略了这一能力。

### 1.2 核心洞察

**利用扩散采样的随机性，提供有统计保证的检测不确定性。**

对同一输入，用 $S$ 个不同初始噪声独立采样，得到 $S$ 个检测结果。通过分析这 $S$ 个结果的分布，可以：
1. 计算检测均值（提升精度）
2. 计算检测方差（量化不确定性）
3. 构建置信区间（统计保证）

---

## 2. 理论推导

### 2.1 设定

- 输入图像 $I$，初始噪声 $\{x_1^{(s)}\}_{s=1}^S \stackrel{iid}{\sim} \mathcal{N}(0, \sigma^2 I_4)$
- 每个噪声独立积分得到检测结果 $\hat{x}_0^{(s)} = \text{ODE}(x_1^{(s)}; I)$
- 由于 ODE 是确定性的，$\hat{x}_0^{(s)}$ 的随机性完全来自 $x_1^{(s)}$

### 2.2 定理 3（采样一致性）

**声明**:

$$\bar{x}_0 = \frac{1}{S}\sum_{s=1}^S \hat{x}_0^{(s)} \xrightarrow{a.s.} \mathbb{E}[X_0 \mid I] \quad (S \to \infty)$$

**证明**: $\hat{x}_0^{(s)}$ 是 $iid$ 样本，由强大数定律（Strong Law of Large Numbers），样本均值几乎必然收敛到期望。$\square$

### 2.3 定理 4（置信区间覆盖率）

**声明**: 对于 $S$ 次采样，构造逐 proposal 置信区间：

$$CI_\alpha^{(i)} = \left[\bar{x}_0^{(i)} - z_{\alpha/2}\frac{\hat{\sigma}^{(i)}}{\sqrt{S}}, \ \bar{x}_0^{(i)} + z_{\alpha/2}\frac{\hat{\sigma}^{(i)}}{\sqrt{S}}\right]$$

其中 $\hat{\sigma}^{(i)2} = \frac{1}{S-1}\sum_{s=1}^S (\hat{x}_0^{(s,i)} - \bar{x}_0^{(i)})^2$。

当 $S \to \infty$，$CI_\alpha^{(i)}$ 的渐近覆盖率为 $1-\alpha$。

**证明**: 由中心极限定理（CLT），$\sqrt{S}(\bar{x}_0^{(i)} - \mathbb{E}[X_0^{(i)}|I]) / \hat{\sigma}^{(i)} \xrightarrow{d} \mathcal{N}(0,1)$。因此 $P(\mathbb{E}[X_0^{(i)}|I] \in CI_\alpha^{(i)}) \to 1-\alpha$。$\square$

### 2.4 命题 5（集成精度提升）

**声明**: 集成均值 $\bar{x}_0$ 的期望 MSE 比单次采样降低 $1/S$：

$$\mathbb{E}\left[\|\bar{x}_0 - \mathbb{E}[X_0|I]\|^2\right] = \frac{1}{S}\text{Var}(X_0|I)$$

**证明**: $\text{Var}(\bar{x}_0) = \text{Var}(X_0)/S$（$iid$ 样本均值方差）。$\square$

### 2.5 检测级不确定性

对每个 proposal $i$，定义：

$$u^{(i)} = \frac{\hat{\sigma}^{(i)}}{\sqrt{S}}$$

$u^{(i)}$ 是标准误差，反映 proposal $i$ 的预测不确定性。

**检测级不确定性**（整图）:

$$U = \frac{1}{M}\sum_{i=1}^M u^{(i)} \cdot \mathbb{1}[\text{score}^{(i)} > \tau]$$

其中 $M$ 是高分 proposal 数量，$\tau$ 是分数阈值。

### 2.6 临床分流策略

| 不确定性 $U$ | 处理策略 | 临床效率 |
|---|---|---|
| $U < \tau_1$ | 自动接受 | 节省专家时间 |
| $\tau_1 \leq U < \tau_2$ | 辅助参考 | 专家快速复核 |
| $U \geq \tau_2$ | 强制人工 | 保证安全 |

**期望专家时间**: $\mathbb{E}[T_{expert}] = (1 - p_{auto}) \cdot T_{full}$，其中 $p_{auto}$ 是自动接受比例。

### 2.7 与 CC-RF 的互补关系

- **CC-RF**: 通过条件化**降低** $\text{Var}(X_0|X_t)$（定理 1）
- **UQ-Det**: 通过多次采样**度量**剩余 $\text{Var}(X_0|X_t)$（定理 3-4）

两者叠加：CC-RF 降低基准方差，UQ-Det 精确度量剩余方差，临床分流更准确。

---

## 3. 架构设计

### 3.1 多采样推理

```python
@torch.no_grad()
def predict_with_uncertainty(self, features, img_metas, num_samples=10, rescale=True):
    all_results = []
    for s in range(num_samples):
        # 每次用不同随机种子
        torch.manual_seed(s)
        results = self.predict(features, img_metas, rescale=rescale)
        all_results.append(results)

    # 聚合结果
    return self._aggregate_uncertainty(all_results, img_metas)
```

### 3.2 不确定性聚合

**Proposal 级聚合**:
1. 对 $S$ 次采样的结果做 box matching（IoU-based 匹配）
2. 匹配到的 proposals 计算 mean 和 std
3. 未匹配到的 proposals 标记为高不确定性

**图像级聚合**:
$$U_{image} = \text{median}(\{u^{(i)} : \text{score}^{(i)} > \tau\})$$

### 3.3 监控指标（SwanLab 插桩）

推理时记录：
- `uq_mean_uncertainty`: 平均不确定性
- `uq_auto_accept_rate`: 自动接受比例（$U < \tau_1$）
- `uq_karyotype_accuracy`: 核型分析准确率（24 条全对率）

### 3.4 集成精度提升

当前 `predict` 方法已有 `use_ensemble=True` 选项，但仅对 ODE 中间步结果做 ensemble。UQ-Det 的 ensemble 是**跨采样**的（不同初始噪声），更符合统计意义。

---

## 4. 实现计划

### 4.1 代码修改清单

| 文件 | 修改内容 |
|---|---|
| `ldmdet/core/head.py` | 新增 `predict_with_uncertainty` 方法 |
| `ldmdet/core/uncertainty.py` | 新建不确定性聚合模块 |
| `experiments/runners/test_uq.py` | 新建 UQ 推理脚本 |
| `tests/unit/test_uncertainty.py` | 新建单元测试 |

### 4.2 关键实现细节

**`predict_with_uncertainty`**:
```python
def predict_with_uncertainty(self, features, img_metas, num_samples=10,
                              uncertainty_thresholds=(0.01, 0.05)):
    all_detections = []
    for s in range(num_samples):
        torch.manual_seed(42 + s)
        x_raw = torch.randn(bs, self.num_proposals, 4, device=device)
        # ... 标准 ODE 积分
        all_detections.append(results)

    # Box matching across samples
    matched_groups = self._match_across_samples(all_detections)

    # Compute per-detection uncertainty
    uncertainties = []
    for group in matched_groups:
        boxes = torch.stack([g.bboxes for g in group])
        scores = torch.stack([g.scores for g in group])
        box_mean = boxes.mean(dim=0)
        box_std = boxes.std(dim=0)
        score_mean = scores.mean(dim=0)
        u = box_std.norm(dim=-1).mean()
        uncertainties.append({
            'bboxes': box_mean, 'scores': score_mean,
            'uncertainty': u
        })

    return uncertainties
```

### 4.3 测试计划

**单元测试** (`tests/unit/test_uncertainty.py`):
1. `test_predict_with_uncertainty_returns_results`: 基本功能
2. `test_different_seeds_different_results`: 不同种子产生不同结果
3. `test_uncertainty_decreases_with_samples`: 样本数增加，不确定性估计更稳定
4. `test_box_matching_iou`: 跨采样 box matching 正确性
5. `test_confidence_interval_coverage`: 蒙特卡洛验证置信区间覆盖率
6. `test_ensemble_improves_precision`: 集成均值比单次采样精度高
7. `test_clinical_triage`: 临床分流逻辑正确性
8. `test_uq_with_cc_rf`: UQ + CC-RF 兼容性

### 4.4 配置设计

UQ-Det 不修改训练配置，仅在推理时启用：

```bash
# UQ-Det 推理
bash test.sh experiments/configs/ldmdet/directions/cc_rf/cc_rf_24obj.py \
    --checkpoint work_dirs/cc_rf/best.pth \
    --num-samples 10 \
    --uncertainty-thresholds 0.01 0.05
```

---

## 5. 实验设计

### 5.1 主实验

| 实验 | 配置 | 指标 |
|---|---|---|
| Baseline (单次) | A3, 1 次采样 | mAP, 核型准确率 |
| **UQ-Det (10 次)** | A3, 10 次采样 + 集成 | mAP (+0.005~0.010), 核型准确率 |
| **UQ-Det 分流** | A3, 10 次 + 不确定性分流 | 自动接受率, 核型准确率（自动+人工） |

### 5.2 不确定性验证

对 100 张验证图，每张 50 次采样：
- 验证置信区间覆盖率（目标 ~95% for $\alpha=0.05$）
- 验证不确定性与检测错误率的相关性
- 绘制不确定性-精度校准曲线

### 5.3 临床分流评估

| 阈值设置 | 自动接受率 | 自动接受核型准确率 | 整体核型准确率 |
|---|---|---|---|
| $\tau_1=0.01, \tau_2=0.05$ | ?% | ?% | ?% |
| $\tau_1=0.02, \tau_2=0.10$ | ?% | ?% | ?% |

---

## 6. 预期贡献

1. **理论**: 定理 3-5 给出扩散检测不确定性量化的统计保证
2. **方法**: 多采样集成 + IoU-based 跨采样匹配 + 临床分流策略
3. **应用**: 临床核型分析的自动分流，提升专家效率

---

## 7. 风险评估

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| 推理速度慢（S 倍） | 高 | 中 | 使用 AMP + 4 步采样，单次 ~55ms，10 次 ~550ms 仍可接受 |
| Box matching 不准确 | 中 | 高 | 使用 IoU + 匹配阈值 + 匈牙利算法 |
| 不确定性与错误率不相关 | 低 | 高 | 扩散模型的随机性天然反映预测难度 |
| 集成精度提升不显著 | 中 | 低 | 定理 5 保证 $1/S$ 降低，S=10 时显著 |
