# 方向 O3：概率核型熵冷却 (Probabilistic Karyotype Entropy Cooling, PKEC)

> **核心思想**: 在推理采样阶段注入全局逻辑约束。将“核型标准”定义为能量函数，通过梯度引导（Guidance）使预测结果向“合法核型”坍缩。

---

## 1. 理论动机

染色体检测的最终输出必须服从严格的离散统计约束（如 1-22 类各 2 条）。
- 当前问题：扩散模型在采样终点附近的微小扰动可能导致类别判定错误（如把一个 A1 误判为 A2），从而破坏整体核型结构。
- **PKEC** 认为：采样不应只由速度场驱动，还应由**全局约束场**驱动。当预测集合偏离“22对+XY”结构时，产生一个“冷却力”将其拉回。

---

## 2. 数学推导

### 2.1 核型能量函数 (Karyotype Energy)
定义集合 $\mathcal{Y} = \{(b_i, c_i)\}_{i=1}^{46}$ 的能量函数 $E(\mathcal{Y})$：
$$E(\mathcal{Y}) = \sum_{k=1}^{24} (Count(c=k) - TargetCount(k))^2$$
其中 $TargetCount(k)$ 是标准核型的数量（常染色体为 2，性染色体为 1/2）。

### 2.2 引导采样 (Guided Sampling)
在推理的每一步 $t$，更新状态 $x_t$：
$$x_{t-dt} = x_t + v_\theta(x_t, t) \cdot dt - \eta_t \nabla_{x_t} E(\hat{x}_0)$$
其中 $\hat{x}_0$ 是当前时刻对最终结果的估计，$\eta_t$ 是随时间衰减的冷却系数。

### 2.3 熵冷却 (Entropy Cooling)
为了处理类别预测的不确定性，使用分类概率分布的熵作为正则项，强迫模型在采样后期产生确定性输出（低熵态）。

---

## 3. 关键代码实现

### 3.1 全局约束引导模块

```python
class KaryotypeGuidance:
    def __init__(self, target_counts):
        self.target_counts = target_counts # [24] 维向量

    def compute_guidance(self, cls_probs, bboxes):
        """
        cls_probs: [N, 24] 预测概率
        bboxes: [N, 4] 预测框
        """
        # 1. 计算当前集合的期望数量分布
        curr_counts = cls_probs.sum(dim=0) # [24]
        
        # 2. 计算能量函数 (MSE 距离)
        energy = torch.sum((curr_counts - self.target_counts) ** 2)
        
        # 3. 计算对坐标和概率的梯度
        # 注意：这里需要通过分类头反传梯度，引导特征改变
        guidance_grad = torch.autograd.grad(energy, bboxes)[0]
        
        return guidance_grad
```

### 3.2 采样循环集成

```python
for t in reversed(timesteps):
    # 模型前向
    v_pred, cls_logits = model(x_t, t)
    
    # 计算全局引导 (PKEC 核心)
    if t < guidance_threshold:
        g = pkec.compute_guidance(F.softmax(cls_logits), x_t)
        v_final = v_pred + λ_t * g
    else:
        v_final = v_pred
        
    # 执行采样步
    x_t = rectified_flow.step(x_t, v_final)
```

---

## 4. 叠加与优化 (Stacking Potential)

- **叠加性**: PKEC 是一个**纯推理期 (Inference-only)** 或 **微调期 (Fine-tuning)** 的技术。它可以无缝叠加在 HCNT (O1) 和 SEF (O2) 训练出的模型之上。
- **协同效应**: 
    - SEF 负责让同源对“长得像”。
    - HCNT 负责让同源对“跑得直”。
    - PKEC 负责最后临门一脚，确保“数量对”。

---

## 5. 风险评估

- **梯度断路**: 如果分类预测非常果断（One-hot），梯度将消失。需要使用 Softmax 温度调节。
- **局部最优**: 强行的全局引导可能导致框位置发生畸变以满足数量要求。需要严格控制引导系数 $\lambda_t$。
