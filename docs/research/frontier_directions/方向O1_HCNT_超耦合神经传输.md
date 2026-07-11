# 方向 O1：超耦合神经传输 (Hyper-Coupled Neural Transport, HCNT)

> **核心思想**: 将 OT 耦合过程从“外部预处理”转化为“网络内置的可学习组件”。通过神经网络动态生成成本矩阵，实现特征驱动的最优传输路径。

---

## 1. 理论动机

在当前的 LDMDet (Rectified Flow) 架构中，训练时的配对 (Coupling) 是通过 Sinkhorn 或随机分配完成的。这种配对存在两个根本局限：
1. **信息孤岛**: 耦合过程只考虑坐标（几何距离），不考虑语义特征。模型无法学习到“为什么这个噪声应该对应这条染色体”。
2. **静态路径**: 传输路径（速度场目标）在训练前就已经确定（或者说在每个 batch 开始前确定），Transformer 只能被动拟合，无法主动优化路径以降低学习难度。

**HCNT** 提出：如果耦合过程是可微的，模型就可以学习去“寻找”最容易学习的传输路径。

---

## 2. 数学推导

### 2.1 传统 OT 耦合
给定噪声集合 $\mathbf{z} = \{z_i\}_{i=1}^N$ 和真值集合 $\mathbf{y} = \{y_j\}_{j=1}^M$，成本矩阵 $C_{ij} = \|z_i - y_j\|^2$。传输计划 $P$ 通过最小化下式获得：
$$\min_{P \in \mathcal{U}(a,b)} \langle P, C \rangle - \epsilon H(P)$$

### 2.2 神经耦合 (Neural Coupling)
我们引入耦合生成器 $f_\phi$，它根据特征生成动态成本：
$$C_{ij}^\phi = f_\phi(\mathbf{F}_{z,i}, \mathbf{F}_{y,j})$$
其中 $\mathbf{F}_z$ 是噪声的查询特征，$\mathbf{F}_y$ 是真值的语义特征。

### 2.3 目标函数
总损失函数包含两部分：
1. **传输损失 (Flow Loss)**: $\mathcal{L}_{flow} = \|v_\theta(x_t, t) - (z_i - y_{P(i)})\|^2$
2. **耦合效率损失 (Efficiency Loss)**: $\mathcal{L}_{eff} = \langle P, C^\phi \rangle$

通过对 $\phi$ 求导，模型会倾向于生成能使 $v_\theta$ 更容易拟合（即路径更直、冲突更少）的配对方案。

---

## 3. 关键代码实现

### 3.1 耦合生成器模块

```python
class CouplingGenerator(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )

    def forward(self, z_feats, y_feats):
        # z_feats: [N, C], y_feats: [M, C]
        N, M = z_feats.shape[0], y_feats.shape[0]
        # 广播拼接
        z_expanded = z_feats.unsqueeze(1).expand(N, M, -1)
        y_expanded = y_feats.unsqueeze(0).expand(N, M, -1)
        pair_feats = torch.cat([z_expanded, y_expanded], dim=-1)
        
        # 生成动态成本矩阵
        cost_matrix = self.proj(pair_feats).squeeze(-1)
        return cost_matrix # [N, M]
```

### 3.2 可微 Sinkhorn 集成

```python
def hcnt_coupling(z_feats, y_feats, z_boxes, y_boxes):
    # 1. 计算神经成本 (语义)
    semantic_cost = coupling_gen(z_feats, y_feats)
    # 2. 计算几何成本 (坐标)
    geometric_cost = torch.cdist(z_boxes, y_boxes)
    # 3. 融合成本
    total_cost = semantic_cost + λ * geometric_cost
    
    # 4. Sinkhorn 迭代 (保持梯度)
    P = log_sinkhorn_iterations(total_cost, eps=0.1, iters=20)
    
    # 5. 根据 P 进行配对
    # ...
```

---

## 4. 叠加与优化 (Stacking Potential)

- **与 SEF (方向 O2) 叠加**: SEF 提供的对称性先验可以作为 HCNT 成本矩阵的正则项。如果两个噪声被识别为“同源对候选”，则它们匹配到同一对 GT 的成本应该被关联。
- **与 PKEC (方向 O3) 叠加**: HCNT 负责训练时的路径优化，PKEC 负责推理时的结构修正，两者分别在“模型潜力”和“输出质量”上发力。

---

## 5. 风险评估

- **梯度循环依赖**: $v_\theta$ 的训练依赖于 $P$，而 $P$ 的生成依赖于特征。需要精细的 Warmup 策略，前期使用静态 OT，中后期引入神经耦合。
- **计算开销**: $N \times M$ 的特征拼接计算量较大，建议在轻量级特征空间进行。
