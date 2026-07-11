# 方向 O2：对称性等变流 (Symmetry-Equivariant Flows, SEF)

> **核心思想**: 显式利用人类染色体 22 对常染色体的 **$Z_2$ 置换对称性**。通过在网络结构和损失函数中注入等变性约束，强制同源染色体在扩散演化过程中保持结构一致性。

---

## 1. 理论动机

染色体检测与通用目标检测的最大区别在于其**严格的配对结构**。
- 当前问题：LDMDet 视 46 条染色体为独立个体，导致在噪声干扰下，属于同一对的染色体（如两个 A1）可能朝着完全不一致的方向演化，甚至发生轨迹交叉。
- **SEF** 认为：速度场 $v_\theta$ 应该对同源染色体的置换具有**等变性 (Equivariance)**。如果输入的一对噪声互换，输出的速度也应该相应互换。

---

## 2. 数学推导

### 2.1 等变性定义
设 $g \in Z_2$ 是作用在同源对索引上的置换群元。对于输入状态 $x$（包含 46 个框），等变性要求：
$$v_\theta(g \cdot x, t) = g \cdot v_\theta(x, t)$$

### 2.2 势函数约束 (Potential-based Flow)
为了满足上述等变性，我们可以定义一个对称势函数 $\Phi(x, t)$，使得速度场为势函数的梯度：
$$v_\theta(x, t) = \nabla_x \Phi(x, t)$$
只要 $\Phi(x, t)$ 是关于同源对置换不变的（Invariant），其梯度 $v_\theta$ 自然满足等变性。

### 2.3 对称性损失 (Symmetry Loss)
引入正则项强制同源对 $(i, i')$ 的演化同步：
$$\mathcal{L}_{sym} = \sum_{(i, i') \in \text{Pairs}} \| \text{Feature}(i) - \text{Feature}(i') \|^2$$

---

## 3. 关键代码实现

### 3.1 对称感知注意力 (Symmetry-Aware Attention)

```python
class HomologousAttention(nn.Module):
    """强制同源对之间进行特征交换和同步"""
    def forward(self, x, pair_indices):
        """
        x: [bs, 46, dim]
        pair_indices: [23, 2] 存储同源对的索引映射
        """
        # 提取同源对特征
        left_idx = pair_indices[:, 0]
        right_idx = pair_indices[:, 1]
        
        x_left = x[:, left_idx]  # [bs, 23, dim]
        x_right = x[:, right_idx] # [bs, 23, dim]
        
        # 交互与同步 (Siamese 结构)
        sync_feats = (x_left + x_right) / 2.0
        
        # 写回并保持残差
        x[:, left_idx] = x[:, left_idx] + sync_feats
        x[:, right_idx] = x[:, right_idx] + sync_feats
        return x
```

### 3.2 等变速度场 Head

```python
class EquivariantVelocityHead(nn.Module):
    def forward(self, x, t):
        # 1. 基础 Transformer 处理
        feat = self.transformer(x, t)
        
        # 2. 注入同源对约束 (SEF 核心)
        feat = self.homo_sync(feat, self.pair_map)
        
        # 3. 预测速度
        v = self.reg_head(feat)
        return v
```

---

## 4. 叠加与优化 (Stacking Potential)

- **与 HCNT (方向 O1) 叠加**: SEF 确保了同源对的特征是接近的，这使得 HCNT 在生成成本矩阵时，能更容易地将这一对噪声匹配到对应的同源 GT 对上，避免了“跨对交叉匹配”。
- **与方向 L (形态先验) 叠加**: 同源染色体不仅位置应该同步，其形态编码也应该高度一致。SEF 可以作为形态先验注入的强约束。

---

## 5. 风险评估

- **非对称异常处理**: 临床上可能存在非整倍体（如 21-三体），此时对称性假设失效。
  - *缓解方案*: 引入“对称性权重” $\omega$，当检测到候选框数量不匹配时，自动降低等变约束的强度。
- **计算开销**: 增加了配对索引维护和额外的交互层。
