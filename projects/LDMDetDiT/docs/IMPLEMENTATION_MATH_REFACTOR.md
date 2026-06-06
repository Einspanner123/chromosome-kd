# LDMDet 架构数学重构与落实方案 (Phase 2)

> 核心目标：彻底移除打破 Lipschitz 连续性的 RoIAlign 与导致流形塌缩的 DynamicConv，基于高斯积分近似与联合流形投影理论，构建纯 Token 化的扩散检测器。

## 1. 理论基础回顾
- **连续性法则 (ODE Lipschitz Continuity)**: RoIAlign 存在 $\delta$ 函数级别的梯度断层，导致 ODE 截断误差爆炸。必须使用 $C^0$ 连续的双线性插值进行近似。
- **联合流形投影 (Joint Manifold Projection)**: 目标检测是对结构化集合的去噪，DynamicConv 的单向信息漏斗无法建模联合概率，必须通过 Self-Attention 计算经验协方差矩阵。

## 2. 代码级落实方案

### 2.1 C-RoI (Continuous-RoI) 的 Deformable 近似
**当前问题**: `BoxTokenizer` 中使用中心点进行 `F.grid_sample`，仅采了一个点，无法感知框的大小（方差）。
**实施动作**:
1. 废弃简单的中心点采样，启用多点 `Deformable Cross-Attention`。
2. **核心数学修正 (Variance-aware Sampling)**:
   - 网络预测的 `sampling_offsets` (形状 `[N, K, 2]`) 必须乘以当前边界框的宽高 $(w, h)$。
   - 代码体现：
     ```python
     # box_wh shape: [N, 2]
     # sampling_offsets shape: [N, K, 2]
     scaled_offsets = sampling_offsets * box_wh.unsqueeze(1)
     sample_points = box_centers.unsqueeze(1) + scaled_offsets
     ```
   - *理论意义*：这是标准的基于边界框协方差矩阵 $\Sigma$ 的蒙特卡洛积分近似。

### 2.2 流形投影仪 (Self-Attention) 的修正与 RoPE 注入
**当前问题**: 之前直接调用 `nn.MultiheadAttention`，丢失了空间相对位置关系。
**实施动作**:
1. 采用 `mods/dit_block.py` 中已修复的拆解版 Self-Attention。
2. 确保在计算 $QK^T$ 之前，`apply_rope` 作用于二维框坐标映射出的旋转位置编码。
3. 移除 `curr_tokens.detach()`，确保从深层到浅层的雅可比矩阵满秩，梯度不被截断。

### 2.3 KCEC (核型配额约束) 的“死者苏生”：Attention Bias
**当前问题**: KCEC 放在 Loss 层面，梯度被 DynamicConv 阻断（信息截断）。
**实施动作**:
1. 在 `DiTBlock` 的深层（如最后 3 层），引入额外的 `Attention Bias` 矩阵 $M_{KCEC} \in \mathbb{R}^{N \times N}$。
2. 根据先验规则（如染色体总数不超过 46，同类不超过 2 条），计算一个抑制/促进矩阵。
3. 代码体现：
   ```python
   # M_KCEC 赋予互斥框极小的负值 (如 -1e9)，赋予同类框正值
   attn_weights = F.softmax((q @ k.transpose(-2, -1)) / scale + M_KCEC, dim=-1)
   ```
   - *理论意义*：将全局的拉格朗日乘子（约束）直接注入联合流形的协方差计算中。

## 3. 实验验证路径
1. **Sanity Check**: 关闭 KCEC，关闭分类，仅回归一个类别的 BBox，验证 4 步 Rectified Flow 的 `loss_bbox` 是否能平滑下降（验证连续性法则）。
2. **Variance Ablation**: 对比开启/关闭 `scaled_offsets`（乘宽高）时的收敛速度，证明高斯积分近似的正确性。
3. **Manifold Ablation**: 引入 KCEC Attention Bias，观测染色体配额错误的发生率是否归零。

## 4. 未来架构探索：纯数学理论驱动的新范式

抛开已有的结构（DiT、Deformable DETR 等），完全从数学理论推导、矩阵变换、奇异值分解 (SVD)、同构空间等出发，我们可以探讨以下三种完全颠覆传统目标检测的数学范式，以针对性解决染色体检测中严格的拓扑和配额约束问题。

### 4.1 范式一：基于奇异值分解 (SVD) 的谱检测器 (Spectral-Det)
**数学洞察**：
目前的检测器都在“像素空间”或者“坐标空间”里硬搜目标。但一幅包含 46 条染色体的图像特征矩阵 $M \in \mathbb{R}^{(H \times W) \times C}$，在代数上其实是由 46 个主要的正交基（Orthogonal Basis）张成的子空间。

**推导与创新结构**：
1. **谱分解定义**：设特征图 $M \in \mathbb{R}^{L \times C}$（其中 $L = H \times W$ 为空间展平维度）。我们对其进行截断奇异值分解 (Truncated SVD)，保留前 $K$ 个奇异值（例如 $K=46$）：
   $$ M \approx U_K \Sigma_K V_K^T $$
   其中 $U_K \in \mathbb{R}^{L \times K}$ 为左奇异矩阵，$\Sigma_K \in \mathbb{R}^{K \times K}$ 为对角奇异值矩阵，$V_K \in \mathbb{R}^{C \times K}$ 为右奇异矩阵。
2. **物理意义映射 (Orthogonal Spatial Masks)**：
   - 根据 SVD 的性质，$U_K$ 的列向量是正交的，即 $U_K^T U_K = I_K$。
   - 这意味着对于任意两列 $U_i, U_j$ (对应两个不同的染色体目标)，其内积为零：$\sum_{l=1}^L U_{i,l} U_{j,l} = 0$。
   - **推论**：如果我们将 $U_i$ 视作第 $i$ 个目标的“空间概率掩码（Spatial Mask）”，这种正交性**天然保证了 46 个目标的掩码是互斥的，在物理空间上绝对不会发生重叠**！这就从纯代数层面上彻底消灭了 NMS（非极大值抑制）的需求。
3. **基于谱空间的扩散方程**：
   - 传统的扩散检测在欧氏坐标 $x_t \in \mathbb{R}^{N \times 4}$ 上加噪，而在 Spectral-Det 中，扩散过程定义在右奇异矩阵 $V_K$（即 $K$ 个正交特征向量）上。
   - 设 $V_t$ 为时间 $t$ 时的加噪特征，网络预测速度场 $v_\theta(V_t, t)$。常微分方程为：
     $$ dV_t = v_\theta(V_t, t) dt $$
   - 解出 $V_0$ 后，通过 $\hat{M} = U_K \Sigma_K V_0^T$ 还原出具备清晰边界的特征矩阵，并通过阈值化 $U_K$ 提取检测框。

### 4.2 范式二：同构空间与拓扑保形流 (Isomorphic Topology-Preserving Flow)
**数学洞察**：
KCEC 试图施加全局约束，但在连续无界的欧氏空间 $\mathbb{R}^4$ 里预测独立坐标，极易发生“拓扑撕裂”（如重叠、缺失、数量错误）。

**推导与创新结构**：
1. **预定义拓扑流形**：我们在隐空间预先定义一个完美的“标准核型流形” $\mathcal{S}$，包含 $N=46$ 个标准插槽。输入图像特征定义为流形 $\mathcal{I}$。
2. **离散同构映射 (Isomorphism via Permutation)**：
   - 目标检测问题转化为：寻找一个从 $\mathcal{I}$ 的局部特征集到 $\mathcal{S}$ 的一一对应关系。
   - 数学上，这被表示为一个置换矩阵（Permutation Matrix） $P \in \{0, 1\}^{N \times N}$。
   - 约束条件：$P \mathbf{1} = \mathbf{1}$ 且 $P^T \mathbf{1} = \mathbf{1}$（每行每列的和严格为 1）。
3. **流形松弛与 Sinkhorn 迭代**：
   - 置换矩阵所在的群是高度离散非连续的，无法直接进行微分方程求解。
   - **推导**：根据 Birkhoff-von Neumann 定理，所有双随机矩阵（Doubly Stochastic Matrices）的凸包正好是置换矩阵的集合。我们将 $P$ 松弛为双随机矩阵 $D \in \mathbb{R}_+^{N \times N}$。
   - 给定网络的预测代价矩阵 $C$，通过 Sinkhorn-Knopp 算法投影到双随机流形：
     $$ D = \text{diag}(u) \exp(-C / \epsilon) \text{diag}(v) $$
4. **拓扑保形的 Rectified Flow**：
   - 扩散方程定义在双随机矩阵流形上：$\frac{dD_t}{dt} = v_\theta(D_t, t)$。
   - **推论**：由于 $D_t$ 在每个时间步都严格保证行和与列和为 1，这意味着模型**在任何时刻的预测都严格符合“总数为 46，一一对应”的拓扑约束**。配额溢出在数学定义上被直接抹杀。

### 4.3 范式三：基于李代数变换的仿射流 (Lie Algebra Affine Flow)
**数学洞察**：
传统边界框 $x = (cx, cy, w, h)$ 将平移（$\mathbb{R}^2$）和尺度（$\mathbb{R}^+$）生硬拼凑在向量空间中，但尺度不具备加法封闭性（$w_1 + w_2$ 可能导致严重形变，$w - \Delta w$ 可能导致负数宽度）。

**推导与创新结构**：
1. **群论视角的边界框 (Box as a Lie Group Element)**：
   - 任意一个边界框，都可以视为对“标准单位中心框”施加的一次仿射变换 $g \in G$。
   - 这个群是平移与尺度/旋转的半直积，例如 $G = SE(2) \times \mathbb{R}^+$。
   - 矩阵表示：
     $$ g = \begin{bmatrix} s \cdot R(\theta) & \mathbf{t} \\ 0 & 1 \end{bmatrix} $$
     其中 $s$ 为尺度，$\mathbf{t} = [cx, cy]^T$ 为平移，$R(\theta)$ 为旋转矩阵。
2. **向李代数的对数映射 (Logarithmic Map)**：
   - 流形 $G$ 是弯曲的，无法直接做线性的 $x_t = (1-t)x_0 + t x_1$（Rectified Flow 加噪）。
   - 我们通过对数映射 $\log: G \to \mathfrak{g}$，将群元素投影到其单位元处的切空间（李代数 $\mathfrak{g}$，即一个平坦的欧氏空间）。
     $$ u = \log(g) \in \mathbb{R}^d $$
3. **切空间扩散与指数映射还原 (Manifold Diffusion)**：
   - 在平坦的李代数空间 $\mathfrak{g}$ 中执行常规的 Rectified Flow 加噪和预测：$u_t = (1-t)u_0 + t u_1$。
   - 预测完成后，通过指数映射 $\exp: \mathfrak{g} \to G$ 还原回物理边界框：
     $$ g_{pred} = \exp(u_{pred}) $$
4. **数学保证**：
   - **推论**：因为 $\exp(x)$ 永远为正，网络无论在李代数空间预测多离谱的值，映射回来的宽 $w$ 和高 $h$ **绝对不可能为负数**。
   - 此外，如果在 $g$ 中加入角度 $\theta$（处理弯曲染色体），李代数上的直线距离本质上就是流形上的**测地线（Geodesic）距离**，完美化解了 $\theta=0$ 和 $\theta=2\pi$ 的周期性断层灾难。
