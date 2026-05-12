<thinking>
The user is asking for a detailed explanation of how to design the conditional vector field and loss function for box coordinates in a Flow Matching-based detection model, along with concrete code implementation.

This is a very specific technical question. Let me think through the mathematics and implementation carefully.

Key aspects to cover:

1. The conditional vector field for box coordinates
2. The interpolation path design
3. Loss function design (including box regression and classification)
4. Handling of the set prediction problem
5. Concrete PyTorch code implementation

Let me think about the mathematical formulation first, then provide detailed code.

For Flow Matching applied to detection:

- We need to transport noise boxes to GT boxes
- The interpolation follows a linear path where boxes move from noise to ground truth as time progresses
- The network learns to predict the velocity field that drives this transformation
- The training loss measures how well the predicted velocity matches the true displacement between noise and target boxes

There are several important considerations: normalizing box coordinates to a consistent scale, deciding whether to use Hungarian matching for set prediction or direct regression, designing the classification head alongside the regression head, accounting for the signal-to-noise ratio challenges when working with low-dimensional box signals, and carefully handling the time conditioning throughout the model.

I also need to think through sampling strategies during inference and how to manage cases where multiple proposals correspond to the same ground truth box.

Writing comprehensive implementation...

</thinking>

# Flow Matching 检测模型：条件向量场与损失函数的完整设计与实现

---

## 一、数学基础：检测任务中的条件向量场

### 1.1 核心公式推导

**目标**：将随机噪声框 $b_0 \sim \mathcal{N}(0, I)$ 传输到 GT 框 $b_1 = b_{\text{gt}}$

**条件概率路径（线性插值）**：

$b_t = (1-t)\, b_0 + t\, b_1, \quad t \in [0, 1]$

**条件向量场（对 $t$ 求导）**：

$u_t(b_t \mid b_1) = \frac{d b_t}{d t} = b_1 - b_0$

**训练目标**：

$\mathcal{L}{\text{FM}} = \mathbb{E}{t \sim \mathcal{U}(0,1),\; b_0 \sim \mathcal{N}(0,I),\; b_1 \sim p_{\text{gt}}} \Big[\big\| v_\theta(b_t,\; \text{feat},\; t) - (b_1 - b_0) \big\|^2 \Big]$

### 1.2 检测任务的特殊性处理

```
关键问题: 检测不是单个框的生成, 而是 "集合到集合" 的映射

噪声框集合: B₀ = {b₀¹, b₀², ..., b₀ᴺ}    N个随机框 (如N=500)
GT框集合:   B_gt = {g¹, g², ..., gᴹ}       M个GT框 (M << N)

需要解决:
  1. N >> M, 如何分配? → 匹配策略
  2. 没有匹配到GT的噪声框怎么办? → 背景类处理
  3. 框坐标范围不同 (x,y vs w,h)? → 归一化策略
```

---

## 二、完整代码实现

### 2.1 框坐标归一化与编码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

class BoxEncoder:
    """
    框坐标的归一化与编码

    核心考虑:
    - Flow Matching 假设噪声为 N(0,I), 所以GT框需要归一化到类似尺度
    - (x,y,w,h) 各维度的自然范围不同, 需要统一
    - 提供多种编码方案供消融实验
    """

    def __init__(self, method='normalized_xyxy', image_size=None):
        """
        Args:
            method: 编码方式
                - 'normalized_xyxy': 坐标除以图像尺寸, 范围[0,1]
                - 'normalized_cxcywh': 中心点+宽高, 除以图像尺寸
                - 'snr_scaled': 根据信噪比自适应缩放 (推荐)
            image_size: (H, W) 图像尺寸
        """
        self.method = method
        self.image_size = image_size

    def encode(self, boxes, image_size=None):
        """
        将原始框坐标编码为适合 Flow Matching 的归一化空间

        Args:
            boxes: (N, 4) 格式为 (x1, y1, x2, y2), 像素坐标
            image_size: (H, W)
        Returns:
            encoded: (N, 4) 归一化后的框
        """
        H, W = image_size or self.image_size

        if self.method == 'normalized_xyxy':
            # 最简单: 直接归一化到 [0, 1]
            scale = torch.tensor([W, H, W, H], device=boxes.device, dtype=boxes.dtype)
            encoded = boxes / scale  # 范围 [0, 1]
            return encoded

        elif self.method == 'normalized_cxcywh':
            # 转为中心点+宽高格式
            cx = (boxes[:, 0] + boxes[:, 2]) / 2 / W
            cy = (boxes[:, 1] + boxes[:, 3]) / 2 / H
            w = (boxes[:, 2] - boxes[:, 0]) / W
            h = (boxes[:, 3] - boxes[:, 1]) / H
            encoded = torch.stack([cx, cy, w, h], dim=-1)  # [0, 1]
            return encoded

        elif self.method == 'snr_scaled':
            # ★ 推荐方案: 归一化后再做仿射变换
            #   使得GT框的分布近似 N(0, 1), 与噪声分布匹配
            scale = torch.tensor([W, H, W, H], device=boxes.device, dtype=boxes.dtype)
            normalized = boxes / scale  # [0, 1]

            # 仿射变换: [0,1] → 近似 N(0,1)
            # 中心化: 减去0.5, 范围变为 [-0.5, 0.5]
            # 缩放: 乘以2, 范围变为 [-1, 1]
            # 这样GT框和噪声框的尺度更匹配
            encoded = (normalized - 0.5) * 2.0  # [-1, 1]
            return encoded

    def decode(self, encoded, image_size=None):
        """逆变换: 将网络输出恢复为像素坐标"""
        H, W = image_size or self.image_size

        if self.method == 'snr_scaled':
            normalized = encoded / 2.0 + 0.5  # [-1,1] → [0,1]
            scale = torch.tensor([W, H, W, H], device=encoded.device, dtype=encoded.dtype)
            boxes = normalized * scale
            return boxes.clamp(min=0)

        elif self.method == 'normalized_xyxy':
            scale = torch.tensor([W, H, W, H], device=encoded.device, dtype=encoded.dtype)
            return (encoded * scale).clamp(min=0)

        elif self.method == 'normalized_cxcywh':
            cx, cy, w, h = encoded.unbind(-1)
            x1 = (cx - w / 2) * W
            y1 = (cy - h / 2) * H
            x2 = (cx + w / 2) * W
            y2 = (cy + h / 2) * H
            return torch.stack([x1, y1, x2, y2], dim=-1).clamp(min=0)
```

### 2.2 Flow Matching 调度器（核心组件）

```python
class DetectionFlowScheduler:
    """
    检测任务专用的 Flow Matching 调度器

    职责:
    1. 构造训练时的插值样本 b_t
    2. 计算条件向量场目标 (b_gt - b_noise)
    3. 推理时的 ODE 积分
    """

    def __init__(
        self,
        sigma_min: float = 1e-4,
        path_type: str = 'linear',        # 'linear' | 'cosine' | 'vp_compat'
        time_sampling: str = 'logit_normal',  # 'uniform' | 'logit_normal' | 'low_snr_emphasis'
    ):
        """
        Args:
            sigma_min: 最小噪声标准差, 避免数值问题
            path_type: 插值路径类型
                - 'linear': b_t = (1-t)*b0 + t*b1  (标准OT路径)
                - 'cosine': 余弦调度, 两端变化慢中间快
                - 'vp_compat': 兼容VP-SDE的路径, 便于与DiffusionDet对比
            time_sampling: 训练时时间步采样策略
                - 'uniform': t ~ U(0, 1)
                - 'logit_normal': t ~ σ(N(0, 1)), SD3使用的策略
                - 'low_snr_emphasis': 更多采样低信噪比区域
        """
        self.sigma_min = sigma_min
        self.path_type = path_type
        self.time_sampling = time_sampling

    # ──────── 训练阶段 ────────

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        采样训练时间步

        不同策略的直觉:
        - uniform: 各时间步均匀训练
        - logit_normal: 更多关注中间时间步 (信号和噪声混合最复杂的区域)
        - low_snr_emphasis: 更多关注 t≈0 (纯噪声端, 最难去噪)
        """
        if self.time_sampling == 'uniform':
            t = torch.rand(batch_size, device=device)

        elif self.time_sampling == 'logit_normal':
            # SD3 的策略: logit-normal 分布, 集中在 t≈0.5 附近
            # 对于检测任务, 这可能是好的选择, 因为中间状态最难预测
            u = torch.randn(batch_size, device=device)
            t = torch.sigmoid(u)  # 集中在 [0.2, 0.8]

        elif self.time_sampling == 'low_snr_emphasis':
            # 更多采样小 t (接近纯噪声), 检测任务中从噪声恢复框最关键
            u = torch.rand(batch_size, device=device)
            t = u ** 2  # 偏向 t≈0

        # 避免 t=0 和 t=1 的边界问题
        t = t.clamp(min=self.sigma_min, max=1.0 - self.sigma_min)
        return t

    def interpolate(
        self,
        b_noise: torch.Tensor,  # (B, N, 4) 噪声框
        b_gt: torch.Tensor,     # (B, N, 4) GT框 (已通过匹配分配)
        t: torch.Tensor,        # (B,) 或 (B, 1, 1) 时间步
    ) -> tuple:
        """
        构造插值样本和目标向量场

        Returns:
            b_t: (B, N, 4) 时间 t 处的插值框
            target_v: (B, N, 4) 条件向量场目标
        """
        # 确保 t 的维度正确: (B,) → (B, 1, 1) 用于广播
        if t.dim() == 1:
            t = t[:, None, None]  # (B, 1, 1)

        if self.path_type == 'linear':
            # ★ 标准 OT 路径: 线性插值
            # b_t = (1-t) * b_noise + t * b_gt
            b_t = (1 - t) * b_noise + t * b_gt

            # 条件向量场: db_t/dt = b_gt - b_noise (常数!)
            target_v = b_gt - b_noise

        elif self.path_type == 'cosine':
            # 余弦路径: 两端变化慢, 中间变化快
            # α_t = cos(π/2 * (1-t)), β_t = sin(π/2 * (1-t))  [示例]
            # 这里用更简单的余弦调度
            cos_t = torch.cos(t * torch.pi / 2)
            sin_t = torch.sin(t * torch.pi / 2)

            b_t = cos_t * b_noise + sin_t * b_gt

            # 对应的向量场: db_t/dt
            target_v = (-torch.sin(t * torch.pi / 2) * (torch.pi / 2) * b_noise
                        + torch.cos(t * torch.pi / 2) * (torch.pi / 2) * b_gt)

        elif self.path_type == 'vp_compat':
            # 兼容 VP-SDE 的路径, 便于与 DiffusionDet 公平对比
            # α_t = 1 - t (简化的线性信噪比)
            alpha_t = 1 - t
            sigma_t = t

            b_t = alpha_t * b_gt + sigma_t * b_noise

            # 向量场: d(b_t)/dt = -b_gt + b_noise
            target_v = b_noise - b_gt  # 注意方向!

        # 可选: 添加微小随机噪声, 提升训练稳定性
        # b_t = b_t + self.sigma_min * torch.randn_like(b_t)

        return b_t, target_v

    # ──────── 推理阶段 ────────

    @torch.no_grad()
    def ode_step(
        self,
        v_pred: torch.Tensor,    # (B, N, 4) 模型预测的速
```