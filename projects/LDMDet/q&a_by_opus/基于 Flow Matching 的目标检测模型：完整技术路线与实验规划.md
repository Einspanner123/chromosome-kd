# 基于 Flow Matching 的目标检测模型：完整技术路线与实验规划

---

## 一、项目总览与研究定位

### 1.1 研究题目（建议）

> **"FlowDet: Rectified Flow Matching for Efficient Object Detection"**
> 
> 
> 或更聚焦：**"From Diffusion to Flow: Straightening the Path for Generative Object Detection"**
> 

### 1.2 核心研究问题

```
Q1: Flow Matching 能否替代 DDPM 成为更高效的生成式检测框架？
Q2: Rectified Flow 的直线路径特性能否实现 1-2 步高质量检测？
Q3: 生成式检测框架在不确定性估计上是否具有独特优势？
```

### 1.3 研究路线图

```
Phase 1 (月1-2)          Phase 2 (月3-4)          Phase 3 (月5-6)
━━━━━━━━━━━━━━━         ━━━━━━━━━━━━━━━         ━━━━━━━━━━━━━━━
基础框架搭建              加速方法研究              应用验证+论文撰写
├─ 复现 DiffusionDet     ├─ Rectified Flow       ├─ 不确定性估计
├─ 实现 Flow Matching    ├─ Consistency 蒸馏       ├─ 开放世界检测
│  基础版本              ├─ 自适应步数策略         ├─ 消融实验
└─ 基准实验对齐          └─ 架构优化              └─ 论文撰写
```

---

## 二、Phase 1：基础框架搭建

### 2.1 整体架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                       FlowDet 架构                           │
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌────────────────────────┐ │
│  │  Image   │───▶│ Backbone │───▶│  FPN 多尺度特征         │ │
│  │  Input   │    │(ResNet/  │    │  {P3, P4, P5, P6, P7}  │ │
│  │          │    │ Swin)    │    │                        │ │
│  └──────────┘    └──────────┘    └───────────┬────────────┘ │
│                                              │              │
│  ┌──────────┐    ┌──────────────────────┐    │              │
│  │ Noisy    │───▶│  Flow Matching Head  │◀───┘              │
│  │ Boxes    │    │                      │                   │
│  │b_t~p_t   │    │  ┌─────────────────┐ │                   │
│  │          │    │  │ RoI Align       │ │                   │
│  │ + time t │    │  │ ↓               │ │                   │
│  │          │    │  │ Self-Attention  │ │                   │
│  └──────────┘    │  │ ↓               │ │                   │
│                  │  │ Cross-Attention │ │                   │
│                  │  │ (box ↔ feature) │ │                   │
│                  │  │ ↓               │ │                   │
│                  │  │ FFN → v_θ, cls  │ │                   │
│                  │  └─────────────────┘ │                   │
│                  └──────────────────────┘                   │
│                           │                                 │
│                    ┌──────┴──────┐                          │
│                    │  输出:       │                          │
│                    │  v_θ (速度场) │                          │
│                    │  cls (类别)  │                          │
│                    └─────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 核心代码框架

### 2.2.1 Flow Matching 训练逻辑

```python
import torch
import torch.nn as nn

class FlowMatchingDetection(nn.Module):
    """
    核心训练逻辑: 将框坐标的 DDPM 去噪替换为 Flow Matching
    """
    def __init__(self, backbone, decoder, num_proposals=500):
        super().__init__()
        self.backbone = backbone          # ResNet-50 / Swin-T
        self.decoder = decoder            # Transformer Decoder
        self.num_proposals = num_proposals
        self.sigma_min = 1e-4             # 最小噪声 (避免数值问题)

    def forward(self, images, gt_boxes, gt_labels):
        """
        训练前向传播
        """
        # ============ Step 1: 提取图像特征 (只做一次) ============
        features = self.backbone(images)   # 多尺度特征

        # ============ Step 2: 构造 Flow Matching 训练对 ============
        # 对每张图的 GT 框进行处理
        batch_size = images.shape[0]

        # 采样时间 t ~ Uniform(0, 1)
        t = torch.rand(batch_size, 1, 1, device=images.device)  # (B, 1, 1)

        # 准备 "源" 和 "目标"
        # b_0 ~ N(0, I): 噪声框 (归一化坐标空间)
        b_noise = torch.randn_like(gt_boxes)  # (B, N, 4)

        # b_1 = gt_boxes: 真实框 (归一化到 [0,1])
        b_gt = gt_boxes                        # (B, N, 4)

        # ============ Step 3: 线性插值得到 b_t ============
        # Optimal Transport 条件路径
        b_t = (1 - t) * b_noise + t * b_gt     # (B, N, 4)

        # 可选: 加微小高斯噪声 (条件路径的方差)
        # b_t = b_t + self.sigma_min * torch.randn_like(b_t)

        # ============ Step 4: 网络预测速度场 ============
        # 条件向量场的目标: v_target = b_gt - b_noise
        v_target = b_gt - b_noise              # (B, N, 4)

        # 网络预测
        v_pred, cls_pred = self.decoder(
            b_t,                    # 当前噪声框位置
            t.squeeze(-1),          # 时间步
            features                # 图像特征 (条件)
        )

        # ============ Step 5: 计算损失 ============
        # 速度场回归损失
        loss_flow = F.mse_loss(v_pred, v_target)
        # 或使用 Huber Loss (对异常值更鲁棒)
        # loss_flow = F.smooth_l1_loss(v_pred, v_target)

        # 分类损失
        loss_cls = focal_loss(cls_pred, gt_labels)

        # GIoU 辅助损失 (可选, 提供几何监督)
        b_pred_direct = b_t + (1 - t) * v_pred  # 预测终点
        loss_giou = giou_loss(b_pred_direct, b_gt)

        loss = loss_flow + loss_cls + 0.5 * loss_giou

        return loss

    @torch.no_grad()
    def inference(self, images, num_steps=1):
        """
        推理: ODE 积分
        """
        features = self.backbone(images)

        # 从纯噪声开始
        b_t = torch.randn(
            images.shape[0], self.num_proposals, 4,
            device=images.device
        )

        # Euler 积分
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full(
                (images.shape[0], 1),
                i * dt, device=images.device
            )
            v_pred, cls_pred = self.decoder(b_t, t, features)
            b_t = b_t + v_pred * dt

        # 最后一步的分类结果
        return b_t, cls_pred
```

### 2.2.2 与 DiffusionDet DDPM 逻辑的代码级对比

```python
# ===============================================
# DiffusionDet (DDPM) 的训练
# ===============================================
def ddpm_train_step(model, images, gt_boxes, gt_labels):
    features = model.backbone(images)

    # 采样离散时间步
    t = torch.randint(0, T, (batch_size,))     # 离散整数

    # DDPM 前向加噪
    alpha_bar = get_alpha_bar(t)
    epsilon = torch.randn_like(gt_boxes)
    b_t = sqrt(alpha_bar) * gt_boxes + sqrt(1 - alpha_bar) * epsilon
    #     ^^^^^^^^^^^^^^^^             ^^^^^^^^^^^^^^^^^^^^^^^^
    #     非线性系数                      非线性系数

    # 预测噪声
    eps_pred = model.decoder(b_t, t, features)
    loss = F.mse_loss(eps_pred, epsilon)        # 预测 ε
    return loss

# ===============================================
# FlowDet (Flow Matching) 的训练
# ===============================================
def flow_train_step(model, images, gt_boxes, gt_labels):
    features = model.backbone(images)

    # 采样连续时间
    t = torch.rand(batch_size, 1, 1)            # 连续 [0,1]

    # Flow Matching 线性插值
    b_noise = torch.randn_like(gt_boxes)
    b_t = (1 - t) * b_noise + t * gt_boxes
    #     ^^^^^^^             ^
    #     线性系数             线性系数

    # 预测速度
    v_pred = model.decoder(b_t, t, features)
    v_target = gt_boxes - b_noise
    loss = F.mse_loss(v_pred, v_target)          # 预测 v
    return loss
```

```
关键代码差异总结:
┌─────────────────┬──────────────────┬──────────────────┐
│                 │    DiffusionDet   │    FlowDet       │
├─────────────────┼──────────────────┼──────────────────┤
│ 时间采样         │ 离散 randint     │ 连续 rand        │
│ 插值方式         │ √ᾱ·x + √(1-ᾱ)·ε │ (1-t)·ε + t·x  │
│ 预测目标         │ 噪声 ε           │ 速度 v=x-ε       │
│ Noise schedule  │ 需要精心设计β     │ 不需要 (线性)     │
│ 推理采样器       │ DDPM/DDIM        │ Euler ODE        │
└─────────────────┴──────────────────┴──────────────────┘
```

### 2.3 GT 框分配与 Padding 策略

```python
class BoxPairConstructor:
    """
    处理不同图像有不同数量GT框的问题
    """
    def __init__(self, num_proposals=500, padding_strategy='repeat'):
        self.num_proposals = num_proposals
        self.padding_strategy = padding_strategy

    def construct_pairs(self, gt_boxes_list):
        """
        gt_boxes_list: List[Tensor], 每个 (M_i, 4), M_i 不等

        策略选择:
        1. repeat:  重复 GT 框填满 N 个位置
        2. noise:   多余位置用特殊 "背景" 框
        3. dynamic: 不同图像用不同数量的框
        """
        batch_targets = []
        batch_labels = []

        for gt_boxes in gt_boxes_list:
            M = gt_boxes.shape[0]

            if self.padding_strategy == 'repeat':
                # 重复GT框到N个 (DiffusionDet的做法)
                if M == 0:
                    # 无目标图像: 全部是背景框
                    padded = torch.zeros(self.num_proposals, 4)
                    labels = torch.zeros(self.num_proposals, dtype=torch.long)
                else:
                    repeat_times = self.num_proposals // M + 1
                    padded = gt_boxes.repeat(repeat_times, 1)[:self.num_proposals]
                    labels = gt_labels.repeat(repeat_times)[:self.num_proposals]

            elif self.padding_strategy == 'noise':
                # GT框 + 随机背景框
                padded = torch.randn(self.num_proposals, 4) * 0.1
                padded[:M] = gt_boxes
                labels = torch.zeros(self.num_proposals, dtype=torch.long)
                labels[:M] = gt_labels

            batch_targets.append(padded)
            batch_labels.append(labels)

        return torch.stack(batch_targets), torch.stack(batch_labels)
```

### 2.4 时间嵌入设计

```python
class TimeEmbedding(nn.Module):
    """
    连续时间嵌入 (与DDPM的离散时间步嵌入不同)
    """
    def __init__(self, dim=256, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t):
        """
        t: (B,) 连续值 [0, 1]
        """
        # Sinusoidal embedding
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) *
            torch.arange(half, device=t.device) / half
        )
        args = t[:, None] * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        return self.mlp(embedding)  # (B, dim)
```

### 2.5 Decoder 设计

```python
class FlowDetDecoder(nn.Module):
    """
    Flow Matching Detection Decoder
    在 DiffusionDet decoder 基础上的修改:
    1. 时间嵌入方式: 连续 → sinusoidal + MLP
    2. 预测头: 预测 v(速度) 而非 ε(噪声)
    3. 可选: 增加 shortcut prediction
    """
    def __init__(self, d_model=256, nhead=8, num_layers=6,
                 num_classes=80):
        super().__init__()

        self.time_embed = TimeEmbedding(d_model)
        self.box_embed = nn.Linear(4, d_model)

        # Transformer Decoder Layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model*4, batch_first=True
        )
        self.decoder_layers = nn.ModuleList(
            [decoder_layer for _ in range(num_layers)]
        )

        # 预测头
        self.velocity_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 4)       # 输出: 速度 v ∈ R^4
        )
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, noisy_boxes, t, image_features):
        """
        noisy_boxes: (B, N, 4)     - 当前框位置
        t:           (B, 1)        - 当前时间
        image_features: 多尺度特征
        """
        B, N, _ = noisy_boxes.shape

        # 框位置编码
        box_tokens = self.box_embed(noisy_boxes)     # (B, N, D)

        # 时间编码 (广播到所有框)
        time_tokens = self.time_embed(t.squeeze(-1))  # (B, D)
        time_tokens = time_tokens.unsqueeze(1)         # (B, 1, D)

        # 初始 query = 框编码 + 时间编码
        query = box_tokens + time_tokens               # (B, N, D)

        # RoI Align 提取每个框的特征
        roi_features = self.roi_align(
            noisy_boxes, image_features
        )   # (B, N, D)

        # 图像全局特征 (memory for cross-attention)
        memory = self.flatten_features(image_features)  # (B, HW, D)

        # Transformer Decoder
        for layer in self.decoder_layers:
            query = layer(
                tgt=query + roi_features,  # self-attn among boxes
                memory=memory              # cross-attn with image
            )

        # 预测
        v_pred = self.velocity_head(query)     # (B, N, 4)
        cls_pred = self.cls_head(query)        # (B, N, num_classes)

        return v_pred, cls_pred
```

---

## 三、Phase 2：加速方法研究

### 3.1 方法 A：Rectified Flow (Reflow)

```python
class RectifiedFlowDet:
    """
    通过 Reflow 拉直传输路径, 实现单步检测
    """

    def reflow_data_generation(self, model_v1, dataloader):
        """
        Phase 1 模型训练完后, 生成 reflow 训练数据
        """
        paired_data = []

        for images, gt_boxes, gt_labels in dataloader:
            features = model_v1.backbone(images)

            # 从噪声出发, 用多步 ODE 求解到终点
            b_0 = torch.randn(B, N, 4)  # 起点: 噪声
            b_1 = self.ode_solve(
                model_v1, b_0, features, num_steps=20
            )  # 终点: 模型认为的 "干净框"

            # 收集 (起点, 终点) 配对
            paired_data.append({
                'images': images,
                'b_start': b_0,       # 噪声框
                'b_end': b_1,         # ODE求解的终点框
                'features': features
            })

        return paired_data

    def reflow_train_step(self, model_v2, paired_batch):
        """
        Reflow 训练: 让 v2 学习 (b_0, b_1) 之间的直线映射
        """
        b_0 = paired_batch['b_start']
        b_1 = paired_batch['b_end']
        features = paired_batch['features']

        t = torch.rand(B, 1, 1)
        b_t = (1 - t) * b_0 + t * b_1

        v_target = b_1 - b_0          # 直线方向
        v_pred, _ = model_v2.decoder(b_t, t, features)

        loss = F.mse_loss(v_pred, v_target)
        return loss

    def one_step_inference(self, model_v2, images):
        """
        Reflow 后: 单步推理
        """
        features = model_v2.backbone(images)
        b_noise = torch.randn(B, N, 4)
        t = torch.zeros(B, 1)

        v_pred, cls_pred = model_v2.decoder(b_noise, t, features)
        b_det = b_noise + v_pred  # 一步到位

        return b_det, cls_pred
```

### 3.2 方法 B：Consistency 蒸馏

```python
class ConsistencyDetDistillation:
    """
    从训练好的 Flow Matching 教师模型蒸馏到
    单步 Consistency 学生模型
    """

    def __init__(self, teacher_model, student_model):
        self.teacher = teacher_model  # 已训练的 FlowDet
        self.student = student_model  # 待训练
        self.ema_student = deepcopy(student_model)  # EMA 目标网络
        self.ema_rate = 0.999

    def train_step(self, images, gt_boxes):
        features = self.teacher.backbone(images)  # 共享特征

        # 采样两个相邻时间步
        t_n = torch.rand(B, 1, 1) * 0.9 + 0.1  # 避免 t≈0
        delta_t = 0.05  # 小步长
        t_n_minus_1 = t_n - delta_t

        # 在 t_n 处采样
        b_noise = torch.randn_like(gt_boxes)
        b_tn = (1 - t_n) * b_noise + t_n * gt_boxes

        # 用教师模型做一步 ODE: t_n → t_{n-1}
        with torch.no_grad():
            v_teacher, _ = self.teacher.decoder(b_tn, t_n, features)
            b_tn_minus_1 = b_tn - v_teacher * delta_t  # 反向一步

        # Consistency 约束:
        # f_student(b_tn, tn) ≈ f_ema(b_{tn-1}, t_{n-1})
        pred_student = self.student.consistency_head(
            b_tn, t_n, features
        )
        with torch.no_grad():
            pred_ema = self.ema_student.consistency_head(
                b_tn_minus_1, t_n_minus_1, features
            )

        loss = F.mse_loss(pred_student, pred_ema)

        # 更新 EMA
        self._update_ema()

        return loss

    def one_step_inference(self, images):
        features = self.student.backbone(images)
        b_noise = torch.randn(B, N, 4)
        t = torch.ones(B, 1) * 0.999  # 从 t≈1 (纯噪声)

        b_det = self.student.consistency_head(
            b_noise, t, features
        )  # 直接映射到干净框

        return b_det
```

### 3.3 方法 C：自适应步数策略（创新点）

```python
class AdaptiveStepFlowDet:
    """
    核心创新: 根据检测难度自适应调整去噪步数

    简单场景 (清晰, 少遮挡): 1步
    困难场景 (密集, 遮挡): 3-4步
    """

    def __init__(self, model, step_predictor):
        self.model = model
        self.step_predictor = step_predictor  # 轻量网络预测步数

    def adaptive_inference(self, images):
        features = self.model.backbone(images)

        # Step 1: 预测场景难度 → 决定步数
        difficulty = self.step_predictor(features)  # (B, 1)
        num_steps = self.map_to_steps(difficulty)    # 1, 2, 3, or 4

        # Step 2: 按需迭代
        b_t = torch.randn(B, N, 4)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((B, 1), i * dt)
            v_pred, cls_pred = self.model.decoder(b_t, t, features)
            b_t = b_t + v_pred * dt

            # 可选: 早停判断
            if self.convergence_check(b_t, v_pred):
                break

        return b_t, cls_pred

    def convergence_check(self, b_t, v_pred):
        """
        如果速度场已经很小, 说明已收敛, 可以提前停止
        """
        velocity_magnitude = v_pred.norm(dim=-1).mean()
        return velocity_magnitude < self.threshold
```

---

## 四、Phase 3：实验设计

### 4.1 实验全景图

```
实验体系:
│
├── 4.2 基础对比实验 (Table 1, 2)
│   ├── FlowDet vs DiffusionDet (同架构, 换训练范式)
│   ├── FlowDet vs DETR 系列 (跨范式对比)
│   └── 不同骨干网络下的一致性验证
│
├── 4.3 加速方法对比 (Table 3, Fig 3)
│   ├── 不同步数下的 AP-FPS 曲线
│   ├── Euler vs Midpoint vs RK4 ODE 求解器
│   ├── Rectified Flow (Reflow) 加速效果
│   └── Consistency Distillation 加速效果
│
├── 4.4 消融实验 (Table 4)
│   ├── 路径设计: 线性 vs VP vs VE vs 余弦
│   ├── 预测目标: v-pred vs ε-pred vs x-pred
│   ├── 损失函数: MSE vs Huber vs GIoU辅助
│   ├── 时间采样策略: uniform vs logit-normal vs importance
│   └── 提案数量: 100 / 300 / 500 / 1000
│
├── 4.5 独特优势验证 (Table 5, Fig 4)
│   ├── 不确定性估计质量 (校准曲线)
│   ├── 遮挡/模糊场景的鲁棒性
│   └── 动态框数量的灵活性
│
└── 4.6 可视化分析 (Fig 5, 6)
    ├── 去噪轨迹可视化
    ├── Flow Matching vs DDPM 路径对比
    └── 速度场可视化
```

### 4.2 基础对比实验

### Table 1: COCO val2017 主实验

```
设计原则: 控制变量, 只改训练范式, 其余全部对齐

┌──────────────────┬──────────┬──────────┬───────┬───────┬────────┐
│ Method           │ Backbone │ Steps    │ AP    │ AP50  │ FPS    │
├──────────────────┼──────────┼──────────┼───────┼───────┼────────┤
│ DiffusionDet     │ ResNet-50│ 4 (DDPM) │ 46.1  │ 63.2  │ 5.6    │
│ FlowDet (ours)   │ ResNet-50│ 4 (Euler)│ ??    │ ??    │ 5.6    │
│ FlowDet (ours)   │ ResNet-50│ 2 (Euler)│ ??    │ ??    │ 10.3   │
│ FlowDet (ours)   │ ResNet-50│ 1 (Euler)│ ??    │ ??    │ 17.5   │
├──────────────────┼──────────┼──────────┼───────┼───────┼────────┤
│ FlowDet+Reflow   │ ResNet-50│ 1        │ ??    │ ??    │ 17.5   │
│ FlowDet+Consist. │ ResNet-50│ 1        │ ??    │ ??    │ 17.5   │
├──────────────────┼──────────┼──────────┼───────┼───────┼────────┤
│ DINO-DETR        │ ResNet-50│ -        │ 49.4  │ 66.9  │ 23     │
│ RT-DETR          │ ResNet-50│ -        │ 46.5  │ 63.8  │ 108    │
└──────────────────┴──────────┴──────────┴───────┴───────┴────────┘

期望结果:
- FlowDet 4步 ≥ DiffusionDet 4步 (路径更直, 同步数更优)
- FlowDet 2步 ≈ DiffusionDet 4步 (核心卖点: 步数减半)
- FlowDet 1步 + Reflow/Consistency 接近 DiffusionDet 4步
```

### Table 2: 不同骨干网络

```
┌──────────────────┬──────────┬───────┬────────┐
│ Method           │ Backbone │ AP    │ FPS    │
├──────────────────┼──────────┼───────┼────────┤
│ FlowDet (1-step) │ ResNet-50│ ??    │ ??     │
│ FlowDet (1-step) │ ResNet-101│??    │ ??     │
│ FlowDet (1-step) │ Swin-T   │ ??    │ ??     │
│ FlowDet (1-step) │ Swin-B   │ ??    │ ??     │
└──────────────────┴──────────┴───────┴────────┘
→ 验证方法的通用性
```

### 4.3 加速方法对比

### Figure 3: AP vs. 推理步数曲线

```python
# 实验代码骨架
def speed_accuracy_curve():
    """
    核心图表: 展示 Flow Matching 在少步时的优势
    """
    results = {}

    for method in ['DDPM', 'FlowMatch', 'FlowMatch+Reflow', 'Consistency']:
        for steps in [1, 2, 3, 4, 6, 8, 10]:
            model = load_model(method)
            ap = evaluate_coco(model, num_steps=steps)
            fps = measure_fps(model, num_steps=steps)
            results[(method, steps)] = (ap, fps)

    # 绘制
    plot_ap_vs_steps(results)    # X: steps, Y: AP
    plot_ap_vs_fps(results)      # X: FPS,   Y: AP (Pareto 曲线)
```

```
期望的图表效果:

AP
48 │           ────────── FlowMatch
   │        ╱    ─────── FlowMatch+Reflow
46 │      ╱╱  ╱────────── DDPM (DiffusionDet)
   │    ╱╱ ╱
44 │  ╱╱╱╱
   │╱╱╱
42 │╱╱
   │╱
40 │
   └──┬───┬───┬───┬───┬──
      1   2   3   4   6  Steps

关键信息:
- FlowMatch 在 1-2 步时显著优于 DDPM
- Reflow 进一步提升 1 步性能
- 4 步时差异缩小 (足够步数下都能收敛)
```

### 4.4 消融实验

### Table 4: 关键设计选择

```
(a) 概率路径设计
┌────────────────┬───────┬───────────────────────────────┐
│ Path Type      │ AP    │ 备注                           │
├────────────────┼───────┼───────────────────────────────┤
│ Linear (OT)    │ ??    │ b_t = (1-t)ε + t·x            │
│ VP-like        │ ??    │ b_t = √ᾱ_t·x + √(1-ᾱ_t)·ε   │
│ Cosine         │ ??    │ b_t = cos(πt/2)·ε + sin(πt/2)x│
│ Quadratic      │ ??    │ b_t = (1-t²)ε + t²·x          │
└────────────────┴───────┴───────────────────────────────┘

(b) 预测目标
┌────────────┬───────┬──────────────┐
│ Target     │ AP    │ 说明          │
├────────────┼───────┼──────────────┤
│ v (速度)    │ ??    │ v = x - ε    │
│ ε (噪声)   │ ??    │ DDPM 标准    │
│ x (数据)   │ ??    │ 直接预测框    │
└────────────┴───────┴──────────────┘

(c) 时间采样策略
┌─────────────────┬───────┬────────────────────────────┐
│ Sampling        │ AP    │ 说明                        │
├─────────────────┼───────┼────────────────────────────┤
│ Uniform [0,1]   │ ??    │ 标准                       │
│ Logit-Normal    │ ??    │ SD3 使用, 聚焦中间时间步      │
│ Beta(2,1)       │ ??    │ 偏向 t≈1 (接近数据端)        │
│ Importance      │ ??    │ 按损失大小加权               │
└─────────────────┴───────┴────────────────────────────┘

(d) 损失函数
┌──────────────────────┬───────┐
│ Loss                 │ AP    │
├──────────────────────┼───────┤
│ MSE only             │ ??    │
│ MSE + GIoU           │ ??    │
│ Huber + GIoU         │ ??    │
│ MSE + GIoU + Focal   │ ??    │
└──────────────────────┴───────┘
```

### 4.5 不确定性估计实验

```python
def uncertainty_experiment(model, test_loader, num_samples=50):
    """
    核心实验: 验证 Flow Matching 检测器的不确定性质量
    """
    all_predictions = []

    for images, gt_boxes in test_loader:
        # 多次采样 (不同噪声初始化)
        samples = []
        for _ in range(num_samples):
            b_noise = torch.randn(1, N, 4)  # 不同噪声
            boxes, scores = model.inference(images, b_noise)
            samples.append(boxes)

        samples = torch.stack(samples)  # (K, 1, N, 4)

        # 计算统计量
        mean_boxes = samples.mean(dim=0)           # 均值框
        var_boxes = samples.var(dim=0)             # 方差 → 不确定性
        detection_rate = (samples > threshold).float().mean(0)

        all_predictions.append({
            'mean': mean_boxes,
            'variance': var_boxes,
            'detection_rate': detection_rate,
            'gt': gt_boxes
        })

    # 评估不确定性质量
    calibration = compute_calibration_curve(all_predictions)
    ece = expected_calibration_error(all_predictions)
    nll = negative_log_likelihood(all_predictions)

    return calibration, ece, nll
```

```
Table 5: 不确定性估计质量

┌──────────────────┬───────┬────────┬────────┬─────────────┐
│ Method           │ AP    │ ECE↓   │ NLL↓   │ 用途         │
├──────────────────┼───────┼────────┼────────┼─────────────┤
│ DINO-DETR        │ 49.4  │  -     │  -     │ 无不确定性   │
│ DINO + MC-Drop   │ 49.1  │ 0.082  │ 3.21   │ 近似贝叶斯  │
│ DINO + Ensemble  │ 50.2  │ 0.065  │ 2.89   │ 5模型集成    │
│ DiffusionDet×50  │ 46.1  │ 0.058  │ 2.67   │ 50次采样    │
│ FlowDet×50(ours) │ ??    │ ??     │ ??     │ 50次采样    │
└──────────────────┴───────┴────────┴────────┴─────────────┘

期望: 生成式方法的不确定性校准优于判别式近似方法
```

### 4.6 可视化实验

```python
def visualize_flow_trajectory(model, image, gt_boxes):
    """
    可视化: 框从噪声到目标的传输轨迹
    """
    features = model.backbone(image)

    num_vis_steps = 20
    b_t = torch.randn(1, 500, 4)
    trajectory = [b_t.clone()]

    dt = 1.0 / num_vis_steps
    for i in range(num_vis_steps):
        t = torch.full((1, 1), i * dt)
        v_pred, _ = model.decoder(b_t, t, features)
        b_t = b_t + v_pred * dt
        trajectory.append(b_t.clone())

    # 绘制轨迹
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    for idx, step in enumerate([0, 5, 10, 15, 19]):
        axes[idx].imshow(image)
        draw_boxes(axes[idx], trajectory[step], color='red', alpha=0.3)
        draw_boxes(axes[idx], gt_boxes, color='green', linewidth=2)
        axes[idx].set_title(f't = {step/20:.2f}')
```

```
期望的可视化效果:

t=0.0 (纯噪声)     t=0.25          t=0.5           t=0.75         t=1.0 (收敛)
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ ·  ·  ·  │    │  · ·     │    │   ┌─┐    │    │  ┌──┐    │    │  ┌──┐    │
│· ·  · ·  │    │ ·  ┌─┐   │    │   │ │    │    │  │🐕│    │    │  │🐕│    │
│  ·  ·  · │    │   ·│ │·  │    │  ·└─┘ ·  │    │  └──┘    │    │  └──┘    │
│ ·  ·  ·  │    │  · └─┘   │    │  ┌──┐    │    │  ┌───┐   │    │  ┌───┐   │
│·  ·   ·  │    │    ·  ·  │    │  │  │    │    │  │🐈 │   │    │  │🐈 │   │
│  · · ·   │    │   ┌──┐   │    │  └──┘    │    │  └───┘   │    │  └───┘   │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
  随机散布           聚集             成形            精修            检测结果

对比: Flow Matching 的轨迹应该更"直", DDPM 的轨迹更"弯曲"
```

---

## 五、训练细节与超参数

### 5.1 训练配置

```yaml
# config.yaml
model:
  backbone: resnet50          # 或 swin_tiny
  decoder_layers: 6
  d_model: 256
  nhead: 8
  num_proposals: 500

flow_matching:
  sigma_min: 1e-4
  path_type: "linear"         # linear / vp / cosine
  prediction: "velocity"      # velocity / epsilon / data
  time_sampling: "logit_normal"  # uniform / logit_normal / beta

training:
  optimizer: AdamW
  lr: 2.5e-5
  weight_decay: 1e-4
  lr_schedule: step           # step decay at epoch 40, 60
  batch_size: 16              # 8 GPUs × 2 images/GPU
  total_epochs: 75
  warmup_epochs: 5

  losses:
    flow_weight: 5.0          # 速度回归权重
    cls_weight: 2.0           # 分类权重
    giou_weight: 2.0          # GIoU 辅助权重

  augmentation:
    - RandomHorizontalFlip(0.5)
    - RandomResize([480, 512, 544, 576, 608, 640])
    - RandomCrop(min_size=384, max_size=600)

inference:
  num_steps: [1, 2, 4]        # 测试不同步数
  ode_solver: "euler"          # euler / midpoint / rk4
  nms_threshold: 0.7
  score_threshold: 0.3

hardware:
  gpus: 8 × A100 (40GB)
  training_time: ~48h (ResNet-50)
  mixed_precision: fp16
```

### 5.2 框坐标归一化策略

```python
class BoxNormalization:
    """
    关键细节: 框坐标的归一化直接影响扩散/流的质量
    """
    @staticmethod
    def normalize_v1(boxes, image_size):
        """方案1: 归一化到 [0, 1]"""
        boxes[:, 0::2] /= image_size[1]  # x / W
        boxes[:, 1::2] /= image_size[0]  # y / H
        return boxes

    @staticmethod
    def normalize_v2(boxes, image_size):
        """方案2: 归一化到 [-1, 1] (推荐, 与高斯噪声更匹配)"""
        boxes = BoxNormalization.normalize_v1(boxes, image_size)
        boxes = boxes * 2 - 1  # [0,1] → [-1,1]
        return boxes

    @staticmethod
    def normalize_v3(boxes, image_size):
        """方案3: log-ratio 编码 (DiffusionDet的做法)"""
        cx, cy, w, h = box_xyxy_to_cxcywh(boxes).unbind(-1)
        cx = cx / image_size[1]
        cy = cy / image_size[0]
        w = torch.log(w / image_size[1] + 1e-8)
        h = torch.log(h / image_size[0] + 1e-8)
        return torch.stack([cx, cy, w, h], dim=-1)
```

---

## 六、论文结构建议

```
Title: FlowDet: Straightening Generative Object Detection
       with Rectified Flow Matching

Abstract

1. Introduction
   - 生成式检测的兴起 (DiffusionDet)
   - DDPM 弯曲路径导致多步推理的问题
   - 本文贡献: Flow Matching 重构 + 加速策略

2. Related Work
   2.1 Object Detection (DETR 系列)
   2.2 Diffusion-based Perception
   2.3 Flow Matching & Fast Sampling

3. Method
   3.1 Preliminaries: DiffusionDet 回顾
   3.2 Flow Matching for Detection (FlowDet)
       - 线性插值路径
       - 速度场预测
       - 分类联合训练
   3.3 Accelerating FlowDet
       - Rectified Flow (Reflow)
       - Consistency Distillation
       - Adaptive Step Selection
   3.4 Uncertainty Estimation via Multiple Sampling

4. Experiments
   4.1 Setup (COCO, 实现细节)
   4.2 Main Results (Table 1, 2)
   4.3 Speed-Accuracy Trade-off (Fig 3)
   4.4 Ablation Studies (Table 4)
   4.5 Uncertainty Estimation (Table 5)
   4.6 Visualization (Fig 5, 6)

5. Discussion
   - 何时使用生成式检测 vs 判别式检测
   - 局限性与未来方向

6. Conclusion

Appendix:
   A. 详细推导: Flow Matching → Detection 的数学形式
   B. 更多可视化
   C. 计算开销详细分析
```

---

## 七、风险评估与应对

```
┌─────────────────────────┬─────────┬───────────────────────────┐
│ 风险                     │ 概率    │ 应对策略                    │
├─────────────────────────┼─────────┼───────────────────────────┤
│ 1步精度大幅下降           │ 中高    │ 保留2步作为backup;          │
│                         │         │ 结合GIoU辅助损失稳定训练     │
├─────────────────────────┼─────────┼───────────────────────────┤
│ 仍然追不上DINO-DETR精度  │ 高      │ 转向卖"不确定性"和           │
│                         │         │ "统一框架"的故事             │
├─────────────────────────┼─────────┼───────────────────────────┤
│ Reflow训练不稳定         │ 中      │ 用Consistency蒸馏替代;      │
│                         │         │ 渐进式Reflow (逐步拉直)     │
├─────────────────────────┼─────────┼───────────────────────────┤
│ 训练资源不足             │ 低中    │ 先用ResNet-50小规模验证;    │
│                         │         │ 利用DiffusionDet预训练初始化│
├─────────────────────────┼─────────┼───────────────────────────┤
│ 同期出现类似工作          │ 中      │ 强调加速方法的系统性对比;    │
│                         │         │ 不确定性分析作为差异化        │
└─────────────────────────┴─────────┴───────────────────────────┘
```

---

## 八、时间线建议

```
月份    任务                                     里程碑
────────────────────────────────────────────────────────
M1W1-2  复现 DiffusionDet, 对齐指标              ✓ COCO AP=46.1±0.3
M1W3-4  实现 FlowDet 基础版                     ✓ 训练跑通, 有合理AP

M2W1-2  FlowDet 调优 + 消融实验                  ✓ 4步AP ≥ DiffusionDet
M2W3-4  不同步数实验 + ODE求解器对比              ✓ AP-步数曲线

M3W1-2  Rectified Flow (Reflow) 实现             ✓ 1步AP提升
M3W3-4  Consistency Distillation 实现             ✓ 两种加速方法对比

M4W1-2  不确定性估计实验                          ✓ 校准曲线
M4W3-4  Swin backbone / 更多消融                 ✓ 完整实验表格

M5W1-2  可视化 + 补充实验                         ✓ 所有图表完成
M5W3-4  论文初稿撰写                              ✓ 初稿完成

M6W1-4  论文修改 + 投稿                           ✓ 提交
```

这份技术路线覆盖了从理论推导到代码实现到实验设计的完整链条。核心建议是：**先快速跑通基础版本验证可行性，再逐步加入加速策略，最后用不确定性估计作为差异化竞争力。**