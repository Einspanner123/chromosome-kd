<thinking>
The user is asking for a detailed technical roadmap and experimental design for implementing a Flow Matching-based object detection model as a research project. This is a very practical and detailed question. Let me provide a comprehensive plan covering architecture design, training strategy, experimental setup, ablation studies, and paper writing strategy.

</thinking>

# 基于 Flow Matching 的目标检测：完整技术路线与实验设计

---

## 一、研究定位与论文故事线

### 1.1 建议的研究定位

```
标题方向 (供参考):
  "FlowDet: Efficient Object Detection via Rectified Flow Matching"
  "Straightening Detection: From Diffusion to Flow Matching for Object Detection"
  "One-Step Detection with Conditional Flow Matching"

核心 Claim:
  ┌────────────────────────────────────────────────────────────┐
  │  1. Flow Matching 的直线传输路径天然适合低维框坐标的去噪     │
  │  2. 结合 Reflow/蒸馏, 实现 1-2 步高质量检测                │
  │  3. 速度大幅超越 DiffusionDet, 精度持平或超越               │
  │  4. 保留生成式检测的独有优势 (不确定性, 动态框数量)          │
  └────────────────────────────────────────────────────────────┘
```

### 1.2 故事线设计

```
引言逻辑链:

P1: 生成式检测 (DiffusionDet) 有独特优势
    → 不确定性量化, 动态框数量, 统一框架

P2: 但推理速度是核心瓶颈 (多步去噪)
    → 限制了实际应用

P3: 扩散加速领域有大量进展 (Flow Matching, Consistency Model)
    → 但尚未系统性地应用到检测任务

P4: 检测任务有独特性质 (低维信号, 条件集合生成)
    → 直接迁移可能不 work, 需要针对性设计

P5: 我们提出 FlowDet
    → 针对检测任务设计的 Flow Matching 框架
    → 实现 1-2 步推理, 速度提升 3-4×, 精度持平
```

---

## 二、整体架构设计

### 2.1 系统总览

```
┌──────────────────────────────────────────────────────────────┐
│                        FlowDet 架构                          │
│                                                              │
│  输入图像 ──▶ Backbone+FPN ──▶ 多尺度特征图 F                  │
│                                    │                         │
│                                    ▼                         │
│  噪声框 b₀~N(0,1) ──▶ ┌─────────────────────┐              │
│          t~U(0,1) ──▶  │  Flow Matching Head  │              │
│                        │                     │              │
│                        │  RoI Align(b_t, F)  │              │
│                        │       ↓             │              │
│                        │  Transformer Decoder │              │
│                        │       ↓             │              │
│                        │  预测 v_θ(b_t, F, t) │              │
│                        └─────────┬───────────┘              │
│                                  │                           │
│                    ┌─────────────┼─────────────┐             │
│                    ▼             ▼             ▼             │
│               框坐标速度     类别logits     (可选)mask        │
│               Δb ∈ R^4      cls ∈ R^C      特征             │
│                                                              │
│  推理: b₁ = b₀ + Σ v_θ(b_t, F, t)·Δt                       │
│        (1步: b₁ = b₀ + v_θ(b₀, F, 0))                      │
└──────────────────────────────────────────────────────────────┘
```

### 2.2 核心模块详细设计

#### 模块 A：Backbone + FPN（复用成熟方案）

```python
# 建议方案: 与DiffusionDet保持一致, 方便对比
class FlowDetBackbone(nn.Module):
    def __init__(self, backbone_type='resnet50'):
        # 方案1: ResNet-50 + FPN (基线, 对标DiffusionDet)
        # 方案2: Swin-Base + FPN (更强backbone)
        # 方案3: ResNet-50 + 简化FPN (速度优先)
        
        self.backbone = build_backbone(backbone_type)  
        self.fpn = FPN(in_channels=[256,512,1024,2048], 
                       out_channels=256)
    
    def forward(self, images):
        features = self.backbone(images)      # 多尺度特征
        fpn_features = self.fpn(features)     # P2-P5
        return fpn_features  # 只计算一次, 所有去噪步共享
```

#### 模块 B：Flow Matching 前向过程

```python
class FlowMatchingScheduler:
    """核心: 线性插值构造训练样本"""
    
    def __init__(self, sigma_min=1e-4):
        self.sigma_min = sigma_min
    
    def sample_training_pair(self, b_gt, num_proposals):
        """
        b_gt: