# LDMDet 架构演进与实验记录

## 1. 当前分支架构 (Phase 2.5: DiT-Deformable-Heavy)
**设计目标**：将 Sora/DiT 的生成能力与 Deformable DETR 的局部采样能力强行缝合，解决扩散模型在检测中的信息截断问题。

### 核心组件
- **BoxTokenizer** (动态采样模式):
  - 逻辑：根据当前时刻 $B_t$ 的中心点，通过 `F.grid_sample` 在 FPN 特征图上执行双线性插值采样。
  - 嵌入：采样特征 + 坐标线性嵌入 (`bbox_pos_embed`) + 层级嵌入。
- **DiTBlock**:
  - 包含 Self-Attention 和 Multi-Scale Deformable Cross-Attention。
  - 使用 AdaLN-Zero (9参数) 进行时间条件注入。
- **Conditioning**: 默认开启零初始化，保护预训练特征。

### 实验结论 (Failure Diagnosis)
- **现象**：`grad_norm` 飙升至 1600+，`loss_bbox` 维持在 6.0 左右，val mAP 持续为 0。
- **根本原因分析**：
  1. **采样冷启动**：由于初期 Box 是随机噪声，Tokenizer 采样的全是背景特征，导致模型“失明”。
  2. **坐标系偏移**：之前未考虑 `pad_shape` 对 FPN 特征图归一化的影响，导致采样点位移。
  3. **梯度阻断**：AdaLN-Zero 在初期屏蔽了所有残差信号，导致坐标更新无法回传。
- **已实施的补救措施**：
  - 统一坐标归一化到 `pad_shape`。
  - 关闭 AdaLN-Zero 零初始化 (`use_adaln_zero=False`)。
  - 延长预热至 20 Epoch，LR 降至 5e-5。
  - 放宽匹配半径 `center_radius=4.0`。

---

## 2. 规划：回归 Sora 风格 (Phase 3: Sora-Pure-DiT)
**设计哲学**：减少对手工设计采样逻辑的依赖，信任 Transformer 的全局注意力与强大的位置编码。

### 核心变更方向
- **Tokenizer 降级 (Lightweight)**:
  - 移除 `F.grid_sample` 中心点采样。
  - 理由：动态采样会导致梯度流极其不稳定，且对细长目标（染色体）容错率极低。
  - 新逻辑：仅保留坐标的强位置编码，将“寻找特征”的任务完全交给 Cross-Attention。
- **RoPE (Rotary Positional Embedding)**:
  - 引入旋转位置编码替代简单的线性投影。
  - 优势：更好地处理相对位置关系，提升模型对 BBox 细微挪动的敏感度。
- **AdaLN-Zero 调优**:
  - 保持“非零初始化”状态，确保训练初期有信号流过。

---

## 3. 待切换方向记录 (Research Backlog)
若 Sora 风格仍不收敛，备选切换路径：
1. **ROI-Tokenizer**: 使用 RoIAlign 替代点采样，提取区域特征。
2. **GlobalQuotaLoss**: 针对染色体分布不均，引入全局正样本配额限制。
3. **Rectified Flow Refinement**: 切换到更高级的 1-step 推理加速器（如 EDM2 风格）。
