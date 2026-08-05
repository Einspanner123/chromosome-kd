# 路径 2：DN-DETR 风格去噪训练 + RF-DETR (RF-Continuous Denoising)

> **核心思想**: 将 DN-DETR / DINO-CDN 的去噪训练机制与 Rectified Flow (RF) 的连续噪声路径结合, 在 RF-DETR 基线上以**训练时辅助任务**的形式引入 RF-CDN (RF-Continuous Denoising)。推理时去噪 query 被完全移除, **零推理开销**, 同时利用 RF 的连续噪声谱改进 CDN 固定小噪声 λ 的局限。
>
> **理论依据**:
> - DN-DETR (CVPR 2022 Oral, arXiv:2203.01305) [1]: 去噪训练稳定二分图匹配, ResNet-50 +1.9 AP, 50% epoch 达基线
> - DINO (ICLR 2023, arXiv:2203.03605) [2]: Contrastive DeNoising (CDN), 正/负样本去噪, +2.7 AP (含 CDN + Mixed Query Selection + Look Forward Twice 三组件, 不可全归因 CDN)
> - RF-DETR (ICLR 2026, arXiv:2511.09554) [3]: Roboflow 实时 DETR, **架构血统为 LW-DETR + DINOv2 backbone + AIFI/CCFM (RT-DETR/D-FINE 系)**, **非 DINO 后代**, 是否内置 CDN 风格去噪需先核实 (见阶段 0)
> - Rectified Flow (Liu et al., ICLR 2023) [4]: $x_t = (1-t)x_0 + t x_1$ 直线路径
> - LDMDet RF 实现: [ldmdet/diffusion/rectified_flow.py](../../ldmdet/diffusion/rectified_flow.py) — 已验证 RF + Heun + AdaLN-Zero + StochasticOT 路径
>
> **重要前置说明 (RF-DETR 架构血统核实)**: 经 WebSearch 核实 RF-DETR 官方声明 (Roboflow 博客与 arXiv:2511.09554), RF-DETR 系 **LW-DETR 现代化改造 + DINOv2 backbone**, 架构组件为 AIFI + CCFM (RT-DETR/D-FINE 血统), 代码库基于 PyTorch Lightning 独立实现, **不依赖 mmdet, 也并非 DINO/CdnQueryGenerator 直接继承**。因此本文档原版本中所有 "RF-DETR 已沿用 DINO CDN" / "decoder 直接继承自 DINO" / "CDN 实现等同于 mmdet/.../dino_layers.py" 的表述均不成立, 已在本次修订中删除。本方案的核心改动应理解为 **"在 RF-DETR 现有去噪机制 (无论何种形式, 见阶段 0 核实) 基础上引入 RF 连续噪声路径"**, 而非 "替换 DINO CDN"。
>
> **与已证伪方向的关系**:
> - 与 CFM (方向 H, mAP=0.823) 的区别: 本方案预测 $x_0$ 而非速度 $v$, 且仅训练时去噪无采样循环
> - 与 SC-RF (mAP=0.860) 的区别: 本方案无自条件化, 去噪 query 与匹配 query 用 attention mask 隔离
> - 与 PD-RF (best mAP=0.851) 的区别: 本方案推理时无扩散, 不存在蒸馏目标偏移问题
> - 与 LDMDet 主路线 a3_full_sota (mAP=0.858) 的区别: 本方案推理零开销, 训练时去噪是辅助任务而非主任务
>
> **当前代码位置** (LDMDet RF 实现参考, **不作为 RF-DETR 的依赖**):
> - [ldmdet/core/head.py](../../ldmdet/core/head.py) — `DiffusionDetHead.loss` (训练目标构建) 与 `predict` (推理多步采样)
> - [ldmdet/core/single_head.py](../../ldmdet/core/single_head.py) — `SingleDiffusionDetHead` (单步扩散头, AdaLN-Zero 时间条件化)
> - [ldmdet/diffusion/rectified_flow.py](../../ldmdet/diffusion/rectified_flow.py) — `RectifiedFlow.q_sample` (前向加噪) 与 `heun_step` (采样), 本方案仅复用 `q_sample` 公式
> - **RF-DETR 官方代码库**: [github.com/roboflow/rf-detr](https://github.com/roboflow/rf-detr) (PyTorch Lightning, Apache 2.0) — 本方案需 fork 此库, **不依赖 mmdet**
> - mmdet 的 `CdnQueryGenerator` (小写 d, 见 M7 修订) 仅作为 **CDN 算法参考实现** (用于对照与算法理解), **RF-DETR 并不直接使用 mmdet**

---

## 1. 摘要

本方案提出 RF-Continuous Denoising (RF-CDN), 将 Rectified Flow 的连续噪声路径 (在 **logit 空间** 加噪: $z_t = (1-t)\,\text{logit}(x_0) + t\,\epsilon$, $x_t = \sigma(z_t)$) 嵌入 DN-DETR / DINO 的去噪训练框架, 作为 RF-DETR 的训练时辅助任务。**前置说明**: RF-DETR 架构血统为 LW-DETR + DINOv2 + AIFI/CCFM (RT-DETR/D-FINE 系), **并非 DINO 后代**, 其去噪机制的具体形式 (LW-DETR 风格 / RT-DETR 风格 / 无去噪) 须在阶段 0 核实。无论 RF-DETR 现有去噪机制为何种形式, 本方案的目标都是 **改进其噪声生成路径为 RF 连续时间 $t$**, 正样本 $t \sim U[0, t_{\text{pos}}]$、负样本 $t \sim U[t_{\text{neg}}, t_{\text{neg\_max}}]$ (上限 $< 1$, 避免梯度信号趋零, 见 §3.3.1), 去噪目标统一为 $\hat{x}_0$ 预测 (避开 CFM 速度预测的失败模式)。logit 空间加噪保证 $w, h > 0$ 且与 RF-DETR 的 inverse_sigmoid 嵌入一致 (见 §3.3.1 S2 修复)。推理时去噪 query 完全移除, 等同 RF-DETR baseline, 零推理开销。**保守预期** 在 24obj 数据集上达到 mAP ≥ baseline, 理想情况下 +0~0.005 (见 §6.2, §7.1 统一估计, 须 3-seed 验证), 推理延迟 = RF-DETR baseline。

---

## 2. 问题动机

### 2.1 RF-DETR 的去噪机制须首先核实

**事实核实 (基于 WebSearch + Roboflow 官方博客 + arXiv:2511.09554)**:

RF-DETR [3] 由 Roboflow 提出, 在 ICLR 2026 发表。其架构血统为:
- **Backbone**: DINOv2 ViT (RF-DETR-L 为 ViT-L, RF-DETR-S 为 ViT-B), 冻结或可训练
- **架构基础**: **LW-DETR 现代化改造** (官方博客原文: "we created RF-DETR by combining LW-DETR with a pre-trained DINOv2 backbone")
- **Encoder/Neck**: AIFI (Attention-in-Attention) + CCFM (Cross-scale Feature-fusion Module), 属 **RT-DETR / D-FINE 血统**
- **训练框架**: PyTorch Lightning, **代码库独立于 mmdet** (见 [github.com/roboflow/rf-detr](https://github.com/roboflow/rf-detr))

**关键澄清 (修订前文档的错误声明)**:
- ❌ ~~"RF-DETR 已沿用 DINO 的 CDN 去噪训练"~~ — 不成立, RF-DETR 并非 DINO 后代
- ❌ ~~"decoder 直接继承自 DINO"~~ — 不成立, RF-DETR decoder 源自 LW-DETR / RT-DETR 系
- ❌ ~~"CDN 实现等同于 mmdet/.../dino_layers.py 的 `CDNQueryGenerator`"~~ — 不成立, RF-DETR 不依赖 mmdet (且 mmdet 实际类名为 `CdnQueryGenerator` 小写 d)

**RF-DETR 现有去噪机制的三种可能性 (须在阶段 0 核实)**:

1. **可能性 A (LW-DETR 风格去噪)**: LW-DETR 原文 (arXiv:2406.03459) 描述了 "noisy query" 去噪训练, 但其实现与 DINO CDN 在 group 机制、attention mask、正负样本划分上**不完全一致**。若 RF-DETR 沿用 LW-DETR 去噪, 则本方案需适配 LW-DETR 的 query 组织方式。
2. **可能性 B (RT-DETR 风格无显式去噪)**: RT-DETR 原文未采用 DN-DETR/CDN 风格去噪, 而是用 "uncertainty-minimal query selection" 稳定匹配。若 RF-DETR 沿用此路线, 则 **RF-DETR 本身无去噪机制**, 本方案相当于 "新增" 而非 "替换"。
3. **可能性 C (RF-DETR 自定义去噪)**: RF-DETR 可能在 LW-DETR 基础上做了去噪机制的修改, 须读源码确认。

**阶段 0 (新增, 见 §6.0) 的核实任务**:
- 阅读 `rf-detr` 源码, 定位去噪 query 生成模块 (若存在)
- 确认噪声分布 (均匀 / 高斯 / 其他)、噪声尺度 (固定 λ / 自适应)、group 机制、attention mask 设计
- 若 RF-DETR 无去噪机制 (可能性 B), 则本方案从 "替换 CDN" 调整为 **"为 RF-DETR 新增 RF-CDN 去噪辅助任务"**, 算法核心不变, 但工程改动更大
- 若 RF-DETR 有去噪机制 (可能性 A/C), 则本方案为 **"改进 RF-DETR 现有去噪机制的噪声路径"**

**本方案的下文 (§3 及之后) 以 "假设 RF-DETR 有某种去噪机制" 为前提撰写**, 所引用的 mmdet `CdnQueryGenerator` 仅作为 **CDN 算法的标准参考实现** (不引用具体文件路径/行号 — RF-DETR 不使用 mmdet, 行号无意义), 用于说明 RF-CDN 与 CDN 的算法差异, **不表示 RF-DETR 使用 mmdet**。阶段 0 完成后, 下文的接口与代码示例可能需要按 RF-DETR 实际去噪模块的形式调整。

### 2.2 CDN 的局限性 (作为算法对照, 不假设 RF-DETR 必然使用 CDN)

虽然 CDN 显著加速了 DETR 收敛 (DN-DETR [1] 报告 +1.9 AP; DINO [2] 在 DN-DETR 基础上 +2.7 AP, **注意此 +2.7 AP 含 CDN + Mixed Query Selection + Look Forward Twice 三个组件, 不可全归因 CDN**), 但其噪声设计存在三个结构性局限。这些局限对本方案的意义在于: 无论 RF-DETR 现有去噪机制是 CDN 风格还是 LW-DETR 风格, 只要使用**固定噪声尺度**, 以下局限均成立。

**局限 1: 噪声尺度固定, 不利用连续噪声谱 (诚实声明: 此为假设, 需实验验证)。**
CDN 的 $\lambda_1, \lambda_2$ 是固定超参数 (DINO 默认 $\lambda_1 = 0.5$, $\lambda_2 = 1.0$, 相对归一化框坐标), 训练全程不变。这意味着:
- 模型只在两个固定噪声尺度上学习去噪, 对中间尺度 (如 $\lambda = 0.7$) 的泛化依赖网络的插值能力。
- **诚实弱化声明**: 连续 $t$ **可能**提供更丰富的噪声谱, 但**也可能稀释学习信号** (固定 λ 集中学习单一尺度, 连续 t 分散到整个区间, 单点信号强度降低)。是否能从连续噪声谱中净获益, 需阶段 3 消融实验 (连续 vs 单点) 验证, 不能先验断言。

**局限 2: 单步重建缺乏难度梯度 (诚实声明: 此为假设, 需实验验证)。**
CDN 的正样本去噪是 "加小噪声 → 单步重建 GT" 的简单任务, 难度恒定。当模型收敛到一定程度后, 这个辅助任务的梯度信号减弱, 对主任务的 regularization 效果递减。RF 的连续 $t$ **可能**提供从易 ($t \to 0$, $x_t \approx x_0$) 到难 ($t \to 1$, $x_t \approx \text{noise}$) 的难度梯度, **但大 $t$ 区间的梯度信号趋零问题 (见 §3.3.1 M3 分析) 可能抵消此优势**。是否能持续提供有用训练信号, 需实验验证。

**局限 3: 正/负样本边界硬切, 不感知数据流形。**
CDN 用固定的 $\lambda_1 < \lambda_2$ 划分正负, 但 "小噪声 = 正样本, 大噪声 = 负样本" 的假设在某些数据分布下不一定成立 (例如染色体这类细长目标, 框的小偏移可能已导致 IoU 急剧下降)。RF 的连续 $t$ 配合 IoU 自适应阈值可更平滑地处理这一边界。

### 2.3 本方案: 用 RF 风格的连续噪声尺度改进 RF-DETR 现有去噪机制

本方案的核心改进是: **保留 RF-DETR 的完整架构与其现有去噪机制的 group + attention mask 框架 (具体形式由阶段 0 核实), 将其固定 $\lambda_1, \lambda_2$ 噪声 (若存在) 改进为 RF 的连续时间 $t$ 噪声路径**。若阶段 0 核实发现 RF-DETR 无显式去噪机制 (可能性 B), 则本方案为 **新增 RF-CDN 去噪辅助任务**。具体而言:

- 正样本: $t \sim U[0, t_{\text{pos}}]$, **logit 空间加噪** $z_t = (1-t)\,\text{logit}(\text{GT}) + t\,\epsilon$, $x_t = \sigma(z_t)$ (见 §3.3.1 S2 修复, 保证 $w,h > 0$), 目标为预测 $\hat{x}_0 \to \text{GT}_{cxcywh}$
- 负样本: $t \sim U[t_{\text{neg}}, t_{\text{neg\_max}}]$ (**上限 $t_{\text{neg\_max}} < 1$, 避免梯度信号趋零**, 见 §3.3.1 M3), 同样的 $x_t$ 公式, 目标为预测 "无目标" (沿用 CDN 负样本语义)
- 间隔 $[t_{\text{pos}}, t_{\text{neg}}]$ 避免正负样本噪声尺度重叠 (类比 CDN 的 $\lambda_1 < \lambda_2$)

这一改动**仅影响训练时的 query 生成**, 推理时去噪 query 被移除, 与 RF-DETR baseline 完全一致。

### 2.4 推理零开销的优势

相比路径 1 (LDMDet 主路线, 推理时多步 Heun 采样, 4 步 = 8 NFE), 本方案的关键优势是**推理零开销**:

| 方案 | 训练开销 | 推理开销 | 推理 NFE |
|------|---------|---------|---------|
| LDMDet a3_full_sota | 单步去噪主任务 | Heun 4 步 + box_renewal | 8 |
| RF-DETR baseline | CDN (固定噪声) | 单次 decoder 前向 | 1 |
| **本方案 (RF-CDN)** | CDN (连续噪声) + 主任务 | **单次 decoder 前向** | **1** |

在染色体核型分析的临床部署场景中, 推理延迟是硬约束 (单核型 24 张图需在秒级完成)。本方案在保持训练时去噪增益的同时, 推理延迟 = RF-DETR baseline, 显著优于路径 1 的多步采样。

---

## 3. 核心理论

### 3.1 DN-DETR / CDN 去噪训练机制详解

**DN-DETR [1] 的核心机制**:

DN-DETR 指出 DETR 慢收敛的根因是二分图匹配 (Hungarian matching) 在训练早期的不稳定性 — 同一 GT 在不同 epoch 可能被匹配到不同 query, 导致优化目标不一致。DN-DETR 的解决方案是引入去噪辅助任务:

1. **去噪 query 生成**: 对每个 GT 框 $b_{gt}$, 加入受控噪声 $\delta$ ($|\Delta x| < \lambda w / 2$ 等), 得到 $b_{noisy} = b_{gt} + \delta$。同时给 GT label 加入翻转噪声 (以概率 $p$ 翻转为随机类)。
2. **去噪 query 输入 decoder**: 将 $b_{noisy}$ 作为 4D anchor box (position query), 翻转后的 label embedding 作为 content query, 送入与匹配 query 共享的 decoder。
3. **重建损失**: 去噪 query 的输出直接监督回原始 GT (框 + label), **不经过 Hungarian matching**, 提供稳定监督信号。
4. **Attention mask 隔离**: 去噪 query 不可见匹配 query (防止信息泄漏), 不同去噪 group 之间也不可见。用 indicator 区分两类 query。
5. **推理移除**: 推理时去噪 query 完全移除, 模型行为等同 baseline。

**DINO CDN [2] 的改进**:

DINO 在 DN-DETR 基础上引入对比去噪 (Contrastive DeNoising), 关键改动是**双噪声尺度 + 正负样本**:

- 正样本 ($\lambda_1$ 小噪声): 重建 GT (与 DN-DETR 相同)
- 负样本 ($\lambda_2 > \lambda_1$ 大噪声): 预测 "无目标" (背景类), 强制模型学会区分 "近 GT 但非 GT" 的框

CDN 的去噪 group 数量通常为 5 (每个 GT 生成 5 正 + 5 负 = 10 去噪 query), attention mask 确保 group 间隔离。CDN 显著减少了 DETR 的重复检测问题 (DINO 论文 Table 1 报告 CDN 贡献 +0.4 AP over DN)。

**CDN 的数学形式化 (S1 修订: 均匀分布 + 硬截断, 非高斯)**:

设 GT 框集合 $\mathcal{G} = \{(b_i, c_i)\}_{i=1}^{N_{gt}}$, 噪声尺度 $\lambda_1 < \lambda_2$。CDN 生成两组 query:

$$
\begin{aligned}
\text{Positive:} \quad & b_i^{pos} = b_i + \delta_i^{pos}, \quad \delta_i^{pos} \sim \text{Uniform}(-\lambda_1 s_i, \lambda_1 s_i), \quad \text{target} = (b_i, c_i) \\
\text{Negative:} \quad & b_i^{neg} = b_i + \delta_i^{neg}, \quad \delta_i^{neg} \sim \text{Uniform}(-\lambda_2 s_i, \lambda_2 s_i), \quad \text{target} = \varnothing
\end{aligned}
$$

其中 $s_i = (w_i, h_i, w_i, h_i) / 2$ 是各分量的尺度归一化 (确保噪声与框大小成比例)。**关键修正**: CDN 实际使用 **均匀分布** 而非高斯分布, 见 mmdet 的 `CdnQueryGenerator` (算法参考实现, **RF-DETR 不依赖 mmdet**, 阶段 0 核实 RF-DETR 实际实现) 中用 `(torch.rand_like(x) * 2 - 1) * scale` 形式生成均匀噪声。**硬截断约束** $|\Delta x| < \lambda w / 2$ 等保证正样本中心仍在 GT 框内 (避免正样本退化为负样本)。

> **修订说明 (S1)**: 修订前文档将 CDN 噪声误写为高斯 $\mathcal{N}(0, \lambda^2 \Sigma)$, 实际 DINO/CDN 用均匀分布 + 硬截断。此修正影响 §3.3.1 中 RF-CDN 与 CDN 的对比 — RF-CDN 用高斯 $\epsilon$ 经线性插值, 与 CDN 的均匀噪声**不可直接对应** (见 §3.3.1 M2 修订)。

### 3.2 RF-DETR 现有去噪机制的具体实现 (待阶段 0 核实)

**修订说明 (C1)**: 修订前文档声称 "RF-DETR 的 decoder 直接继承自 DINO, 其 CDN 实现等同于 mmdet/.../dino_layers.py 的 `CDNQueryGenerator`"。此声明**不成立** (RF-DETR 为 LW-DETR + DINOv2 血统, 非 DINO 后代, 不依赖 mmdet; 且 mmdet 实际类名为 `CdnQueryGenerator` 小写 d, 见 M7)。以下内容为 **CDN 算法的标准参考实现描述** (基于 mmdet 的 `CdnQueryGenerator`, **具体行号不引用** — RF-DETR 不使用 mmdet, 行号无意义), 用于在阶段 0 核实 RF-DETR 实际去噪机制后进行对照。**RF-DETR 的实际去噪实现可能与此不同 (LW-DETR 风格 / RT-DETR 风格 / 自定义 / 无)**。

**CDN 标准参考实现** (mmdet `CdnQueryGenerator`, 作为算法对照; 具体实现须从 [rf-detr 源码](https://github.com/roboflow/rf-detr) 阶段 0 核实):

1. **label embedding**: `nn.Embedding(num_classes, embed_dims)`, 对 noisy label 查表。
2. **label noise**: 以 `label_noise_scale * 0.5` 比例随机翻转 label。
3. **bbox noise**: 对归一化 (cx, cy, w, h) 框, 在每个分量上加**均匀噪声**, 正样本在 `box_noise_scale` 内, 负样本在 `[box_noise_scale, 2*box_noise_scale]` 内。
4. **group 机制**: 每个 GT 生成 `2 * num_groups` 个去噪 query (正负各 `num_groups`), `num_groups` 动态计算保证总去噪 query 数 ≤ `num_dn_queries`。
5. **attention mask**: `generate_dn_mask` 生成 block-diagonal mask, 去噪 group 间 + 去噪/匹配间隔离。
6. **dn_meta**: 记录 `num_denoising_queries` 和 `num_denoising_groups`, 用于 loss 计算时分离去噪和匹配部分。

**关键超参数** (DINO 默认, **RF-DETR 是否沿用未经验证**):
- `num_dn_queries = 100` (去噪 query 总数上限)
- `box_noise_scale = 0.4` (即 $\lambda_1 = 0.4$)
- `label_noise_scale = 0.5`
- `num_groups = 5` (静态) 或动态

> **修订说明 (M9)**: 修订前文档声称 "RF-DETR 沿用 DINO box_noise_scale=0.4"。此为**未经验证的假设** — RF-DETR 基于 LW-DETR, 其去噪超参数 (若有) 应来自 LW-DETR 而非 DINO。须在阶段 0 从 `rf-detr` 源码核实实际数值。

### 3.3 本方案: RF-Continuous Denoising (RF-CDN)

本方案将 CDN 的固定 $\lambda_1, \lambda_2$ 噪声替换为 RF 的连续时间 $t$ 噪声路径, 同时保留 CDN 的 group + attention mask + 正负样本机制。

#### 3.3.1 噪声采样 (RF 线性路径)

采用 LDMDet 已验证的 RF 前向扩散公式 (见 [rectified_flow.py](../../ldmdet/diffusion/rectified_flow.py) 的 `q_sample`), 但**在 logit 空间加噪**以保证 $w, h > 0$ (S2 修复, 见下文):

$$
\begin{aligned}
z_0 &= \text{logit}(x_0) = \log\frac{x_0}{1 - x_0} \\
z_t &= (1 - t) \cdot z_0 + t \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I) \\
x_t &= \text{sigmoid}(z_t)
\end{aligned}
$$

其中 $x_0 = \text{GT}_{cxcywh}$ (归一化 cxcywh 框, 范围 $[0, 1]$), $\epsilon$ 是高斯噪声, $t \in [0, 1]$ 是连续时间, $\sigma = 1.0$。

> **修订说明 (S2, 关键修复)**: 修订前文档直接在 $[0, 1]$ cxcywh 空间加噪 `x_t = (1-t)·x_0 + t·ε`, 其中 `ε ~ N(0,1)`。这在 $t > 0.5$ 时大量 $w, h$ 分量为负数 (GT ∈ [0,1], ε ~ N(0,1) 在 t=1 时完全覆盖负值), 导致框退化。**改为 logit 空间加噪**: 先将 $x_0 \in [0,1]$ 映射到 $z_0 \in (-\infty, +\infty)$, 在无界空间线性插值后再 sigmoid 回 $[0,1]$, **保证 $w, h > 0$**。此外, logit 空间与 RF-DETR decoder 的 `inverse_sigmoid` 框嵌入一致 (RF-DETR 的 position query 经 `inverse_sigmoid` 嵌入, 见阶段 0 核实), 加噪空间与嵌入空间对齐, 梯度路径更自然。

**与 LDMDet 主路线的关键区别**: LDMDet 主路线 (a3_full_sota) 的 $\sigma = \text{snr\_scale} = 2.0$ 且 $x_0$ 经过 `gt_diffusion = (norm_gt_cxcywh * 2 - 1) * snr_scale` 非线性映射 (见 [head.py](../../ldmdet/core/head.py) 的 `loss` 方法)。本方案用 logit 映射替代 snr_scale 重映射, 二者目的相同 (将 $[0,1]$ 框映射到无界空间以适配高斯加噪), 但 logit 映射保证 sigmoid 回 $[0,1]$ 后框始终合法。

**噪声尺度采样**:

$$
\begin{aligned}
\text{Positive:} \quad & t \sim U[0, t_{\text{pos}}], \quad \text{target} = (x_0, c) \\
\text{Negative:} \quad & t \sim U[t_{\text{neg}}, t_{\text{neg\_max}}], \quad \text{target} = \varnothing
\end{aligned}
$$

其中 $t_{\text{pos}} < t_{\text{neg}} < t_{\text{neg\_max}} < 1$ 是正负样本边界。建议默认 $t_{\text{pos}} = 0.3$, $t_{\text{neg}} = 0.7$, $t_{\text{neg\_max}} = 0.9$, 中间区间 $[t_{\text{pos}}, t_{\text{neg}}]$ 留空避免歧义 (类比 CDN 的 $\lambda_1 < \lambda_2$)。

> **修订说明 (M3, 关键修复)**: 修订前文档负样本采样 $t \sim U[t_{\text{neg}}, 1]$, 上限为 1。问题: $t \to 1$ 时 $z_t \to \epsilon$ (纯噪声), $x_t = \text{sigmoid}(\epsilon)$ 趋近 $0.5$ 附近的随机框, 预测 "背景" 平凡正确, **梯度信号趋零**。CDN 负样本用有限 $\lambda_2$ (近 GT 大偏移) 作为 hard negative, 是有判别力的样本。**改为有限上限 $t_{\text{neg\_max}} = 0.9$** (而非 1), 使负样本仍保留一定 GT 信号 $(1-t) \cdot z_0$, 成为 hard negative 而非平凡背景。此 $t_{\text{neg\_max}}$ 作为超参数, 阶段 3 消融。

**与 CDN 固定 λ 的关系**:
- CDN 用均匀分布噪声 + 硬截断, RF-CDN 用高斯噪声经 logit 空间线性插值, **两者噪声分布不同, 不存在精确的 $\lambda \leftrightarrow t$ 等价关系**。修订前文档 "CDN 的 $\lambda_1 = 0.4$ 等价于 $t \approx 0.3$" 的粗略类比不严谨, 已删除。
- RF-CDN 的正样本覆盖 $t \in [0, t_{\text{pos}}]$ 的连续区间, **可能**比 CDN 单点 $\lambda_1$ 提供更丰富的噪声难度谱, 但也可能稀释学习信号 (见 §2.2 局限 1 的诚实声明), 需阶段 3 消融验证。
- RF-CDN 的负样本覆盖 $t \in [t_{\text{neg}}, t_{\text{neg\_max}}]$ (有限区间), $x_t$ 含残量 GT 信号, 是 hard negative, 与 CDN 负样本语义一致。

#### 3.3.2 去噪目标: 预测 $\hat{x}_0$ (而非速度 $v$)

**关键设计决策**: 去噪目标为预测 $\hat{x}_0 \to x_0$ (模型预测 $\hat{x}_0$ 逼近真值 $x_0 = \text{GT}$), **不**预测速度 $v = \epsilon - x_0$。

> **修订说明 (M16)**: 修订前文档标题为 "预测 $x_0$", 符号混用 (同一 $x_0$ 既表真值又表预测)。改为区分: $x_0$ = 真值 (GT 框), $\hat{x}_0$ = 模型预测。下方公式与表格同步更正。

**理由 (基于 CFM 失败教训)**:
LDMDet 项目已证伪 CFM 速度预测 (方向 H, mAP=0.823 vs 0.856, -0.033)。失败根因是级联架构下 Head 1-5 输入 ≈ $x_0$ 丢失 $\epsilon$ 信息, 速度损失收敛到 $\text{Var}(\epsilon) = 4$ 注入梯度噪声。本方案虽不在级联架构下, 但同样存在 "去噪 query 输入是 $x_t$ (含 GT 信息), 直接预测 $\hat{x}_0$ 是最短路径监督" 的考量。预测 $\hat{x}_0$ 的优势:

1. **监督信号直接**: $L_{denoise} = \|\hat{x}_0 - x_0\|_1$, 无需速度转换, 梯度路径最短。
2. **与 RF-DETR 现有去噪一致**: CDN 的去噪目标就是重建 GT (即 $\hat{x}_0 \to x_0$), 本方案保持一致。
3. **避开 CFM 失败模式**: 不引入速度预测, 避免 $\text{Var}(\epsilon)$ 梯度噪声问题。

**与 LDMDet RF 的 $\hat{x}_0$ 预测对比**: LDMDet 主路线也预测 $\hat{x}_0$ (见 [head.py](../../ldmdet/core/head.py) 的 `x0 = self._sampler.xyxy_to_raw(...)`), 但其 $\hat{x}_0$ 预测用于多步采样 (Heun step)。本方案的 $\hat{x}_0$ 预测**仅用于单步重建损失**, 无采样循环。

#### 3.3.3 与 DN-DETR 的核心区别

| 维度 | DN-DETR / CDN | 本方案 (RF-CDN) |
|------|---------------|----------------|
| 噪声尺度 | 固定 $\lambda_1, \lambda_2$ (离散双点) | 连续 $t \sim U[0, t_{\text{pos}}] \cup U[t_{\text{neg}}, t_{\text{neg\_max}}]$ |
| 噪声路径 | 加性噪声 $\delta$ (均匀分布, 硬截断) | logit 空间 RF 线性路径 $z_t = (1-t)\text{logit}(x_0) + t\epsilon$, $x_t = \sigma(z_t)$ (高斯) |
| 去噪目标 | 单步重建 GT (即 $x_0$) | 单步预测 $\hat{x}_0$ (相同) |
| 难度梯度 | 恒定 (固定 $\lambda$) | 连续 (从 $t=0$ 易到 $t=t_{\text{neg\_max}}$ 难, 上限 $< 1$ 避免梯度趋零) |
| 正负样本 | $\lambda_1$ 正, $\lambda_2$ 负 (硬切) | $t < t_{\text{pos}}$ 正, $t \in [t_{\text{neg}}, t_{\text{neg\_max}}]$ 负 (连续区间) |
| 推理 | 移除去噪 query | 移除去噪 query (相同) |

### 3.4 Attention Mask 隔离机制

沿用 DN-DETR / CDN 的 attention mask 设计, 确保:
1. **去噪 query 不可见匹配 query**: 防止去噪 query "偷看" 匹配 query 的信息, 保证去噪任务是独立的辅助监督。
2. **不同去噪 group 间不可见**: 每个 group 独立去噪, 防止 group 间信息泄漏。
3. **匹配 query 可见去噪 query**: 这是 DN-DETR 的标准设计, 允许匹配 query 从去噪 query 的 "已知的 GT 锚点" 中获益 (类似软监督)。

Attention mask 的形式 (沿用 DN-DETR / CDN 的 `generate_dn_mask` 标准设计; RF-DETR 实际实现须阶段 0 核实):

$$
M_{ij} = \begin{cases}
\text{True (mask)} & \text{if } i \in \text{DN}(g_1), j \in \text{DN}(g_2), g_1 \neq g_2 \\
\text{True (mask)} & \text{if } i \in \text{DN}, j \in \text{Match} \\
\text{False (visible)} & \text{otherwise}
\end{cases}
$$

其中 $\text{DN}(g)$ 表示去噪 group $g$, $\text{Match}$ 表示匹配 query 集合。注意 $i \in \text{Match}, j \in \text{DN}$ 时 visible (匹配可见去噪)。

### 3.5 与 LDMDet RF 的理论映射

本方案与 LDMDet 主路线的 RF 实现共享**前向加噪公式**, 但在以下维度根本不同:

| 维度 | LDMDet 主路线 (a3_full_sota) | 本方案 (RF-CDN) |
|------|------------------------------|----------------|
| 加噪公式 | $x_t = (1-t)x_0 + t\epsilon$ ([rectified_flow.py](../../ldmdet/diffusion/rectified_flow.py) `q_sample`) | logit 空间: $z_t = (1-t)\text{logit}(x_0) + t\epsilon$, $x_t = \sigma(z_t)$ |
| $x_0$ 空间 | snr_scale 重映射: `gt_diffusion = (norm_gt * 2 - 1) * snr_scale` | logit 映射: $\text{logit}(\text{norm\_gt})$ (保证 sigmoid 回 $[0,1]$ 合法) |
| $t$ 采样 | $t \sim U[0, 1]$ 全区间 (主任务) | $t \sim U[0, t_{\text{pos}}] \cup U[t_{\text{neg}}, t_{\text{neg\_max}}]$ (辅助任务, 上限 $< 1$) |
| 预测目标 | $\hat{x}_0$ (用于 Heun 多步采样) | $\hat{x}_0$ (仅用于单步重建损失) |
| 采样循环 | 推理时 Heun 4 步 (8 NFE) | **无采样循环** (推理 = RF-DETR baseline) |
| 主任务 | 扩散去噪 (主路线) | RF-DETR 匹配检测 (主任务) + RF-CDN 去噪 (辅助) |
| 架构 | 级联 6 个 SingleDiffusionDetHead | RF-DETR 标准 decoder (不动) |

**关键洞察**: 本方案**只在训练时借用 RF 的加噪公式**, 推理时完全不用 RF 采样。这是 "训练时 RF, 推理时零开销" 的设计, 与 LDMDet "训练+推理都 RF" 形成对比。

---

## 4. 与已证伪方向的区分

本节明确区分本方案与 LDMDet 项目已证伪的五个方向, 解释为何本方案不会重蹈覆辙。

### 4.1 与 CFM (方向 H, mAP=0.823) 的区别

**CFM 失败原因** (见 [方向H_FlowMatching检测.md](../archived/FlowMatching检测.md)):
- 级联架构下 Head 1-5 输入 ≈ $x_0$ (因为前一个 head 已重建), 丢失 $\epsilon$ 信息
- 速度损失 $L_v = \|v_{pred} - (\epsilon - x_0)\|^2$ 收敛到 $\text{Var}(\epsilon) = 4$ (因 snr_scale=2), 注入梯度噪声
- 速度预测在级联架构下不可学

**本方案为何不适用此失败模式**:
1. **预测 $x_0$ 而非 $v$**: 本方案去噪目标是 $x_0$ (GT 框), 不是速度 $v$, 完全避开速度损失的 $\text{Var}(\epsilon)$ 问题。
2. **无级联架构**: 本方案基于 RF-DETR (单 decoder, 非 6 头级联), 不存在 "Head 1-5 输入 ≈ $x_0$" 的问题。
3. **无 snr_scale 重映射**: 本方案在 RF-DETR 归一化坐标空间操作, 不引入 snr_scale=2 的方差放大。
4. **仅训练时**: CFM 是主任务 (训练 + 推理都用速度), 本方案是辅助任务 (训练时去噪, 推理时无)。

### 4.2 与 PD-RF (best mAP=0.851, v4 崩塌至 0.252) 的区别

**PD-RF 失败原因** (4→1 蒸馏):
- 推理时蒸馏目标 (4 步教师 → 1 步学生) 偏移, 学生无法匹配教师的多步轨迹
- v4 灾难性崩塌 (mAP=0.252) 因蒸馏损失与主任务损失冲突

**本方案为何不适用此失败模式**:
1. **推理时无扩散**: 本方案推理时等同 RF-DETR baseline, 不存在 "1 步学生 vs 4 步教师" 的蒸馏目标偏移问题。
2. **无教师网络**: 本方案不需要训练教师, 不存在蒸馏损失。
3. **训练时辅助任务与主任务一致**: 去噪目标 ($x_0$ 重建) 与主任务 (检测框回归) 在坐标空间一致, 不冲突。

### 4.3 与 SC-RF (mAP=0.860 < 0.862, -0.002) 的区别

**SC-RF 失败原因** (Self-Conditioning RF):
- 自条件化要求模型把上一步的 $x_0$ 预测作为当前步输入, 引入训练-推理不一致 (推理时用模型自身预测, 训练时用 GT)
- 在 4 维框扩散上, 自条件化的额外信息量有限, 负增益 -0.002

**本方案为何不适用此失败模式**:
1. **无自条件化**: 本方案去噪 query 的输入是 $x_t$ (加噪 GT), 不是模型自身的 $x_0$ 预测, 不存在训练-推理不一致。
2. **去噪 query 与匹配 query 隔离**: attention mask 确保去噪 query 不依赖匹配 query 的预测, 信息流清晰。

### 4.4 与 ScaleConditionedRF (mAP=0.741 < 0.746) 的区别

**ScaleConditionedRF 失败原因**:
- 用框尺度条件化时间步采样, 但小框和大框的最优 $t$ 分布不同, 单一条件化函数无法覆盖
- 负增益 -0.005

**本方案为何不适用此失败模式**:
1. **不条件化时间步**: 本方案的 $t$ 采样是无条件的 $U[0, t_{\text{pos}}] \cup U[t_{\text{neg}}, t_{\text{neg\_max}}]$ ($t_{\text{neg\_max}} < 1$, 见 §3.3.1 M3), 不依赖框尺度。
2. **框尺度通过 RF-DETR 现有机制处理**: RF-DETR 的归一化 cxcywh 坐标已隐式处理尺度, 无需额外条件化。

### 4.5 与端到端可微 Cascade (方向 N, mAP=0.684) 的区别

**端到端可微 Cascade 失败原因** (见 [端到端可微Cascade.md](../archived/端到端可微Cascade.md)):
- 去 detach 后 6 stage 链式梯度导致训练严重不稳定, mAP 震荡 0.35~0.68
- 级联结构对梯度截断有强依赖

**本方案为何不适用此失败模式**:
1. **无级联结构**: 本方案基于 RF-DETR (单 decoder), 不存在 6 stage 链式梯度。
2. **去噪 query 不参与级联**: 去噪 query 在 decoder 内与匹配 query 并行处理, 不形成链式依赖。

### 4.6 与 LDMDet 主路线 (a3_full_sota, mAP=0.858) 的区别

**LDMDet 主路线**: RF + Heun + AdaLN-Zero + StochasticOT, 推理时 4 步 Heun 采样 (8 NFE), 主任务是扩散去噪。

**本方案**: RF-DETR baseline + RF-CDN 辅助去噪, 推理时单次 decoder 前向 (1 NFE), 主任务是匹配检测。

**核心区别**: LDMDet 主路线的 RF 是**主任务** (训练 + 推理都用 RF 采样), 本方案的 RF 是**训练时辅助** (只在去噪 query 生成时用 RF 加噪, 推理时无 RF)。这意味着:
- LDMDet 主路线的推理延迟 = 8 NFE + box_renewal
- 本方案推理延迟 = 1 NFE (RF-DETR baseline)

**为何本方案可能超越 LDMDet 主路线**: 本方案借助 RF-DETR 的 DINOv2 骨干 (远强于 LDMDet 的 ResNet50/Swin) 和 NAS 优化检测头, 同时通过 RF-CDN 获得训练时去噪增益。骨干网络的代际差距可能弥补 "辅助任务 vs 主任务" 的差距。

### 4.7 失败原因不适用性总结

| 已证伪方向 | 失败根因 | 本方案是否适用 | 理由 |
|-----------|---------|--------------|------|
| CFM | 速度预测 $\text{Var}(\epsilon)$ 噪声 | ❌ 不适用 | 预测 $x_0$ 不预测 $v$ |
| PD-RF | 推理蒸馏目标偏移 | ❌ 不适用 | 推理无扩散, 无蒸馏 |
| SC-RF | 自条件化训练-推理不一致 | ❌ 不适用 | 无自条件化 |
| ScaleConditionedRF | 尺度条件化失效 | ❌ 不适用 | 不条件化 $t$ |
| 端到端可微 Cascade | 链式梯度不稳定 | ❌ 不适用 | 无级联结构 |

---

## 5. 架构设计

### 5.1 RF-DETR Baseline 架构 (保持不变)

本方案**完全保留** RF-DETR baseline 的架构:

```
Image → DINOv2 Backbone → AIFI (Attention-in-Attention) → CCFM (Cross-scale Feature-fusion Module)
                                          ↓
                            Decoder (N layers, 含 CDN 去噪 query)
                                          ↓
                            分类头 + 回归头 → Predictions
```

- **Backbone**: DINOv2 ViT-L (RF-DETR-L) 或 ViT-B (RF-DETR-S), 冻结或可训练
- **Encoder**: AIFI (单尺度 self-attention)
- **Neck**: CCFM (跨尺度特征融合)
- **Decoder**: RF-DETR 的 decoder, 含 CDN query 生成器 (本方案替换其噪声逻辑)
- **Query Selection**: 不确定性最小化 query selection (RT-DETR 风格)
- **Head**: 分类 + 框回归, 概率损失 (RF-DETR 用 GIN 损失)

**关键约束**: 本方案不修改 backbone / encoder / neck / decoder 结构, 只修改 **训练时的 query 生成逻辑** (CDN → RF-CDN)。

### 5.2 新增模块: RFContinuousDenoisingGroup

本方案新增一个 `RFContinuousDenoisingGroup` 模块, 替换 RF-DETR 现有的去噪 query 生成模块 (具体类名须在阶段 0 从 `rf-detr` 源码核实)。**接口说明 (M6 修订)**: 修订前文档声称 "接口与 mmdet 的 `CdnQueryGenerator` 完全一致", 此声明不严谨 — RF-DETR 不依赖 mmdet, 其去噪模块的实际接口 (输入是 `batch_data_samples` 还是 `(gt_labels, gt_bboxes)`, 返回值结构) 须在阶段 0 核实后对齐。下方代码以 mmdet `CdnQueryGenerator` 的标准接口 (`__call__(batch_data_samples)` 返回 `dn_label_query, dn_bbox_query, attn_mask, dn_meta`) 为**参考模板**, 阶段 0 后按 RF-DETR 实际接口调整。

> **修订说明 (M7)**: 修订前文档将 mmdet 中的类名误写为 `CDNQueryGenerator` (大写 D)。mmdet 实际类名为 `CdnQueryGenerator` (小写 d, 见 mmdet 源码)。已全文更正。但须强调: RF-DETR **不使用** mmdet 的 `CdnQueryGenerator`, 此处引用仅作算法对照。

```python
class RFContinuousDenoisingGroup:
    """RF-CDN: 用 RF 连续时间路径替换 CDN 的固定 lambda 噪声 (logit 空间加噪)。

    Args:
        num_classes: 类别数 (24obj = 24)
        embed_dims: query 嵌入维度 (与 RF-DETR decoder 一致, 默认 256)
        num_dn_queries: 去噪 query 总数上限 (默认 100)
        t_pos: 正样本时间上界, t ~ U[0, t_pos] (默认 0.3)
        t_neg: 负样本时间下界 (默认 0.7)
        t_neg_max: 负样本时间上界, t ~ U[t_neg, t_neg_max] (默认 0.9, < 1 避免梯度趋零)
        label_noise_scale: label 翻转噪声比例 (沿用 CDN, 默认 0.5)
        num_groups: 去噪 group 数 (沿用 CDN, 默认 5)
        noise_std: RF 噪声标准差 (默认 1.0, 在 logit 空间)
    """

    def __init__(
        self,
        num_classes: int,
        embed_dims: int = 256,
        num_dn_queries: int = 100,
        t_pos: float = 0.3,
        t_neg: float = 0.7,
        t_neg_max: float = 0.9,
        label_noise_scale: float = 0.5,
        num_groups: int = 5,
        noise_std: float = 1.0,
    ):
        self.t_pos = t_pos
        self.t_neg = t_neg
        self.t_neg_max = t_neg_max
        self.noise_std = noise_std
        # 其余参数沿用 RF-DETR 现有去噪模块 (阶段 0 核实具体类名)
        self.label_embedding = nn.Embedding(num_classes, embed_dims)
        # ...

    def __call__(self, batch_data_samples):
        # 1. 归一化 GT 框到 cxcywh, 范围 [0, 1]
        gt_labels, gt_bboxes = self._collate_gt(batch_data_samples)
        max_num_target = max(len(b) for b in gt_bboxes)
        num_groups = self._get_num_groups(max_num_target)

        # 2. RF 连续时间采样 (核心改动)
        # 正样本: t ~ U[0, t_pos]
        t_pos_samples = torch.rand(
            (len(gt_labels), num_groups), device=device
        ) * self.t_pos
        # 负样本: t ~ U[t_neg, t_neg_max]  (上限 < 1, M3 修复)
        t_neg_samples = (
            torch.rand((len(gt_labels), num_groups), device=device)
            * (self.t_neg_max - self.t_neg) + self.t_neg
        )

        # 3. RF 前向加噪 (logit 空间, S2 修复): 保证 w, h > 0
        #    z_t = (1-t) * logit(x_0) + t * epsilon;  x_t = sigmoid(z_t)
        z0 = torch.logit(gt_bboxes.clamp(1e-6, 1 - 1e-6))  # logit 映射到无界空间
        epsilon_pos = torch.randn_like(z0) * self.noise_std
        epsilon_neg = torch.randn_like(z0) * self.noise_std
        # 正样本去噪框
        z_t_pos = (1 - t_pos_samples).unsqueeze(-1) * z0 \
                  + t_pos_samples.unsqueeze(-1) * epsilon_pos
        x_t_pos = torch.sigmoid(z_t_pos)  # 回到 [0, 1], 保证 w, h > 0
        # 负样本去噪框
        z_t_neg = (1 - t_neg_samples).unsqueeze(-1) * z0 \
                  + t_neg_samples.unsqueeze(-1) * epsilon_neg
        x_t_neg = torch.sigmoid(z_t_neg)

        # 4. Label noise (沿用 CDN)
        dn_label_query = self._generate_dn_label_query(
            gt_labels, num_groups, self.label_noise_scale
        )

        # 5. Bbox query embedding (inverse_sigmoid, 沿用 RF-DETR 现有机制)
        #    注意: x_t 已在 [0,1], inverse_sigmoid(x_t) = logit(x_t) = z_t,
        #    即嵌入空间与加噪空间自然对齐 (logit 空间)
        dn_bbox_query = self._embed_bbox_query(
            x_t_pos, x_t_neg  # [pos; neg] 拼接
        )

        # 6. Attention mask (沿用 RF-DETR 现有机制)
        attn_mask = self._generate_dn_mask(max_num_target, num_groups)

        dn_meta = {
            'num_denoising_queries': max_num_target * 2 * num_groups,
            'num_denoising_groups': num_groups,
            't_pos_samples': t_pos_samples,  # 供 loss 加权用
            't_neg_samples': t_neg_samples,
        }
        return dn_label_query, dn_bbox_query, attn_mask, dn_meta
```

### 5.3 噪声采样器 (logit 空间, S2 修复)

噪声采样器复用 LDMDet 的 `RectifiedFlow.q_sample` 线性插值公式 (见 [rectified_flow.py](../../ldmdet/diffusion/rectified_flow.py) 的 `q_sample`), 但**在 logit 空间加噪**, 外加 logit/sigmoid 包装:

```python
from ldmdet.diffusion.rectified_flow import RectifiedFlow

rf = RectifiedFlow(snr_scale=1.0)  # snr_scale=1.0, 不重映射 (logit 映射替代)

# 1. logit 映射: 将 [0,1] GT 框映射到无界空间
z0 = torch.logit(gt_bboxes_cxcywh.clamp(1e-6, 1 - 1e-6))  # [N, 4]

# 2. RF 线性插值 (在 logit 空间): z_t = (1-t) z_0 + t * epsilon
z_t_pos, _ = rf.q_sample(
    x_start=z0,  # logit 空间的 GT
    x_noise=torch.randn_like(z0) * noise_std,
    t=t_pos_samples,  # [N] 来自 U[0, t_pos]
)
z_t_neg, _ = rf.q_sample(
    x_start=z0,
    x_noise=torch.randn_like(z0) * noise_std,
    t=t_neg_samples,  # [N] 来自 U[t_neg, t_neg_max]  (M3: 上限 < 1)
)

# 3. sigmoid 回 [0, 1], 保证 w, h > 0
x_t_pos = torch.sigmoid(z_t_pos)
x_t_neg = torch.sigmoid(z_t_neg)
```

**关键区别于 LDMDet 主路线**: LDMDet 主路线用 `snr_scale=2.0` 重映射扩大动态范围; 本方案用 logit 映射替代, 且 sigmoid 回 $[0,1]$ 保证框合法。LDMDet 的 `q_sample` 仅提供线性插值 $(1-t)z_0 + t\epsilon$, logit/sigmoid 包装是本方案新增。

### 5.4 去噪 Query 生成 (从加噪 GT 框 + GT label)

去噪 query 由两部分组成 (沿用 DINO CDN 的双 group 设计):

1. **Position query (4D anchor)**: 加噪后的 GT 框 $x_t$ (cxcywh 归一化), 经 `inverse_sigmoid` 嵌入 (与 DINO CDN 一致; RF-DETR 实际嵌入方式须阶段 0 核实)
2. **Content query (label embedding)**: 翻转后的 GT label 经 `label_embedding` 查表 (沿用 CDN 标准设计)

正样本 group 和负样本 group 拼接成 `[num_denoising_queries, ...]`, 其中前一半是正样本 (重建 GT), 后一半是负样本 (预测背景)。

### 5.5 Attention Mask 设计 (去噪 query ↔ matching query 隔离)

完全沿用 DINO CDN 的 `generate_dn_mask` 标准设计 (RF-DETR 实际实现须阶段 0 核实), 不做修改:

```
                  ┌─────────────────────────────────┐
                  │  DN Group 1  DN Group 2  Match  │
                  ├─────────────────────────────────┤
  DN Group 1      │   Visible    Masked    Masked   │
  DN Group 2      │   Masked     Visible   Masked   │
  Match           │   Visible    Visible   Visible  │
                  └─────────────────────────────────┘
```

- 去噪 group 间互相 masked (隔离)
- 去噪 query 对 matching query masked (防止去噪"偷看"匹配)
- matching query 对去噪 query visible (允许匹配从去噪锚点获益)

### 5.6 损失函数: $L_{det}$ (matching) + $\lambda \cdot L_{denoise}$ (x0 回归 + cls)

总损失:

$$
L_{total} = L_{det}(\text{matching queries}) + \lambda \cdot L_{denoise}(\text{denoising queries})
$$

**$L_{det}$ (主任务, 沿用 RF-DETR)**:
- 分类: 概率损失 (RF-DETR 用 GIN loss) 或 focal loss
- 框回归: L1 + GIoU
- Hungarian matching (一对一)

**$L_{denoise}$ (辅助任务, 本方案)**:

$$
L_{denoise} = L_{denoise}^{pos} + L_{denoise}^{neg}
$$

正样本 (重建 GT):
$$
L_{denoise}^{pos} = \sum_{i \in \text{pos}} \left[ \alpha_{cls} \cdot L_{cls}(\hat{c}_i, c_i) + \alpha_{L1} \cdot \|\hat{b}_i - b_i\|_1 + \alpha_{giou} \cdot (1 - \text{GIoU}(\hat{b}_i, b_i)) \right]
$$

负样本 (预测背景):
$$
L_{denoise}^{neg} = \sum_{j \in \text{neg}} \alpha_{cls} \cdot L_{cls}(\hat{c}_j, \varnothing)
$$

其中 $\hat{c}, \hat{b}$ 是去噪 query 的预测, $c, b$ 是 GT label/box, $\varnothing$ 是背景类。

**关键设计**: 正样本的框回归目标直接是 GT ($x_0$), **不经过 Hungarian matching**, 这正是 DN-DETR 的核心优势 — 提供稳定的非匹配监督。

**$\lambda$ 的选择**: 沿用 DN-DETR / DINO 默认 $\lambda = 1.0$ (去噪 loss 与主 loss 等权)。可在消融实验中探索 $\lambda \in \{0.5, 1.0, 2.0\}$。

**$t$ 加权 (可选)**: 可对正样本 loss 按 $t$ 加权, 让小 $t$ (易) 权重低, 大 $t$ (难) 权重高:

$$
w(t) = \frac{t}{t_{\text{pos}}}, \quad L_{denoise}^{pos} = \sum_i w(t_i) \cdot [\dots]
$$

但这会增加超参数, 建议阶段 3 先不加权, 阶段 4 消融。

### 5.7 训练 vs 推理流程对比

**训练流程** (本方案改动):

```
1. Image → Backbone → Encoder → 多尺度特征
2. Query Selection → 匹配 query (N=300)
3. RF-CDN Query Generator:                # ← 本方案改动点
   a. GT 框 + label
   b. t_pos ~ U[0, 0.3], t_neg ~ U[0.7, 0.9]  (t_neg_max=0.9, 见 §3.3.1 M3)
   c. z_t = (1-t)·logit(GT) + t·ε, x_t = sigmoid(z_t)  (logit 空间 RF 加噪, 见 §3.3.1 S2)
   d. 生成去噪 query (正 + 负) + attention mask
4. Decoder forward: 匹配 query + 去噪 query (共享 decoder, mask 隔离)
5. Loss = L_det(matching) + λ * L_denoise(denoising)
6. Backprop
```

**推理流程** (与 RF-DETR baseline 完全一致):

```
1. Image → Backbone → Encoder → 多尺度特征
2. Query Selection → 匹配 query (N=300)
3. RF-CDN Query Generator: 跳过 (推理时不生成去噪 query)
4. Decoder forward: 仅匹配 query
5. Output → NMS-free predictions
```

**关键性质**: 推理时模型参数与训练时相同 (decoder 权重共享), 但输入只有匹配 query, **零推理开销**。这与 DN-DETR / DINO / RF-DETR 的设计一致。

---

## 6. 实验设计

### 6.1 阶段 1: RF-DETR Baseline 在 24obj 数据集 fine-tune

**目的**: 建立 RF-DETR 在 24obj 数据集的基线 mAP, 作为本方案的对照。

**配置**:
- 模型: RF-DETR-S (或 RF-DETR-L, 视资源)
- 预训练: COCO 预训练权重
- Fine-tune: 24obj 数据集, 100 epoch, batch=8, AdamW, lr=1e-4
- 去噪: 启用 RF-DETR 默认去噪机制 (具体形式 + 超参数由阶段 0 核实, **不可先验假设 $\lambda_1=0.4$**)

**预期 mAP**: 0.85~0.87 (RF-DETR 在 COCO 上 SOTA, 24obj 是染色体子集, 应接近 LDMDet a3=0.858)

**消融变量**: 无 (建立基线)

### 6.2 阶段 2: 加入 RF-CDN (连续噪声, $\hat{x}_0$ 预测)

**目的**: 验证本方案核心 — RF 连续噪声去噪。

**配置**:
- 在阶段 1 基线上, 替换 RF-DETR 的去噪机制为 RF-CDN
- $t_{\text{pos}} = 0.3$, $t_{\text{neg}} = 0.7$, $t_{\text{neg\_max}} = 0.9$, noise_std=1.0, num_dn_queries=100, num_groups=5
- **logit 空间加噪** (见 §3.3.1 S2)
- 其余同阶段 1

**预期 mAP**: 与阶段 1 持平 ~ +0.005 (保守估计, 见 §7.1)。连续噪声谱的边际增益在单 seed 训练波动 (±0.005) 内, 须多 seed 验证 (见 §6.4)。

**消融变量**: RF-DETR 默认去噪 (固定 λ) vs RF-CDN (连续 t)

> **修订说明 (M13)**: 修订前文档预期 "+0.005~0.015", 过于乐观。考虑到 CDN 已是成熟技术, RF-CDN 的边际增益可能 < 0.005 (见 §7.1 风险评估), 统一为保守估计 +0~0.005。

### 6.3 阶段 3: 消融噪声尺度 ($t \sim U[0, 0.3]$ vs $U[0, 1]$ vs LogNormal)

**目的**: 找到最优的 $t$ 采样分布。

**配置** (在阶段 2 基础上):
- 3a: $t_{\text{pos}} = 0.3, t_{\text{neg}} = 0.7, t_{\text{neg\_max}} = 0.9$ (阶段 2 默认)
- 3b: $t_{\text{pos}} = 0.5, t_{\text{neg}} = 0.5$ (退化为单点, 类似 CDN 固定 λ; **此配置替代原阶段 2 的 DN-DETR 对照**, 见 M14)
- 3c: $t \sim U[0, 1]$ 全区间 (无正负区分, 全当正样本)
- 3d: $t \sim \text{LogNormal}(-1, 1)$ 截断到 [0, 1] (偏向小 t, 模仿 LDMDet 主路线的 t 分布偏好)
- 3e: $t_{\text{pos}} = 0.2, t_{\text{neg}} = 0.8, t_{\text{neg\_max}} = 0.95$ (更窄的正负区间, 更接近 CDN)

> **修订说明 (M14)**: 修订前文档有独立 "阶段 2: DN-DETR 对照" (关闭 RF-DETR CDN 负样本)。此操作需 fork RF-DETR 源码修改去噪逻辑, 非轻量操作, 且其对照价值可通过阶段 3b ($t_{\text{pos}}=t_{\text{neg}}=0.5$ 退化为单点) 等价实现。已删除独立阶段 2, 其消融变量合并到阶段 3b。

**预期 mAP** (保守, 仅作趋势参考, 实际以多 seed 均值为准):
- 3a (默认): baseline + 0~0.005
- 3b (单点): 接近阶段 1 (退化, 类似 CDN)
- 3c (全正): 略低于 3a (无负样本对比)
- 3d (LogNormal): 可能略优 (偏向易样本)
- 3e (窄区间): 接近 3a

**消融变量**: $t$ 采样分布

### 6.4 阶段 4: 消融去噪 query 数量 (100/300/500) + 多 seed 验证

**目的**: 找到最优的去噪 query 数量, 平衡增益与训练开销; 通过多 seed 训练验证 +0.005 量级增益的统计显著性。

**配置** (在阶段 2 基础上):
- 4a: num_dn_queries = 100 (默认)
- 4b: num_dn_queries = 300
- 4c: num_dn_queries = 500
- 4d: num_dn_queries = 0 (关闭去噪, = 无去噪版本)

**多 seed 计划 (M15 修订)**:
- 阶段 1 (baseline) 和阶段 2 (RF-CDN 默认) 各跑 **≥3 seeds** (seed=42, 123, 2024)
- 报告 mean ± std, 用 paired t-test 检验阶段 2 vs 阶段 1 的显著性
- 若 +0.005 增益在 3 seeds 的 std (典型 ±0.005) 内不可区分, 则承认 RF-CDN 增益不显著, 回退到 RF-DETR baseline
- 阶段 3-4 的消融配置可单 seed (资源约束), 但若发现 < 0.003 差异则需补 seed

> **修订说明 (M15)**: 修订前文档未纳入多 seed 训练。+0.005 增益在单 seed 训练波动 (±0.005) 中不可信, 必须多 seed 验证。

**预期 mAP** (保守):
- 4a (100): baseline + 0~0.005
- 4b (300): 可能略优 (更多去噪信号)
- 4c (500): 可能饱和或损害主任务
- 4d (0): 低于阶段 1 (无去噪)

**消融变量**: num_dn_queries

### 6.5 实验汇总

| 阶段 | 配置 | 预期 mAP (vs 阶段 1) | 消融变量 | seeds |
|------|------|---------|---------|-------|
| 1 | RF-DETR baseline | 基线 | 基线 | ≥3 |
| 2 | RF-CDN (连续 t, logit 空间) | +0~0.005 | RF-DETR 去噪 vs RF-CDN | ≥3 |
| 3a-3e | RF-CDN 噪声分布 | -0.005~+0.005 | $t$ 分布 | 1 (差异<0.003 补 seed) |
| 4a-4d | RF-CDN query 数量 | -0.01~+0.005 | num_dn_queries | 1 (差异<0.003 补 seed) |

**成功判据** (见第 8 节):
- 最低: 阶段 2 mAP (3-seed mean) ≥ 阶段 1 mAP (3-seed mean) (RF-CDN 不损基线)
- 理想: 阶段 2 vs 阶段 1 的 3-seed paired t-test p < 0.05 且均值提升 ≥ 0.003 (统计显著增益)

---

## 7. 风险评估

### 7.1 RF-DETR 已有 CDN, 新增 RF-CDN 是否冲突或冗余

**风险**: RF-DETR 已内置 CDN (固定 $\lambda$), 本方案替换为 RF-CDN (连续 $t$)。如果 RF-CDN 的连续噪声不能比固定 $\lambda$ 提供更多增益, 则改动是冗余的。

**评估**:
- **替换关系 (非共存)**: 本方案明确**替换** RF-DETR 的 CDN, 不是叠加。去噪 group 数量、attention mask 机制、正负样本语义都保留, 只改噪声分布。因此不会因 "双重去噪" 冲突。
- **冗余风险**: 如果 24obj 数据集的 GT 框分布对噪声尺度不敏感 (例如框大小均匀), 则连续 $t$ 相比固定 $\lambda$ 的增益可能 < 0.005, 落入噪声范围。
- **缓解**: 阶段 3 的消融实验 (3a vs 3b) 直接对比连续 vs 单点, 如果差异 < 0.002 则承认冗余, 回退到 RF-DETR 默认去噪。

**诚实评估**: 这是本方案最大的风险。CDN 已是成熟技术, RF-CDN 的边际增益可能有限。但考虑到 LDMDet 项目在 4 维框扩散上验证了 RF 连续 $t$ 的有效性 (a3=0.858 > DDPM 基线), 将其引入去噪训练有理论支撑。

### 7.2 连续噪声尺度在小数据集 (24obj) 上的过拟合风险

**风险**: 24obj 数据集规模小 (约几千张图), 连续 $t$ 增加了训练时的 "噪声维度", 模型可能过拟合到特定的 $t$ 分布。

**评估**:
- **去噪是辅助任务**: 去噪 loss 权重 $\lambda = 1.0$, 但主任务 (matching) 仍是主要监督源。去噪 query 在推理时移除, 不会直接过拟合到测试时表现。
- **CDN 经验**: DINO 在 COCO (118k 图) 上 CDN 有效, 但在小数据集 (如 VisDrone 10k 图) 上 CDN 仍有效 (DN-DETR 论文 Table 5 验证)。24obj 规模介于两者之间, 风险中等。
- **缓解**: 阶段 3 的 LogNormal 分布 (3d) 偏向小 $t$ (易样本), 减少大 $t$ (难样本) 的过拟合风险。也可引入 $t$ 的 dropout (随机跳过去噪 loss)。

### 7.3 去噪 query 占用 decoder 容量, 可能损害主任务

**风险**: 去噪 query (100~500) 与匹配 query (300) 共享 decoder, 去噪 query 占用 decoder 的 attention 容量, 可能损害匹配 query 的表现。

**评估**:
- **Attention mask 隔离**: 去噪 query 对匹配 query masked, 但匹配 query 对去噪 query visible。这意味着匹配 query 会 "看到" 去噪 query 的信息, 如果去噪 query 噪声过大, 可能干扰匹配。
- **CDN 经验**: DINO 在 num_dn_queries=100 时不损害主任务, 但 300+ 时可能有问题 (DINO 论文未消融此变量)。
- **缓解**: 阶段 4 的消融 (4a-4d) 直接测试 query 数量的影响。如果 4c (500) 显著低于 4a (100), 则确认容量风险, 回退到 100。

### 7.4 与 RF-DETR NAS 搜索空间的交互

**风险**: RF-DETR 的 NAS 在 COCO 上搜索了 decoder 层数、query 数量等超参数。本方案修改了去噪逻辑, 可能与 NAS 搜索到的最优配置不兼容。

**评估**:
- **NAS 不搜索去噪**: RF-DETR 的 NAS 搜索空间 (论文 Table 2) 包括 image resolution, patch size, decoder layers, query tokens, 但**不包括 CDN 的噪声尺度**。因此本方案的 $t$ 修改不直接与 NAS 冲突。
- **间接交互**: NAS 搜索到的 query tokens (默认 300) 与去噪 query (100) 共用 decoder, 如果去噪 query 改变 decoder 的最优 query tokens, NAS 结果可能次优。
- **缓解**: 阶段 4 消融 query 数量时, 同时测试匹配 query 数量 (300 vs 500), 验证是否需要重新 NAS。

### 7.5 RF-DETR baseline 可能已超过 LDMDet a3

**风险**: 如果阶段 1 的 RF-DETR baseline 已超过 LDMDet a3 (0.858), 则本方案的 "超越 a3 + 0.005" 目标可能被 baseline 自身达成, 无法归因于 RF-CDN。

**评估**:
- **可能性高**: RF-DETR 在 COCO 上 56.5 AP (L) 远超 LDMDet 的 ResNet50 基线, 在 24obj 上也可能已超 0.858。
- **缓解**: 成功判据明确区分 "≥ baseline" (最低) 和 "统计显著提升" (理想)。即使 baseline 已超 a3, 只要 RF-CDN 相比 baseline 仍有 3-seed 显著增益, 即算成功。阶段 2 vs 阶段 1 的对比是核心判据。

### 7.6 梯度冲突风险 (高风险, 引用 LDMDet 教训)

**风险**: RF-CDN 引入去噪 loss $L_{denoise}$ 与主任务 matching loss $L_{det}$ 在 decoder 共享参数上**梯度耦合**, 可能产生梯度冲突, 导致训练不稳定或性能退化。

**LDMDet 项目教训 (5 个已证伪方向中 3 个与梯度冲突相关)**:
- **Consistency Loss (方向已证伪)**: 在共享 Transformer 层上附加一致性 loss, mAP 从 0.739 退化至 0.721 (-0.018)。根因: 附加 loss 的梯度与主任务梯度在共享层冲突。测量得梯度余弦 $\cos = -0.104$, **86.8% 的共享层存在梯度冲突**。
- **PCGrad (部分缓解, 仍证伪)**: 用梯度投影消除冲突分量, mAP=0.724, 仍低于 baseline, 说明 PCGrad 只能部分缓解, 无法根治。
- **CFM 速度预测 (方向 H, mAP=0.823)**: 速度 loss $L_v$ 收敛到 $\text{Var}(\epsilon)=4$, 注入梯度噪声, 本质也是梯度耦合问题。
- **结论**: LDMDet 项目的核心教训是 — **任何在共享 decoder 参数上附加辅助 loss 的方案, 都必须显式评估梯度冲突**, 不能假设辅助 loss 一定与主任务协同。

**本方案的梯度冲突分析**:
- **共享参数**: RF-CDN 的去噪 query 与 matching query 共享同一个 decoder (自注意力 + 交叉注意力 + FFN)。$L_{denoise}$ 和 $L_{det}$ 的梯度都流经这些共享参数。
- **冲突点**: $L_{det}$ 通过 Hungarian matching 监督 matching query, 梯度方向是 "让 query 更好地匹配检测目标"; $L_{denoise}$ 直接监督去噪 query 重建 GT, 梯度方向是 "让 query 更好地回归已知框"。两者在 decoder 的注意力权重和 FFN 参数上**可能方向不一致** (尤其当去噪 query 的噪声尺度 $t$ 较大时, 去噪梯度的方向偏离检测梯度)。
- **与 LDMDet Consistency Loss 的相似性**: 二者都是 "在共享层附加辅助回归 loss"。LDMDet 实测 86.8% 层冲突, 本方案在 RF-DETR decoder 上的冲突比例未知, 但风险不可忽略。

**缓解方案**:
1. **梯度余弦监控 (必做)**: 训练前 10 个 epoch, 每 epoch 测量 $\cos(\nabla_\theta L_{det}, \nabla_\theta L_{denoise})$ 在 decoder 各层 (self-attn, cross-attn, FFN) 的值。若平均 $\cos < 0$ 或冲突层比例 > 50%, 触发降权。
2. **冲突时降权**: 若监控发现冲突, 将 $L_{denoise}$ 权重 $\lambda$ 从 1.0 降至 0.3~0.5, 或对去噪 loss 梯度做 PCGrad 投影 (参考 LDMDet 的 `measure_gradient_conflict.py` 工具)。
3. **极端回退**: 若降权后仍冲突 (mAP 下降 > 0.005), 关闭负样本去噪 (只保留正样本重建, 梯度方向更接近主任务), 或完全回退到 RF-DETR baseline。
4. **不建议的策略**: 避免直接 freeze 共享层 (LDMDet 实测 freeze 可稳定 mAP 但无法提升, 因为去噪 loss 无法影响表示学习)。

> **修订说明 (M4, 高风险项)**: 修订前文档完全未讨论梯度冲突风险。LDMDet 项目 5 个已证伪方向中 3 个 (CFM / PD-RF / Consistency Loss) 与梯度耦合相关, 是项目最重要的教训之一。本方案虽不在级联架构下, 但 "共享 decoder + 辅助去噪 loss" 的结构与 Consistency Loss 失败模式高度相似, 必须显式监控。

### 7.7 风险汇总

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| **梯度冲突 (M4)** | **中** | **高** | **梯度余弦监控 + 冲突时降权 $\lambda$** |
| RF-CDN 与 CDN 冗余 | 中 | 中 | 阶段 3a vs 3b 消融 |
| 小数据集过拟合 | 中 | 中 | LogNormal 分布 + dropout |
| Decoder 容量损害主任务 | 低 | 高 | 阶段 4 query 数量消融 |
| NAS 交互次优 | 低 | 中 | 重新 NAS 匹配 query 数量 |
| Baseline 已超 a3 | 高 | 低 | 用阶段 2 vs 1 对比作判据 |

---

## 8. 成功判据

### 8.1 最低目标 (必须达成)

- **mAP ≥ RF-DETR baseline (阶段 1, 3-seed mean)**: 本方案至少不能损害 RF-DETR 的基线性能。如果阶段 2 mAP (3-seed mean) < 阶段 1 mAP (3-seed mean), 则 RF-CDN 是负增益, 方案证伪。
- **推理延迟 = RF-DETR baseline**: 推理时去噪 query 移除, 延迟必须等同 baseline (零开销)。这是硬约束, 不达成则方案无意义。

### 8.2 理想目标 (期望达成)

- **阶段 2 vs 阶段 1 的 3-seed mean mAP 提升 ≥ 0.003 且 paired t-test p < 0.05**: 归因于 RF-CDN (而非 RF-DETR baseline 自身或训练波动), 验证连续噪声的统计显著增益。
- **若 RF-DETR baseline 已超 LDMDet a3 (0.858)**: 则 "路径 2 优于路径 1" 的判据转为 "阶段 2 > 阶段 1 (显著) + 推理零开销", 不再硬性要求绝对值 ≥ 0.863。

> **修订说明 (M13)**: 修订前文档理想目标为 "mAP ≥ 0.863 (+0.005)"。考虑到 RF-CDN 增益保守估计为 +0~0.005 (见 §6.2), 且单 seed 波动 ±0.005, 将绝对值判据改为 **3-seed 统计显著性判据** (提升 ≥ 0.003 + p < 0.05), 更稳健。

### 8.3 速度约束 (硬约束)

- **推理延迟 = RF-DETR baseline**: 在 NVIDIA T4 GPU 上测量, 延迟差异 < 1ms (考虑测量噪声)。
- **推理 NFE = 1**: 单次 decoder 前向, 无多步采样。

### 8.4 失败判据 (方案证伪)

满足以下任一条件, 方案证伪:
- 阶段 2 mAP (3-seed mean) < 阶段 1 mAP (3-seed mean) - 0.002 (RF-CDN 显著损害基线)
- 推理延迟 > RF-DETR baseline + 2ms (去噪 query 未正确移除)
- 阶段 3 所有配置 mAP < 阶段 1 (RF-CDN 在任何噪声分布下都无效)
- 梯度冲突监控 (§7.6) 显示 > 50% 层冲突且降权后仍无法缓解 (mAP 持续下降)

---

## 9. 实施计划

### 9.1 代码改动范围估计 (仅训练时, 不改推理)

**改动文件** (基于 RF-DETR 开源代码, 路径假设为 `rf-detr/`):

1. **新增**: `rf_detr/denoising/rf_cdn.py` — `RFContinuousDenoisingGroup` 模块 (~200 行)
2. **修改**: `rf_detr/denoising/__init__.py` — 导出新模块 (~5 行)
3. **修改**: `rf_detr/detector.py` — 训练时 query 生成调用 RF-CDN (~10 行)
4. **修改**: `rf_detr/loss.py` — 去噪 loss 计算适配 RF-CDN 的 $t$ 信息 (~30 行)
5. **配置**: `experiments/configs/rf_detr/rf_cdn_24obj.py` — 实验配置 (~50 行)

**总改动**: ~300 行新增 + ~50 行修改, 集中在训练时 query 生成和 loss 计算。**推理代码完全不动**。

### 9.2 依赖项

- **RF-DETR**: `pip install rfdetr>=1.5.0` (Apache 2.0 协议, N/S/M/L 可商用)
- **LDMDet RF 工具**: 复用 [ldmdet/diffusion/rectified_flow.py](../../ldmdet/diffusion/rectified_flow.py) 的 `RectifiedFlow.q_sample` (可直接 import 或复制 ~60 行)
- **PyTorch**: ≥ 2.0 (RF-DETR 要求)
- **数据集**: 24obj 染色体数据集 (项目已有)

### 9.3 计算资源需求

- **训练**: **1× A100 80G** (推荐, RF-DETR-L 含 DINOv2 ViT-L 骨干需较大显存; RF-DETR-S 可降至 40G)
- **16GB GPU 可行性评估**: RF-DETR-S (ViT-B 骨干, 32M 参数) 理论上可在 16GB GPU 上以 batch=2 + gradient accumulation 运行, 但 DINOv2 ViT-B 微调时激活值较大, 实测可能 OOM。建议先用 Colab/单卡 16G 做 stage 0 核实 + 小规模试跑, 正式实验用 A100 80G。
- **训练时长**: ~12 小时 / 100 epoch (RF-DETR-S, 24obj); RF-DETR-L ~20 小时
- **推理评估**: < 10 分钟 / epoch (24obj val)
- **总实验算力 (M10 修订)**:
  - 阶段 1 (baseline, 3 seeds): 3 × 12h = 36h
  - 阶段 2 (RF-CDN, 3 seeds): 3 × 12h = 36h
  - 阶段 3 (5 噪声分布配置, 单 seed): 5 × 12h = 60h
  - 阶段 4 (4 query 数量配置, 单 seed): 4 × 12h = 48h
  - 阶段 0 (源码核实 + 试跑): ~20h
  - **合计: ~200 GPU 小时**

> **修订说明 (M10)**: 修订前文档估计 "5 周, 120 GPU 小时", 严重低估。增加多 seed 训练 (M15) + 阶段 0 源码核实后, 重新估计为 **9-12 周, ~200 GPU 小时**。

### 9.4 时间线

| 周次 | 任务 | 产出 |
|------|------|------|
| 1-2 | 阶段 0: fork RF-DETR 源码, 核实去噪机制 (§2.1 三种可能性) | 去噪机制核实报告 |
| 3 | RF-DETR baseline 部署 + 24obj 适配 | 阶段 1 环境就绪 |
| 4-5 | RFContinuousDenoisingGroup 实现 + 单测 + 梯度冲突监控工具 | 代码 PR |
| 6-7 | 阶段 1 实验 (baseline, 3 seeds) | 阶段 1 基线 mAP (3-seed) |
| 8-9 | 阶段 2 实验 (RF-CDN, 3 seeds) + 梯度冲突分析 | 阶段 2 mAP (3-seed) + 冲突报告 |
| 10-11 | 阶段 3-4 消融 (噪声分布 + query 数量) | 消融表 |
| 12 | 多 seed 统计检验 + 结果分析 + 文档更新 | 最终报告 |

### 9.5 代码改动示例 (关键部分)

**RF-CDN query 生成** (核心改动, 替换 RF-DETR 的去噪 query 生成模块; **接口须按阶段 0 核实的 RF-DETR 实际签名调整**, 下方 `forward(self, batch_data_samples)` 为参考模板):

```python
# rf_detr/denoising/rf_cdn.py
import torch
import torch.nn as nn

class RFContinuousDenoisingGroup(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embed_dims: int = 256,
        num_dn_queries: int = 100,
        t_pos: float = 0.3,
        t_neg: float = 0.7,
        t_neg_max: float = 0.9,  # M3: 上限 < 1, 避免梯度趋零
        label_noise_scale: float = 0.5,
        num_groups: int = 5,
        noise_std: float = 1.0,
    ):
        super().__init__()
        self.t_pos = t_pos
        self.t_neg = t_neg
        self.t_neg_max = t_neg_max
        self.noise_std = noise_std
        self.label_noise_scale = label_noise_scale
        self.num_groups = num_groups
        self.num_dn_queries = num_dn_queries
        self.label_embedding = nn.Embedding(num_classes, embed_dims)

    def forward(self, batch_data_samples):
        """
        Args (参考 mmdet CdnQueryGenerator 接口, 阶段 0 后按 RF-DETR 实际接口调整):
            batch_data_samples: list of data samples with GT labels/bboxes

        Returns:
            dn_label_query, dn_bbox_query, attn_mask, dn_meta
        """
        # 1. Collate GT (padding to max_num_target)
        gt_labels, gt_bboxes = self._collate_gt(batch_data_samples)
        max_num_target = max(len(b) for b in gt_bboxes)
        bs = len(gt_bboxes)
        num_groups = min(
            self.num_groups,
            max(1, self.num_dn_queries // max(1, max_num_target))
        )

        # 2. RF continuous time sampling
        # Positive: t ~ U[0, t_pos]
        t_pos = torch.rand(bs, max_num_target, num_groups) * self.t_pos
        # Negative: t ~ U[t_neg, t_neg_max]  (M3: 上限 < 1)
        t_neg = torch.rand(bs, max_num_target, num_groups) \
                * (self.t_neg_max - self.t_neg) + self.t_neg

        # 3. RF forward noising in LOGIT space (S2 修复): 保证 w, h > 0
        #    z_t = (1-t) * logit(x_0) + t * epsilon;  x_t = sigmoid(z_t)
        gt_pad = self._pad_gt(gt_bboxes, max_num_target)  # [bs, N, 4] in [0,1]
        z0 = torch.logit(gt_pad.clamp(1e-6, 1 - 1e-6))    # logit 映射到无界空间
        eps_pos = torch.randn_like(z0) * self.noise_std
        eps_neg = torch.randn_like(z0) * self.noise_std

        # Broadcast t: [bs, N, num_groups, 1]
        t_pos_exp = t_pos.unsqueeze(-1)
        t_neg_exp = t_neg.unsqueeze(-1)

        # z_t: [bs, N, num_groups, 4]  (logit 空间线性插值)
        z_t_pos = (1 - t_pos_exp) * z0.unsqueeze(2) + t_pos_exp * eps_pos.unsqueeze(2)
        z_t_neg = (1 - t_neg_exp) * z0.unsqueeze(2) + t_neg_exp * eps_neg.unsqueeze(2)
        # sigmoid 回 [0, 1], 保证 w, h > 0
        x_t_pos = torch.sigmoid(z_t_pos)
        x_t_neg = torch.sigmoid(z_t_neg)

        # 4. Flatten: [bs, N * num_groups * 2, 4]  (pos + neg)
        x_t = torch.cat([x_t_pos, x_t_neg], dim=2)  # [bs, N, 2*num_groups, 4]
        dn_bbox_query = x_t.reshape(bs, max_num_target * 2 * num_groups, 4)

        # 5. Label noise (CDN-style)
        gt_labels_pad = self._pad_labels(gt_labels, max_num_target)  # [bs, N]
        dn_label_query = self._generate_noisy_labels(
            gt_labels_pad, num_groups
        )  # [bs, N*2*num_groups, embed_dims]

        # 6. Attention mask (CDN-style)
        attn_mask = self._generate_attn_mask(
            max_num_target, num_groups, bs
        )

        dn_meta = {
            'num_denoising_queries': max_num_target * 2 * num_groups,
            'num_denoising_groups': num_groups,
            't_pos': t_pos,  # for optional t-weighted loss
            't_neg': t_neg,
        }
        return dn_label_query, dn_bbox_query, attn_mask, dn_meta
```

**Loss 计算适配**:

```python
# rf_detr/loss.py (修改)
def compute_dn_loss(self, dn_outputs, dn_meta, gt_labels, gt_bboxes):
    """Compute RF-CDN denoising loss.

    dn_outputs: dict with 'pred_logits' and 'pred_boxes' for denoising queries
    dn_meta: dict with 'num_denoising_queries', 'num_denoising_groups', 't_pos', 't_neg'
    """
    num_dn = dn_meta['num_denoising_queries']
    num_groups = dn_meta['num_denoising_groups']
    num_pos = num_dn // 2  # 正样本占一半
    num_neg = num_dn // 2  # 负样本占一半

    pred_logits = dn_outputs['pred_logits']  # [bs, num_dn, num_classes]
    pred_boxes = dn_outputs['pred_boxes']    # [bs, num_dn, 4]

    # 正样本: 重建 GT (x0 prediction)
    pred_logits_pos = pred_logits[:, :num_pos, :]
    pred_boxes_pos = pred_boxes[:, :num_pos, :]
    # ... match to GT (no Hungarian, direct supervision)
    loss_pos_cls = self.cls_loss(pred_logits_pos, gt_labels_expanded)
    loss_pos_box = self.l1_loss(pred_boxes_pos, gt_bboxes_expanded) \
                   + (1 - giou(pred_boxes_pos, gt_bboxes_expanded))

    # 负样本: 预测背景
    pred_logits_neg = pred_logits[:, num_pos:, :]
    target_neg = torch.full_like(pred_logits_neg[..., 0], self.bg_label)
    loss_neg_cls = self.cls_loss(pred_logits_neg, target_neg)

    loss_dn = loss_pos_cls + loss_pos_box + loss_neg_cls
    return loss_dn
```

---

## 10. 参考文献

[1] Li, F., Zhang, H., Liu, S., Guo, J., Ni, L.M., & Zhang, L. (2022). DN-DETR: Accelerate DETR Training by Introducing Query DeNoising. *CVPR 2022 (Oral)*. arXiv:2203.01305. https://arxiv.org/abs/2203.01305

[2] Zhang, H., Li, F., Liu, S., Zhang, L., Su, H., Zhu, J., Ni, L.M., & Shum, H.Y. (2023). DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection. *ICLR 2023*. arXiv:2203.03605. https://arxiv.org/abs/2203.03605

[3] Robinson, I., Robicheaux, P., Popov, M., Ramanan, D., & Peri, N. (2026). RF-DETR: Neural Architecture Search for Real-Time Detection Transformers. *ICLR 2026*. arXiv:2511.09554. https://arxiv.org/abs/2511.09554

[4] Liu, X., Gong, C., & Liu, Q. (2023). Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow. *ICLR 2023*. arXiv:2209.03003. https://arxiv.org/abs/2209.03003

[5] Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., & Zagoruyko, S. (2020). End-to-End Object Detection with Transformers. *ECCV 2020*. arXiv:2005.12872.

[6] Zhao, Y., Lv, W., Xu, S., Wei, J., Wang, G., Dang, Q., Liu, Y., & Chen, J. (2024). DETRs Beat YOLOs on Real-time Object Detection. *CVPR 2024*. arXiv:2304.08069.

[7] Zong, Z., Song, G., & Liu, Y. (2023). DETRs with Collaborative Hybrid Assignments Training. *ICCV 2023*. arXiv:2211.12860.

[8] Peng, S., et al. (2024). D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement. *arXiv:2410.13842*.

[9] NAN-DETR: Multi-Anchor based Denoising DETR. *Frontiers in Computer Science*, 2024.

[10] Li, F., et al. (2023). Mask DINO: Towards A Unified Transformer-based Framework for Object Detection and Segmentation. *arXiv:2206.02777*.

[11] Oquab, M., Darcet, T., Moutakanni, T., et al. (2024). DINOv2: Learning Robust Visual Features without Supervision. *TMLR 2024*. arXiv:2304.07193.

[12] DiffusionDet: Chao, J., et al. (2023). DiffusionDet: Diffusion Model for Object Detection. *ICCV 2023*. arXiv:2211.09788.

[13] LDMDet 项目内部实验记录: a3_full_sota (RF + Heun + AdaLN-Zero + StochasticOT ε=5, mAP=0.858, 24obj). 见 [docs/EXPERIMENT_RESULTS.md](../EXPERIMENT_RESULTS.md).

[14] LDMDet 项目内部证伪记录: 方向 H (CFM 速度预测, mAP=0.823), 方向 N (端到端可微 Cascade, mAP=0.684), SC-RF (mAP=0.860), PD-RF (best mAP=0.851). 见 [docs/research/frontier_directions/](.).

---

## 附录 A: 与路径 1 (LDMDet 主路线) 的对比

| 维度 | 路径 1 (LDMDet a3) | 路径 2 (本方案) |
|------|-------------------|----------------|
| 架构 | ResNet50/Swin + FPN + 6 级联 SingleDiffusionDetHead | DINOv2 + AIFI + CCFM + RF-DETR decoder |
| 训练主任务 | 扩散去噪 (RF, $t \sim U[0,1]$) | 匹配检测 (Hungarian) |
| 训练辅助任务 | 无 | RF-CDN 去噪 (RF, logit 空间, $t \sim U[0, t_{pos}] \cup U[t_{neg}, t_{neg\_max}}]$) |
| 推理 | Heun 4 步 (8 NFE) + box_renewal + ensemble | 单次 decoder 前向 (1 NFE) |
| 推理延迟 | 高 (8 NFE) | 低 (1 NFE, 零开销) |
| 已验证 mAP | 0.858 (24obj) | 待验证 (阶段 1-4) |
| 风险 | 低 (已验证) | 中 (RF-CDN 增益不确定) |
| 优势 | 已有完整训练 pipeline | 推理零开销, 骨干更强 |

**结论**: 路径 1 是已验证的稳健路线, 路径 2 是探索性的高潜力路线。两者**不互斥**, 可并行推进。如果路径 2 阶段 2 达成理想目标 (3-seed 显著提升 + 推理零开销, 见 §8.2), 则路径 2 在精度-延迟权衡上优于路径 1。

## 附录 B: RF-CDN 与 LDMDet RF 加噪公式的对比

**LDMDet RF** (见 [rectified_flow.py](../../ldmdet/diffusion/rectified_flow.py) 的 `q_sample`):

```python
t_view = t.view(-1, *([1] * (x_start.dim() - 1)))
x_t = (1.0 - t_view) * x_start + t_view * x_noise  # 直接在 x_start 空间插值
```

**本方案 RF-CDN (logit 空间, S2 修复)**:

```python
z0 = torch.logit(x_0.clamp(1e-6, 1 - 1e-6))      # logit 映射到无界空间
z_t = (1 - t) * z0 + t * epsilon                   # logit 空间线性插值
x_t = torch.sigmoid(z_t)                            # sigmoid 回 [0,1], 保证 w,h > 0
```

两者**线性插值部分等价**: $z_t = (1-t) z_0 + t \epsilon$, 其中 $z_0$ 是数据 (GT 框的 logit), $\epsilon$ 是噪声, $t \in [0, 1]$。**区别在于 $z_0$ 空间 + sigmoid 包装**:
- LDMDet: $x_0 = (\text{norm\_gt\_cxcywh} \times 2 - 1) \times \text{snr\_scale}$ (重映射到 $[-\text{snr\_scale}, \text{snr\_scale}]$), 无 sigmoid 包装
- RF-CDN: $z_0 = \text{logit}(\text{norm\_gt\_cxcywh})$ (映射到 $(-\infty, +\infty)$), **有 sigmoid 包装**回 $[0,1]$

**为何 RF-CDN 用 logit 而非 snr_scale 重映射 (S2 修复)**: RF-DETR 的 decoder 在 $[0, 1]$ 归一化坐标空间操作, 直接在此空间加噪 (原方案) 在 $t > 0.5$ 时产生负 $w, h$。logit 映射 + sigmoid 包装**保证框始终合法** ($w, h > 0$), 且 logit 空间与 RF-DETR 的 `inverse_sigmoid` 框嵌入自然对齐。LDMDet 的 snr_scale 重映射无 sigmoid 约束, 但 LDMDet 在 $[-2, 2]$ 空间多步采样, 负值可接受; RF-CDN 单步输出须回 $[0,1]$。

---

## 附录 C: 阶段 3 LogNormal 分布的数学形式

阶段 3d 的 LogNormal 采样:

$$
t = \min\left(\max\left(\exp(\mathcal{N}(-1, 1)), 0\right), 1\right)
$$

其中 $\mathcal{N}(-1, 1)$ 是均值为 -1、标准差为 1 的高斯分布。该分布的 $t$ 中位数为 $e^{-1} \approx 0.37$, 偏向小 $t$ (易样本), 模仿 LDMDet 主路线在 RF 采样时小 $t$ 更有用的经验。

**实现**:

```python
t_lognormal = torch.randn(num_samples).normal_(-1, 1).exp()
t_lognormal = t_lognormal.clamp(0, 1)
```
