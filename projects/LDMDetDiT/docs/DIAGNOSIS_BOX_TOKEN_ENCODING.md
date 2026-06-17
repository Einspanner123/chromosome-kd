# Box Tokenizer 编码区分度诊断与修复

**日期**: 2025-06-11  
**问题**: mAP=0 的根因诊断 — Box Token 缺乏区分度  
**作者**: AI-assisted diagnosis

---

## 1. 问题现象

训练 8 个 epoch 后 mAP 始终为 0。推理时 NMS 后产生 334-350 个预测框，但所有框分数集中在 [0.10, 0.18]，无任何框 > 0.5。框的空间分布虽均匀，但框尺寸全部约 40x40（初始噪声框默认大小），模型未学会调整。

---

## 2. 诊断假设

最初怀疑 Self-Attention 导致预测框趋同（homogenization）。创建 `diagnose_box_independence.py` 验证。

## 3. 诊断方法

### 3.1 预测框多样性测试

加载 epoch_8 checkpoint，对随机图像（而非训练集图像）做推理，计算：
- 框间 IoU 矩阵
- 中心距离矩阵
- 分数分布

### 3.2 Token 多样性测试

对 100 个均匀分布在 800x800 画面上的初始框，通过 Box Tokenizer 后计算 token 间的余弦相似度矩阵。

---

## 4. 关键发现

### 4.1 Self-Attention 不是主因

框间 IoU = 0.003（几乎不重叠），中心距离 mean = 395（广泛分布）。  
**Self-Attention 并未导致框趋同。**

### 4.2 真正原因：Box Tokenizer 输出同质化

**100 个不同位置的框（从 (20,20) 到 (740,740)）经过 Box Tokenizer 后，token 余弦相似度 = 0.9905（几乎完全相同）。**

这意味着在 Self-Attention 之前，模型就已经无法区分不同框了。

### 4.3 幅度不平衡分析

Token 由四部分相加构成：

```
token = sampled_feat + pos_embed + lvl_embed + content_proj(content)
```

各组件 magnitude 和区分度：

| 组件 | Magnitude (L2 norm) | cos_sim (越低越好) | 问题 |
|------|-------------------|-------------------|------|
| pos_embed | **1.51** | 0.752 (区分度好) | 幅度太小，被淹没 |
| lvl_embed | 20.27 | 1.000 (完全一样) | 100 个框全在同一 FPN 层级 |
| content_proj | 19.61 | 0.984 (几乎一样) | ~40x40 框在随机噪声图上内容相似 |
| **完整 token** | 27.13 | **0.9905** | ← 位置信号仅占 5.6% |

**pos_embed 仅贡献 token 总量的 ~5.6%**，被 lvl_embed 和 content_proj 完全淹没。

### 4.4 MLP 位置编码本身区分度良好

对极端坐标的测试：
- 左上角 vs 右下角 cos_sim = 0.07（非常不同）
- 坐标距离 [0.7, 1.0] 范围 cos_sim = 0.73（合理区分）

### 4.5 正弦编码对比

正弦位置编码作为对照：cos_sim = 0.887，不如 MLP 的 0.752。  
MLP 位置编码在区分度上优于正弦编码，但幅度太小。

---

## 5. 修复方案

### 5.1 核心修复：pos_norm

在 `bbox_pos_embed` 输出后加 `nn.LayerNorm`：

```python
# box_tokenizer.py
self.pos_norm = nn.LayerNorm(feat_channels)

# forward()
pos_embed = self.bbox_pos_embed(bboxes)
pos_embed = self.pos_norm(pos_embed)  # magnitude 1.51 → 19.57
```

### 5.2 效果验证

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| pos_embed magnitude | 1.51 | 19.57 |
| pos_embed cos_sim | 0.752 | 0.752 (不变) |
| 完整 token cos_sim | **0.9905** | **0.8976** |
| 完整 token min cos_sim | 0.9790 | **0.6427** |
| 完整 token std | 0.0020 | **0.0868** |

完整 token 的 cos_sim 从 0.9905 降至 0.8976，最小相似度从 0.979 降至 0.643。

### 5.3 原理

LayerNorm 将均值归一为 0，方差归一为 1，使 magnitude = sqrt(feat_channels) ≈ 19.6，与 lvl_embed 和 content_proj 处于同一数量级。不改变方向（cos_sim 保持 0.752），仅放大幅度。

### 5.4 修改文件

- `projects/LDMDet/mods/box_tokenizer.py` — 添加 `pos_norm` + forward 中使用
- `projects/LDMDetDiT/mods/box_tokenizer.py` — 同步修改

---

## 6. 讨论

### 6.1 为什么不减小 lvl_embed 幅度？

减小 lvl_embed 是治标：让噪声变小，但信号（pos_embed）仍然弱。放大 pos_embed 使信号与噪声在同一水平。

### 6.2 是否需要处理 content_proj？

content_proj 的 cos_sim=0.984 仍会稀释区分度。但在真实图像上（训练时），不同框的内容特征应具有差异。当前推理时使用随机噪声图，内容同质化是预期行为。

### 6.3 是否需要正弦位置编码？

当前 MLP 区分度（0.752）优于正弦（0.887）。如果未来需要更强的区分度，可考虑 Fourier 特征编码（增加频率维度）。

### 6.4 LDMDetDiT 的差异

LDMDetDiT 版本已通过 `nn.init.normal_(level_embed, std=0.01)` 减小 lvl_embed，但 pos_embed 仍只有 magnitude ~1.5。添加 pos_norm 后两者的修复思路互补。

---

## 7. 相关诊断脚本

- `projects/LDMDet/tests/diagnose_tokenizer_encoding.py` — Box Tokenizer 编码区分度分析
- `projects/LDMDet/tests/diagnose_box_independence.py` — 预测框多样性与 token 相似度分析