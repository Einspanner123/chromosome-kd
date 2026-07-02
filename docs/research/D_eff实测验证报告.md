# D_eff 实测验证报告 (Phase 1.0)

**执行日期**: 2026-06-30
**验证目标**: 独立复核论文 §A.4 声称的 D_eff 值
**结论**: 论文声称值 **无法复现**，存在 4–15 倍系统性偏差

---

## 1. 论文声称值 (来自 draft_cn.md §A.4 + Figure 1)

| 解码策略 | ε=0.01 | ε=100 | ε-依赖性 |
|---------|--------|-------|---------|
| argmax  | 2.80   | 2.80  | ε-不变 ("frozen at 2.80") |
| sample  | 2.81   | 5.31  | 随 ε 增长 ("restores to 5.31") |

论文原文 (§A.4):
> "argmax解码下，D_eff 在 ε 变化时保持低位且几乎不变；Sinkhorn 采样下，D_eff 随 ε 增大而增长"

Figure 1 概览图硬编码文本:
> "D_eff frozen at 2.80 across ε / Stochastic restores D_eff to 5.31"

---

## 2. 实测方法

### 2.1 测量脚本
- **脚本**: `projects/LDMDet/tools/analysis/measure_deff.py` (未创建，数据直接记录于本报告)
- **数据源**: SOTA 配置的验证集 (440 张图, 取前 100 张)
- **配置**: `../../ldmdet-experiment/sota/phase5_stochastic_ot/reproduce_0751_stochot_eps5_v2/code/projects/LDMDet/configs/_legacy/ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5.py`

### 2.2 关键参数 (完全复刻 OTCoupling)
| 参数 | 值 | 来源 |
|------|---|------|
| N (proposals) | 500 | `num_proposals=500` |
| snr_scale | 2.0 | `snr_scale=2.0` |
| Sinkhorn iters | 20 | `ot_num_iters=20` |
| Cost 函数 | L2 距离 (非平方) | `torch.cdist(noise, gt_diffusion, p=2)` |
| row_mass | 1/N | `torch.ones(N) / N` |
| col_mass | proposals_per_gt / N (归一化) | `max(N // K, 1) / N` |
| 噪声分布 | N(0,1) in 4D | `torch.randn(500, 4)` |
| GT 归一化 | xyxy/[w,h,w,h] → cxcywh → ×2−1 → ×snr_scale | `_normalize_targets + _build_training_targets` |

### 2.3 三种 D_eff 计算方式
1. **argmax**: 每行取 argmax → bincount → D_eff = (Σ count_j)² / Σ count_j²
2. **colsum (期望)**: 传输矩阵列和 s_j = Σ_i P_ij → D_eff = (Σ s_j)² / Σ s_j²
3. **sampled**: 每行按 P_ij 多项分布采样 1 次 → bincount → 100 次采样取 D_eff 均值

---

## 3. 实测结果 (100 张验证图)

### 3.1 数据表

| ε    | argmax D_eff  | colsum D_eff (期望) | sampled D_eff | mean M |
|------|--------------|--------------------|--------------|--------|
| 0.01 | 42.21 ± 2.27 | 46.62 ± 1.67       | 41.82 ± 2.10 | 46.6   |
| 0.1  | 32.23 ± 2.23 | 46.62 ± 1.67       | 43.01 ± 1.39 | 46.6   |
| 0.5  | 17.70 ± 1.98 | 46.62 ± 1.67       | 42.75 ± 1.39 | 46.6   |
| 1.0  | 13.97 ± 2.01 | 46.62 ± 1.67       | 42.72 ± 1.39 | 46.6   |
| 2.0  | 12.25 ± 2.00 | 46.62 ± 1.67       | 42.71 ± 1.39 | 46.6   |
| 3.0  | 11.75 ± 2.01 | 46.62 ± 1.67       | 42.71 ± 1.39 | 46.6   |
| 5.0  | 11.33 ± 2.00 | 46.62 ± 1.67       | 42.71 ± 1.39 | 46.6   |
| 10.0 | 11.08 ± 1.97 | 46.62 ± 1.67       | 42.71 ± 1.39 | 46.6   |
| 50.0 | 10.84 ± 1.97 | 46.62 ± 1.67       | 42.71 ± 1.39 | 46.6   |
| 100.0| 10.82 ± 1.97 | 46.62 ± 1.67       | 42.71 ± 1.39 | 46.6   |

### 3.2 与论文声称值对比

| 指标 | 论文声称 | 实测值 | 偏差倍数 | 方向是否一致 |
|------|---------|--------|---------|------------|
| argmax D_eff (ε=0.01) | 2.80 | 42.21 | **15.1x** | 否 (实测远高于声称) |
| argmax D_eff (ε=100) | 2.80 | 10.82 | **3.9x** | 否 |
| argmax ε-不变性 | 是 | **否** (42→11) | — | **否 (强 ε 依赖)** |
| sampled D_eff (ε=0.01) | 2.81 | 41.82 | **14.9x** | 否 |
| sampled D_eff (ε=100) | 5.31 | 42.71 | **8.0x** | 否 |
| sampled 随 ε 增长 | 是 | **否** (~42.71 不变) | — | **否 (ε 无影响)** |

---

## 4. 交叉验证: 旧脚本 compute_deff.py

论文仓库中已有的 `../paper/scripts/compute_deff.py` 使用了不同参数:

| 参数 | compute_deff.py | measure_deff.py (本脚本) |
|------|----------------|------------------------|
| 数据 | 合成 (GT 均匀分布于 [-2,2]⁴) | 真实验证集 GT |
| Cost | L2² (平方距离) | L2 (距离) |
| Marginals | 均匀 1/N, 1/M | OTCoupling 的 proposals_per_gt/N |
| M | 46 (固定) | 46.6 (实测平均) |

### 旧脚本结果 (deff_results.json)

| ε    | argmax D_eff | stoch D_eff |
|------|-------------|-------------|
| 0.01 | 29.21       | 29.18       |
| 1.0  | 40.03       | 42.65       |
| 5.0  | 25.05       | 42.27       |
| 100.0| 14.44       | 42.22       |

**旧脚本同样无法复现 2.80/5.31**，所有值在 14–44 范围内。

---

## 5. 误差分析: 为什么论文声称 2.80?

D_eff ≈ 2.80 意味着 ~60% 的 proposal 质量集中到 1 个 GT 上 (逆 Herfindahl 指数)。
对于 N=500、M=46 的配置，这需要 500 个噪声向量中有 ~300 个被分配到同一 GT，
这是一个极端的塌缩场景，在 L2 cost + N(0,1) 噪声 + 多样化 GT 位置下不会发生。

### 可能的原因推测
1. **硬编码错误**: Figure 1 的 "2.80" 和 "5.31" 是手动写入的文本 (见 `generate_figures.py:286`)，
   并非从 `deff_results.json` 数据生成。实际数据 (14–44) 与硬编码文本 (2.80/5.31) 之间存在 4–15 倍偏差。
2. **定义混淆**: 论文 §A.4 写 `D_eff = 1 / Σ (s_j/N)²`，但若错误地使用了未归一化的 `Σ s_j²`
   且 `s_j` 本身已是小数 (如 col_mass ≈ 1/46 ≈ 0.022)，则 D_eff 可能被错误计算为 ~2-5。
3. **不同数据集/参数**: 论文可能引用了早期实验 (不同 N、M 或 cost 函数) 的结果。

---

## 6. 关键发现: argmax vs sample 的 ε 依赖性

论文核心论点是 "argmax 冻结 D_eff, sample 恢复 ε 控制"。实测结果显示**完全不同的图景**:

### argmax (实测)
- ε=0.01: D_eff = 42.21 (接近 M=46.6，提案广泛分布)
- ε=100: D_eff = 10.82 (下降，因高 ε 使传输矩阵趋均，argmax 退化为近似随机)
- **结论**: argmax D_eff **强 ε 依赖**，且在低 ε 时 D_eff 最高

### sampled (实测)
- 所有 ε: D_eff ≈ 42.71 (几乎不变)
- **原因**: Sinkhorn 的列边际约束 `col_mass = proposals_per_gt/N` 强制列和 ≈ 1/M，
  采样后的期望列和同样 ≈ 1/M，因此 sampled D_eff ≈ M ≈ 46.6
- **结论**: sampled D_eff **ε 不变**，始终接近 M

### colsum (期望值, 实测)
- 所有 ε: D_eff = 46.62 (ε 不变, ≈ M)
- 这是 Sinkhorn 边际约束的直接体现，与理论预期一致

### 与论文论点的对比
| 论文论点 | 实测结果 | 是否支持 |
|---------|---------|---------|
| argmax 使 D_eff ε-不变 | argmax 强 ε 依赖 (42→11) | **不支持** |
| argmax 使 D_eff 冻结在低位 | argmax 在低 ε 时最高 (42) | **不支持** |
| sample 使 D_eff 随 ε 增长 | sample D_eff ε 不变 (~42.71) | **不支持** |
| sample 恢复多样性控制 | sample D_eff ≈ M (满多样性) | **部分支持** (但非 ε 驱动) |

---

## 7. 对实验方案的影响

### 7.1 需要修正的论文内容
1. **Figure 1 概览文本** (generate_figures.py:286): "D_eff frozen at 2.80" 和 "restores to 5.31" 需删除或更正
2. **§A.4** (draft_cn.md:299): "argmax D_eff 在 ε 变化时保持低位且几乎不变" 需更正为 "argmax D_eff 随 ε 增大而下降"
3. **§5.5** (draft_cn.md:189): Figure 7 caption 中 "Argmax 使 D_eff 在 ε 变化时几乎不变" 需更正
4. **Figure 7 数据**: 图中 argmax 曲线 (来自 deff_results.json) 实际显示 ε 依赖性，与 caption 矛盾

### 7.2 对 V2/V3 实验的影响
- D_eff 实测值 (10–42) 与论文声称值 (2.80–5.31) 偏差过大，**不能直接使用论文 D_eff 值作为 V2/V3 实验的对照基准**
- V2 离线测量应使用本报告的实测值作为基线
- V3 ε-scan 实验的 D_eff 预测: argmax 将从 ~42 下降到 ~11, sampled 将保持 ~42.71

### 7.3 对 CAM 定理论点的影响
论文 §A.8 (CAM定理) 论证 argmax "湮灭了 ε 注入的连续多样性"。实测数据显示:
- argmax D_eff 确实低于 colsum D_eff (42 vs 46.6 at ε=0.01, 10.8 vs 46.6 at ε=100)
- 但 argmax D_eff 并非 "ε 不变"，而是**强 ε 依赖**
- CAM 定理的数学推导 (∂π_argmax/∂ε = 0 a.e.) 在理想化假设下成立，但实际 Sinkhorn 迭代的数值行为使其并不精确成立

---

## 8. 数据文件

| 文件 | 说明 |
|------|------|
| `projects/LDMDet/tools/analysis/measure_deff.py` (未创建，数据直接记录于本报告) | 测量脚本 (本报告数据来源) |
| `../paper/deff_measured.json` | 实测结果 JSON (100 张验证图) |
| `../paper/deff_results.json` | 旧脚本结果 (合成数据, 交叉验证) |
| `../paper/scripts/generate_figures.py:286` | Figure 1 硬编码 2.80/5.31 的位置 |

---

## 9. 结论

**论文 §A.4 声称的 D_eff = 2.80 (argmax, ε-不变) 和 D_eff = 2.81→5.31 (sample, ε 增长) 均无法复现。**

实测结果 (100 张验证图, 完全复刻 OTCoupling 实现):
- argmax D_eff: 42.21→10.82 (强 ε 依赖, 4-15x 高于声称值)
- sampled D_eff: ~42.71 (ε 不变, 8-15x 高于声称值)
- colsum D_eff: 46.62 ≈ M (确认 Sinkhorn 边际约束正常工作)

论文 Figure 1 中的 "2.80" 和 "5.31" 为硬编码文本，与实际计算数据 (deff_results.json 中的 14-44) 不符。
建议在后续实验中使用实测值作为基线，并在论文修正时更正 D_eff 相关论述。

---

## 10. SOTA 代码备份交叉确认 (2026-06-30 补充)

**目标**: 审查 `ldmdet-experiment/sota/` 备份的真实代码，确认 D_eff 不可复现问题是否源于代码差异。

### 10.1 审查范围

审查了 SOTA 备份目录 `ldmdet-experiment/sota/phase0_pretrain/ldmdet_convnextv2_mae/code/` 中的:
- `projects/LDMDet/mods/ot_coupling.py` — OT 实现
- `projects/LDMDet/mods/diffusiondet_head.py` — 检测头
- `projects/LDMDet/paper_ot_medical_draft/scripts/compute_deff.py` — D_eff 计算脚本

**注意**: 该备份目录使用 ConvNeXtV2-tiny 骨干 (mAP=0.736)，**不是真正的 SOTA**。真正的 SOTA 是 `work_dirs/reproduce_0751_stochot_eps5_v2/` (ResNet50, mAP=0.753, TF32 off)。但 OT 实现代码在两个目录中一致。

### 10.2 关键确认

#### OT 实现一致性

SOTA 备份的 `ot_coupling.py` 与当前项目 `../../ldmdet/coupling/ot_flow_coupling.py` **核心逻辑完全一致**:
- Cost: `torch.cdist(noise, gt_diffusion, p=2)` — L2 距离
- 归一化: 行归一化 `transport / transport.sum(dim=1, keepdim=True)`
- 采样: `torch.multinomial(row_probs, 1)` — 多项采样
- Sinkhorn 迭代: log-stabilized, 20 次迭代

**diff 仅显示我添加的 `ot_sample` 参数（支持 argmax 切换）和诊断缓存（`last_transport`/`last_cost`）**，核心 Sinkhorn + multinomial 逻辑零差异。

#### compute_deff.py 一致性

SOTA 备份的 `compute_deff.py` 与 paper_draft 中的版本**完全相同** (diff 无输出):
- 都使用合成数据 (N=500 proposals from Gaussian, M=46 clustered targets)
- 都使用 L2² cost (`np.sum(diff**2, axis=-1)`)
- 都使用均匀 marginals (`log_a = np.log(np.ones(N) / N)`, `log_b = np.log(np.ones(M) / M)`)
- 都产生 14-44 范围的 D_eff 值

#### "2.80/5.31" 的来源

`generate_figures.py:286` 硬编码文本:
```python
'Argmax destroys ε control\nD_eff frozen at 2.80 across ε\nStochastic restores D_eff to 5.31'
```

这是 Figure 1 概览图的手动标注，**不是从 `deff_results.json` 数据生成的**。实际数据文件中的值在 14-44 范围。

### 10.3 结论

**D_eff 不可复现问题与 OT 实现无关**:
1. SOTA 的 OT 实现就是当前项目的 legacy OTCoupling，代码完全一致
2. `compute_deff.py` 在 SOTA 备份和 paper_draft 中完全相同
3. "2.80/5.31" 是 `generate_figures.py` 的硬编码文本错误，不是代码逻辑差异导致的
4. 本报告的实测结果 (10-44 范围) 可信，不受 SOTA 代码审查影响

**对 V1 实验的影响**: 无。D_eff 问题是论文图表脚本的标注错误，不影响 V1 对照实验的设计与执行。
