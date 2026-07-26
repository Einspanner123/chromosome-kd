# 数据校验审计日志 (Data Audit Log)

> 本文档记录对论文相关数据的独立校验过程, 确保所有 mAP 数值的 val/test 来源标注完整且一致。
> 校验方式: 通过干净的 subagent (无先验上下文) 独立审计, 不参与修复操作。
> 关联文档: [EXPERIMENT_LINEAGE.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/EXPERIMENT_LINEAGE.md), [EXPERIMENT_CATALOG.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/EXPERIMENT_CATALOG.md), [paper_draft_CN.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/paper/paper_draft_CN.md)

---

## 审计 #1: 2026-07-26 val/test 来源标注全量校验

### 触发背景

用户要求: "更新并记录检查是否所有结果都在 test 上得到的数据, 必须通过干净的 subagent 独立进行数据校验"。

此前的会话中已完成:
1. 9 模型 test set 评估 (results/test_eval_20260726_181932/, 含 ANALYSIS.md)
2. LINEAGE/CATALOG 文档补充 test mAP 标注
3. Cascade R-CNN 路径双拼接 / YOLOX-S EMAHook / A1 checkpoint 错误等修复

本次审计为"独立第三方复核", 验证前序修改是否完整、一致、无遗漏。

### 审计方法

- 启动 `general-purpose` subagent (agent_id: 13d885ca-1ad6-44be-8026-53fbea406a33), 无任何先验上下文
- 审计范围: 论文草稿 / LINEAGE / CATALOG / ANALYSIS.md / 9 个模型原始 .log 文件
- 审计任务分 4 类:
  - **任务 A**: 9 模型 test mAP 一致性 (log 提取值 vs ANALYSIS 记录值)
  - **任务 B**: 论文草稿 val/test 标注充分性
  - **任务 C**: LINEAGE vs CATALOG 数值一致性
  - **任务 D**: 5 项关键叙事一致性 (A3 唯一显著下降 / Top-K test 增强 / K=100 robust / DiffusionDet 0.787→0.803 / RTMDet-L=A3 val)

### 审计结论

| 任务 | 范围 | 结论 |
|------|------|------|
| A | 9 模型 test mAP | ✅ **全部通过** (9/9 数值精确到 4 位小数一致) |
| B | 论文 val/test 标注 | ⚠ **有问题** (7 处 P1 标注缺失) |
| C | LINEAGE vs CATALOG | ⚠ **有 P0 问题** (DiffusionDet val mAP 在 CATALOG §6.4/§3.3 仍为 0.787) |
| D | 5 项关键叙事 | ✅ **基本通过** (4/5 完全一致, 1 项 DiffusionDet 更新在 CATALOG 两处未同步) |

### 发现的问题清单

#### P0 级 (数据错误或严重不一致)

| 编号 | 问题 | 位置 | 修复方式 | 修复状态 |
|------|------|------|----------|----------|
| P0-1 | CATALOG §6.4 FPS 表 DiffusionDet val mAP=0.787 (CRASHED run), test mAP=0.804 (ep26 checkpoint), val/test 来自不同 checkpoint, Δ=+0.017 误导 | CATALOG L985 | val mAP 改为 0.803 (与 test 同 checkpoint) | ✅ 已修复 |
| P0-2 | CATALOG §3.3 SOTA 表 DiffusionDet mAP=0.787 (CRASHED run), 未注明论文已改用 0.803 | CATALOG L479, L481 | mAP 改为 0.803, 表下注释补充 DiffusionDet 修正说明 | ✅ 已修复 |

#### P1 级 (标注缺失但不影响结论)

| 编号 | 问题 | 位置 | 修复方式 | 修复状态 |
|------|------|------|----------|----------|
| P1-1 | 论文 Table 10 caption 未标 "val" (仅 "seed 42"), 且 A3 DPM++ 在 Table 5 (0.859) vs Table 10 (0.863) 差异未解释 | 论文 L381 | caption 加 "Dataset 2 验证集, seed 42 best checkpoint; mAP 与 FPS 测量使用同一 checkpoint, 3-seed 均值见 Table 5, 测试集评估见 §4.5.4" | ✅ 已修复 |
| P1-2 | 论文 Table 5/6 caption 用 "独立推理" 而非 "val set" | 论文 L236, L257 | caption 改为 "Dataset 2 验证集; A0–A2/比较方法为 seed 42 best checkpoint, A3 为 3-seed 均值" | ✅ 已修复 |
| P1-3 | 论文 §4.5.4 未说明 test 0.859 = val 3-seed mean 0.859 巧合, 错失泛化性证据 | 论文 L363 | §4.5.4 重写, 显式说明 test 0.859 与 val 3-seed mean 0.859±0.003 完全一致 | ✅ 已修复 |
| P1-4 | 论文 §4.3.1 Dataset 1 KaryoFlow mAP=0.753 为单 seed best, 与 Dataset 2 用 3-seed mean 不一致 | 论文 L259 | 改为 "3-seed 均值 0.747±0.003 (单 seed 最佳 0.753)" | ✅ 已修复 |
| P1-5 | 论文未利用 test set 强化 Top-K 叙事 (K=300 反超 / K=200 持平 / K=100 robust) | 论文 §4.5.4 | §4.5.4 补充 Top-K test 叙事段 | ✅ 已修复 |
| P1-6 | CATALOG §3.3 表下注释仅提 YOLOX-S 修正, 未提 DiffusionDet | CATALOG L481 | 注释补充 DiffusionDet 0.787→0.803 修正说明 | ✅ 已修复 (与 P0-2 合并) |
| P1-7 | 论文 Table 6 DINO R50 mAP=0.868 来自 CRASHED run, 未标注 | 论文 L257 | caption 加 "DINO R50 数值取自最终崩溃状态 checkpoint (训练未完整完成), 其 0.868 应视为该 checkpoint 的上界估计" | ✅ 已修复 |

#### P2 级 (格式统一性建议)

| 编号 | 问题 | 位置 | 修复方式 | 修复状态 |
|------|------|------|----------|----------|
| P2-1 | LINEAGE §十七 / CATALOG §3.3 SOTA 表 mAP 列未标 "val" | LINEAGE L1138-1146, CATALOG L471 | 表头改为 "mAP (val)"; LINEAGE 进一步加 "mAP (test)" 双列 | ✅ 已修复 |
| P2-2 | CATALOG §1.2 Cascade R-CNN mAP=0.854 未标 "val" | CATALOG L101 | 改为 "0.854 val (test: 0.853)" 格式, 同时 RTMDet-L/DINO R50/YOLOX-S 同步加 "val" | ✅ 已修复 |
| P2-3 | 命名约定不一致: 论文 A3 DPM++ = CATALOG A4 DPM-Solver++; 论文 A2 Stoch. Coup. = CATALOG A3 StochOT eps5 | 论文 vs CATALOG | CATALOG §1.1 已有注释 "论文 Table 5/6 中 'A2 + Stoch. Coup.' 对应此 checkpoint"; A4 行注释已含 "论文 'A3 DPM++' 对应此 checkpoint" (前序已存在, 本次未重复) | ⏸ 已存在 (未改动) |
| P2-4 | 论文 §1 "0.859 对 0.868, 3-seed 均值" 修饰对象歧义 | 论文 L99 | 维持现状 (上下文已清晰: 0.859 为 3-seed mean, 0.868 为 DINO 单 seed) | ⏸ 维持现状 |
| P2-5 | CATALOG §3.3 表标题 "基线对比实验" 但 DiffusionDet 用 CRASHED 值 | CATALOG L462 | 状态列已标 "⚠ 早期 CRASHED, 已用 ep26 替代" | ✅ 已修复 (与 P0-2 合并) |

### 9 模型 val vs test 一致性表 (审计后状态)

| 模型 | Val mAP (seed42) | Test mAP | Δ (test−val) | 数据源 (test) | 备注 |
|------|:---------:|:--------:|:----------:|---------------|------|
| A1 RF+Heun | 0.856 | 0.857 | +0.001 | a1_rf_heun.log (L1606) | 稳定 |
| A2 + Stoch. Coup. | 0.858 | 0.858 | 0.000 | a2_stochot_heun.log (L1613) | 稳定 |
| **A3 DPM++** | **0.863** | **0.859** | **−0.004** | a3_dpm_pp.log (L1613) | 唯一显著下降; test 0.859 = val 3-seed mean 0.859 |
| A3+TopK K=300 | 0.861 | 0.860 | −0.001 | a3_io3_k300.log (L1616) | 稳定; test 上反超 A3 |
| A3+TopK K=200 | 0.860 | 0.859 | −0.001 | a3_io3_k200.log (L1616) | 稳定; test 上与 A3 持平 |
| A3+TopK K=100 | 0.850 | 0.847 | −0.003 | a3_io3_k100.log (L1616) | 稳定; K=100 有害结论 robust |
| Cascade R-CNN | 0.854 | 0.853 | −0.001 | cascade_rcnn_rerun.log (L1719) | 稳定 (首次因路径双拼接失败, 已修复) |
| YOLOX-S | 0.796 | 0.795 | −0.001 | yolox_s_rerun.log (L1348) | 稳定 (首次因 EMAHook 失败, 已修复) |
| DiffusionDet | 0.803 | 0.804 | +0.001 | diffusiondet.log (L1564) | 稳定 (val 0.803 来自 ep26 checkpoint, 非 CRASHED 0.787) |

### 文档修改清单

本次审计后修改的文件:

1. **[EXPERIMENT_CATALOG.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/EXPERIMENT_CATALOG.md)**:
   - §1.2: RTMDet-L / DINO R50 / Cascade R-CNN / YOLOX-S 行加 "val" / "test" 标注 (P2-2)
   - §3.3: 表头 mAP → mAP (val); DiffusionDet 行 0.787 → 0.803, 状态列加 "已用 ep26 替代" (P0-2, P2-1, P2-5)
   - §3.3 表下注释: 补充 DiffusionDet 0.787→0.803 修正说明 (P1-6)
   - §6.4 FPS 表: DiffusionDet 行 val mAP 0.787 → 0.803 (P0-1)

2. **[EXPERIMENT_LINEAGE.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/EXPERIMENT_LINEAGE.md)**:
   - §十七 SOTA 表: 标题加 "val", mAP 列拆为 mAP (val) + mAP (test) 双列, 各行加 test mAP + 增强备注 (P2-1)

3. **[paper_draft_CN.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/paper/paper_draft_CN.md)**:
   - Table 5 caption (L236): "独立推理" → "Dataset 2 验证集", 加 test 评估交叉引用 (P1-2)
   - Table 6 caption (L257): "独立推理" → "Dataset 2 验证集", 加 DINO R50 CRASHED 说明 (P1-2, P1-7)
   - §4.3.1 (L259): Dataset 1 KaryoFlow 0.753 → 3-seed 均值 0.747±0.003 (单 seed 最佳 0.753) (P1-4)
   - §4.5.4 (L363): 重写, 加 9 模型 test 评估背景 / test=val 3-seed mean 巧合 / Top-K test 叙事增强 (P1-3, P1-5)
   - Table 10 caption (L381): 加 "Dataset 2 验证集, seed 42 best checkpoint", 加 3-seed mean 与 test 评估交叉引用 (P1-1)

### 关键叙事最终状态

1. ✅ **A3 DPM++ 唯一 val→test 显著下降 (Δ=−0.004)**: 所有文档一致
2. ✅ **Top-K K=200 在 test set 上无 mAP 损失**: 论文 §4.5.4 已显式提及 "K=200 与 A3 完全持平 (Δ=0.000, 验证集为 −0.003)"
3. ✅ **K=100 有害结论在 test set 上 robust**: 论文 §4.5.4 已显式提及 "测试集 −0.012 对验证集 −0.013"
4. ✅ **DiffusionDet 主列 mAP 已从 0.787 更新为 0.803**: 所有文档 (论文/LINEAGE/CATALOG/ANALYSIS) 同步
5. ✅ **RTMDet-L mAP=0.863 (val) = A3 DPM++ val seed42 best 0.863**: 所有文档一致; 论文 Table 6 用 A3 的 3-seed mean (0.859) 进行 SOTA 对比

### 后续建议 (非本次审计范围)

- CATALOG §3.3 DINO R50 行的 mAP=0.868 (CRASHED, count=93) 暂未替换, 因无替代 checkpoint; 若需进一步处理可考虑重训 DINO R50 (但成本高, 且当前论文叙事已明确标注 CRASHED 状态, 不影响结论)
- 若 arXiv companion preprint 引用本审计, 可将 9 模型 val/test 对照表与 ANALYSIS.md 完整日志链接放入 supplementary (受 TMI 禁止 supplementary text materials 限制, 主刊不放入)

### 审计员签名

- 审计执行: 独立 subagent (agent_id: 13d885ca-1ad6-44be-8026-53fbea406a33, model: GLM-5.2)
- 修复执行: 主会话 (基于审计报告)
- 修复完成日期: 2026-07-26

---

## 审计 #2: (待未来触发)

后续若再次进行 val/test 数据校验, 在此追加新章节, 保持历史可追溯。
