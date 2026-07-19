# C 类推理任务规划（阶段 4）

> 创建日期: 2026-07-18
> 论文: KaryoFlow (Rectified Flow for Chromosome Detection)
> 状态: 草案（待 A14 完成后定稿）
> 约束: 用户要求"推理任务最后计划"

---

## 一、背景

5 位审稿人核心质疑（平均 4.0/10）：
1. **单种子结果** — 主要结果仅基于 seed 42
2. **无统计检验** — mAP 差异未做显著性检验
3. **SOTA 选择性遗漏** — 已通过 A1 解决
4. **理论证明缺口** — 已通过 A2/A3 解决
5. **临床声明缺失** — 已通过 A6 解决（移除）

C 类任务目标：使用已有 ckpt 进行推理，补充多种子验证、统计检验、测试集评估等，回应审稿人质疑。

---

## 二、已有 ckpt 清单

### Dataset 2 (24obj, 5000 张) — 主要实验

| 实验 | seed 42 | seed 123 | seed 789 | 说明 |
|------|---------|----------|----------|------|
| A0 baseline (Euler) | best_ep12 | — | — | ⚠️ 仅 1 seed, ep12 可能未收敛 |
| A1 RF+Heun | best_ep62 | — | — | 仅 1 seed |
| A2 RF+Heun+AdaLN | best_ep82 | — | — | 仅 1 seed |
| A3 full SOTA | best_ep114 | — | — | 仅 1 seed（= A2 + StochOT） |
| **A4 DPM-Solver++** | best_ep117 (0.863) | best_ep62 (0.857) | best_ep72 (0.856) | ✅ 3 seeds |
| RF+Heun+AdaLN (multi) | best_ep47 (0.712*) | best_ep47 (0.718*) | best_ep25 (0.708*) | ⚠️ 可能是 Dataset 1 |
| Hard OT (multi) | best_ep30 (0.705*) | best_ep49 (0.703*) | best_ep43 (0.707*) | ⚠️ 可能是 Dataset 1 |
| StochOT eps5 (multi) | best_ep60 (0.746*) | best_ep57 (0.746*) | best_ep69 (0.749*) | ⚠️ 可能是 Dataset 1 |

> *标星数值约 0.7x，推测为 Dataset 1（original, mAP≈0.72-0.75）而非 Dataset 2

### Dataset 1 (original/Chromosome20240904, 1540 张)

| 实验 | seed 42 | seed 123 | 说明 |
|------|---------|----------|------|
| LDMDet random | best_ep68 (0.745) | best_ep7 (0.550) | ⚠️ seed_123 可能训练失败 |
| RTMDet-L | ❓ | — | **未找到 Dataset 1 ckpt** |
| DINO R50 | ❓ | — | **未找到 Dataset 1 ckpt** |

### Baselines (Dataset 2)

| 模型 | ckpt 路径 | mAP |
|------|----------|-----|
| Cascade R-CNN | work_dirs/baselines/cascade_rcnn_r50 | 0.854 |
| YOLOX-S | work_dirs/baselines/yolox_s | 0.796 |
| DiffusionDet | work_dirs/baselines/diffusiondet_24obj | 0.787 |
| RTMDet-L | work_dirs/baselines/rtmdet_l_24obj | 0.863 (ep86 中断) |
| DINO R50 | ❓ | 0.869（来源待核实） |

---

## 三、C 类任务清单

### C1: A4 (DPM-Solver++) 多种子推理验证 [P0]

**目的**：验证主结果 A3=0.863 mAP 的跨种子稳定性

**已有数据**（从训练 scalars.json 提取的 best mAP）：
- seed 42: 0.863, seed 123: 0.857, seed 789: 0.856
- mean = 0.859, std = 0.003

**待执行推理**：
- 对 3 个 seed 的 best ckpt 运行 `test.py --dataset val`
- 收集完整指标：mAP/AP50/AP75/AP_S/AP_M/AP_L
- 计算 mean ± std
- 更新论文 SOTA 表（当前仅报告 seed 42 单值）

**命令模板**：
```bash
python experiments/runners/test.py \
  experiments/configs/ldmdet/directions/mainline_ablation_24obj/a4_dpm_pp_24obj.py \
  --checkpoint work_dirs/multi_seed/a4_dpm_pp_24obj/seed_789/best_coco_bbox_mAP_epoch_72.pth \
  --dataset val --gpu-id 0 --solver-type dpm_solver_pp --sampling-steps 4
```

**优先级**：P0（直接回应"单种子"质疑）

---

### C2: 耦合消融多种子推理验证 [P0]

**目的**：验证 Stochastic Coupling 的稳定性声明（4.6× epoch std 降低）

**已有数据**（推测为 Dataset 1）：
- Hard OT: 0.705 ± 0.002 (3 seeds)
- RF+Heun+AdaLN: 0.713 ± 0.004 (3 seeds)
- StochOT eps5: 0.747 ± 0.001 (3 seeds)

**待核实**：
1. 确认这些多种子实验是在哪个数据集上（Dataset 1 or 2）
2. 如果是 Dataset 1，需要补充 Dataset 2 的耦合多种子实验
3. 运行推理收集 per-image AP 用于统计检验

**优先级**：P0（回应"单种子"+"无统计检验"）

---

### C3: 统计显著性检验 [P0]

**目的**：对关键 mAP 差异进行配对统计检验

**待执行**：
1. 收集 per-image AP（COCO evaluator 输出）
2. 对以下对比进行 paired t-test / Wilcoxon signed-rank test：
   - A3 (DPM++) vs A2 (Heun) — 验证 +0.005 mAP 是否显著
   - StochOT vs Random — 验证 +0.002 mAP 是否显著（预期不显著）
   - LDMDet vs DiffusionDet — 验证 +0.076 mAP 是否显著
3. 在论文中添加 p 值报告

**优先级**：P0（直接回应"无统计检验"质疑）

---

### C4: 测试集评估 [P1]

**目的**：在 test split 上验证模型泛化性

**待执行**：
```bash
python experiments/runners/test.py \
  experiments/configs/ldmdet/directions/mainline_ablation_24obj/a4_dpm_pp_24obj.py \
  --checkpoint work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth \
  --dataset test --gpu-id 0
```

**约束**（来自 project_memory）：
- `--dataset val` 时必须使用 val_dataloader
- 测试集评估需确保 annotation 文件路径正确

**优先级**：P1（增强模型泛化性声明）

---

### C5: Dataset 1 SOTA 数值核实 [✅ 已解决 — 数据完整]

**问题**：论文 A1 段落（L643-646）声称：
> "On Dataset~1, where the training set is smaller (1,540 images), our LDMDet (0.753 mAP) in fact exceeds both RTMDet-L (0.742) and DINO R50 (0.737)"

**核实结果**（2026-07-18）：
- ✅ **LDMDet 0.753**：来自 `ldmdet-experiment/sota/phase5_stochastic_ot/reproduce_0751_stochot_eps5_v2`，是 StochOT ε=5 变体（非 random coupling 的 0.745）。来源：PAPER_RESULTS.md L78, EXPERIMENT_CATALOG.md L298
- ✅ **RTMDet-L 0.742**：来自 SwanLab benchmark。来源：PAPER_RESULTS.md L79, L201
- ✅ **DINO R50 0.737**：来自 SwanLab benchmark。来源：PAPER_RESULTS.md L80, L202

**结论**：三个数值均有实验依据，A1 段落声明正确，无需修改论文。

---

### C6: Per-class AP 多种子稳定性 [P2]

**目的**：验证 per-class AP（特别是 Y 染色体 AP=0.776）的跨种子稳定性

**待执行**：
- 对 A4 的 3 个 seed 运行推理，收集 per-class AP
- 计算 per-class mean ± std
- 特别关注 Y 染色体和 C 组的稳定性

**优先级**：P2（增强 per-class 分析可信度）

---

### C7: Shift 消融独立推理验证 [P1]

**目的**：验证 SG6 shift 消融数据（已从训练 scalars.json 提取，但未独立推理验证）

**已有数据**（从训练 log 提取）：
- Linear (shift=0): best mAP = 0.857 @ ep89
- Shifted (shift=3.0): best mAP = 0.856

**待执行**：
- 对两个 ckpt 运行独立推理验证
- 收集完整 AP_S/AP_M/AP_L 指标

**优先级**：P1（支撑新增 Appendix 的数据可信度）

---

### C8: FPS 基准复测 [P2] — ✅ 决策：跳过

**目的**：验证 FPS 表中各配置的 latency 数值

**已有数据**：FPS 表已包含 latency ± std（2026-07-14 测量，warmup=10, iters=100, 512×512, batch=1, 200 images）

**决策**：⏸️ **跳过** — 现有 FPS 数据方法论足够严谨，无需复测。

**理由**：
1. FPS 是确定性计算延迟（无推理随机性），复测结果在 ±2% 噪声范围内
2. 已使用 benchmark_fps.py 的 MODEL_REGISTRY 和 KNOWN_MAP 进行配置化测量
3. A3+IO3 K=200 的 14.2 FPS 声明已通过 benchmark 验证
4. FPS 表 caption 标注 "seed 42"，与主消融表 3-seed mean±std 在噪声范围内一致（A3: 0.863 vs 0.859±0.003）
5. 复测无信息增量，30 分钟 GPU 时间不值得

**优先级**：P2（已有数据，可选验证）

---

## 四、执行优先级排序

| 优先级 | 任务 | 理由 | 预计耗时 |
|--------|------|------|---------|
| ~~P0~~ | ~~C5: Dataset 1 SOTA 数值核实~~ | ✅ 已解决（数值均有实验依据） | — |
| **P0** | C1: A4 多种子推理 | 回应单种子质疑，3 ckpt 本地可用 | 15min（3 次推理） |
| **P0** | C2: 耦合多种子推理 | 回应单种子质疑，Dataset 1 上 3×3 ckpt | 45min（9 次推理） |
| **P0** | C3: 统计显著性检验 | 回应无统计检验质疑，需 per-image AP | 30min（分析+报告） |
| **P1** | C4: 测试集评估 | 增强泛化性声明 | 5min |
| **P1** | C7: Shift 消融验证 | 支撑新增 Appendix | 10min（2 次推理） |
| **P2** | C6: Per-class 多种子 | 增强分析深度 | 15min |
| ~~P2~~ | ~~C8: FPS 复测~~ | ✅ 跳过（现有数据方法论足够严谨） | — |

---

## 五、执行环境确认（2026-07-18）

| 资源 | 状态 |
|------|------|
| **本地 GPU** | RTX A6000, 49GB, 0% 利用率 ✅ |
| **Dataset 1 数据** | `data/Chromosome20240904_NoAug_NoResize_coco/` ✅ |
| **Dataset 2 数据** | `data/24_chromosomes_object/coco/` ✅ |
| **A4 ckpt (3 seeds)** | 本地全部可用 ✅ |
| **耦合 ckpt (3×3 seeds)** | 本地全部可用 ✅（Dataset 1） |
| **test.py** | 支持 --dataset val/test, --solver-type, --sampling-steps ✅ |
| **per-image AP 输出** | ⚠️ test.py 不支持，需修改脚本或用独立评估工具 |

---

## 六、具体执行命令

### C1: A4 多种子推理（Dataset 2, val）

```bash
# Seed 42 (主实验)
python experiments/runners/test.py \
  experiments/configs/ldmdet/directions/mainline_ablation_24obj/a4_dpm_pp_24obj.py \
  --checkpoint work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth \
  --dataset val --gpu-id 0 --seed 42

# Seed 123
python experiments/runners/test.py \
  experiments/configs/ldmdet/directions/mainline_ablation_24obj/a4_dpm_pp_24obj.py \
  --checkpoint work_dirs/multi_seed/a4_dpm_pp_24obj/seed_123/best_coco_bbox_mAP_epoch_62.pth \
  --dataset val --gpu-id 0 --seed 42

# Seed 789
python experiments/runners/test.py \
  experiments/configs/ldmdet/directions/mainline_ablation_24obj/a4_dpm_pp_24obj.py \
  --checkpoint work_dirs/multi_seed/a4_dpm_pp_24obj/seed_789/best_coco_bbox_mAP_epoch_72.pth \
  --dataset val --gpu-id 0 --seed 42
```

> 注：推理 seed 统一用 42（确保相同噪声），比较的是训练种子对模型权重的影响

### C2: 耦合多种子推理（Dataset 1, test）

```bash
# Hard OT (3 seeds)
for seed in 42 123 789; do
  ckpt=work_dirs/multi_seed/hard_ot/seed_${seed}/best_coco_bbox_mAP_epoch_*.pth
  python experiments/runners/test.py \
    work_dirs/multi_seed/hard_ot/seed_${seed}/hard_ot.py \
    --checkpoint $ckpt --dataset test --gpu-id 0 --seed 42
done

# RF+Heun+AdaLN (3 seeds) — 对应 Random coupling
for seed in 42 123 789; do
  ckpt=work_dirs/multi_seed/rf_heun_adaln/seed_${seed}/best_coco_bbox_mAP_epoch_*.pth
  python experiments/runners/test.py \
    work_dirs/multi_seed/rf_heun_adaln/seed_${seed}/rf_heun_adaln.py \
    --checkpoint $ckpt --dataset test --gpu-id 0 --seed 42
done

# StochOT eps5 (3 seeds)
for seed in 42 123 789; do
  ckpt=work_dirs/multi_seed/stochot_eps5_old/seed_${seed}/best_coco_bbox_mAP_epoch_*.pth
  python experiments/runners/test.py \
    experiments/configs/ldmdet/directions/mainline_ablation_old/stochot_eps5_old.py \
    --checkpoint $ckpt --dataset test --gpu-id 0 --seed 42
done
```

### C3: 统计显著性检验

**需要 per-image AP**：修改 test.py 或使用独立脚本收集 per-image COCO eval 结果，然后：
```python
from scipy.stats import wilcoxon, ttest_rel
# A3 vs A2
stat, p_value = wilcoxon(a3_per_image_ap, a2_per_image_ap)
```

### C4: 测试集评估
```bash
python experiments/runners/test.py \
  experiments/configs/ldmdet/directions/mainline_ablation_24obj/a4_dpm_pp_24obj.py \
  --checkpoint work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth \
  --dataset test --gpu-id 0
```

### C7: Shift 消融验证
```bash
# Linear (shift=0)
python experiments/runners/test.py \
  experiments/configs/ablation_supplement/a1_rf_heun_unshifted_24obj.py \
  --checkpoint work_dirs/ablation_supplement/a1_rf_heun_unshifted_24obj/best_coco_bbox_mAP_epoch_89.pth \
  --dataset val --gpu-id 0

# Shifted (shift=3.0) — 使用 A1 主实验
python experiments/runners/test.py \
  experiments/configs/ldmdet/directions/mainline_ablation_24obj/a1_rf_heun_24obj.py \
  --checkpoint work_dirs/a1_rf_heun_24obj/best_coco_bbox_mAP_epoch_62.pth \
  --dataset val --gpu-id 0
```

---

## 七、风险更新

### ~~风险 1: Dataset 1 SOTA 数值不存在~~ [✅ 已解决]
三个数值均有实验依据（PAPER_RESULTS.md + EXPERIMENT_CATALOG.md 确认）。

### ~~风险 2: 耦合多种子实验数据集归属不明~~ [✅ 已解决]
确认 hard_ot/rf_heun_adaln/stochot_eps5_old 均在 Dataset 1（Chromosome20240904）上训练。
config 中 `data_root = 'data/Chromosome20240904_NoAug_NoResize_coco/'` 已验证。

### 风险 3: per-image AP 输出 [⚠️ 中风险]
test.py 当前不输出 per-image AP，C3 统计检验需要此数据。
**解决方案**：修改 test.py 添加 per-image AP 输出，或使用 mmdet COCO evaluator 的 evalImgs 接口。

### 风险 4: A0 baseline 可能未收敛 [⚠️ 低风险]

A0 baseline best_ep12 mAP 未提取，且 ep12 可能过早停止。
但 A0 在论文中仅作为 A0→A1 对比的起点，不影响主要结论。

---

## 八、执行计划

### 阶段 4a: 即时可执行（C1 + C4 + C7）
1. C1: A4 多种子推理（3 次，~15min）
2. C4: 测试集评估（1 次，~5min）
3. C7: Shift 消融验证（2 次，~10min）

### 阶段 4b: 需要脚本修改（C3）
1. 修改 test.py 或编写独立脚本收集 per-image AP
2. 运行统计检验

### 阶段 4c: 大批量推理（C2）
1. 耦合多种子推理（9 次，~45min）

### 阶段 4d: 论文更新
1. 根据推理结果更新 SOTA 表（多种子 mean ± std）
2. 添加统计检验 p 值
3. 更新耦合消融表

---

## 九、待确认问题

1. ~~C5 风险~~ ✅ 已解决（数值均有依据）
2. ~~C2 数据集归属~~ ✅ 已解决（确认在 Dataset 1 上）
3. **C3 统计检验方法**：使用 paired t-test 还是 Wilcoxon signed-rank test？（审稿人未指定，建议 Wilcoxon 因无需正态假设）
4. ~~执行环境~~ ✅ 已确认（本地 A6000，GPU 空闲）
5. **是否现在开始执行 C1/C4/C7**：本地 GPU 空闲，可立即开始即时代执行的任务

---

## 十、执行进度（2026-07-18 实时更新）

### C1: A4 多种子推理 ✅ 完成

A4 (DPM-Solver++ 4-step, Dataset 2 val, 推理 seed=42) 三种子结果：

| Seed | mAP | AP50 | AP75 | AP_S | AP_M | AP_L |
|------|------|------|------|------|------|------|
| 42 | 0.863 | 0.989 | 0.972 | 0.499 | 0.860 | 0.901 |
| 123 | 0.857 | 0.988 | 0.969 | 0.521 | 0.855 | 0.901 |
| 789 | 0.856 | 0.988 | 0.964 | 0.527 | 0.853 | 0.868 |
| **mean** | **0.859** | **0.988** | **0.968** | **0.516** | **0.856** | **0.890** |
| **std** | **0.003** | **0.0005** | **0.003** | **0.012** | **0.003** | **0.016** |

### ⚠️ 关键发现：论文 SOTA 表 A3/A4 标签串列（数据完整性问题）

两个子代理独立验证（scalars.json + swanlab_export.json）确认：

| 论文行 | 论文值 (mAP/AP_S) | 真实 best-mAP 记录 (mAP/AP_S) | swanlab 独立 max | 问题 |
|--------|-------------------|-------------------------------|------------------|------|
| "A3" | 0.863 / 0.583 | A3: 0.858 / 0.558 (step 114) | A3: 0.858 / 0.586 | **数值来自 A4，标签串列** |
| "A2 Heun" | 0.858 / 0.565 | A2: 0.856 / 0.502 (step 82) | A2: 0.856 / 0.565 | **mAP 错误，AP_S 取自 step 48 (mAP=0.846)** |
| A4 | 0.863 / 0.574 | A4: 0.863 / 0.547 (step 117) | A4: 0.863 / 0.583 | **AP_S cherry-picked 自 step 132** |

详细：
- A3 best mAP=0.858@ep114（对应 AP_S=0.558），全程 AP_S 最高=0.586@ep59（mAP 仅 0.848）
- A4 best mAP=0.863@ep117（对应 AP_S=0.547），但 step 132 也是 mAP=0.863（AP_S=0.574），论文选了 step 132
- A2 best mAP=0.856@ep82（对应 AP_S=0.502），论文 AP_S=0.565 来自 step 48（mAP 仅 0.846）

### 独立推理结果汇总（val, seed42）

| 模型 | 独立推理 mAP | 独立推理 AP_S | 独立推理 Y AP | scalars best-mAP AP_S | 论文声称 AP_S |
|------|-------------|---------------|---------------|------------------------|---------------|
| A2 (Heun) | 0.857 | 0.502 | 0.769 | 0.502 | 0.565 |
| A3 (Heun+StochOT) | 0.857 | 0.523 | 0.778 | 0.558 | 0.583 |
| A4 (DPM-Solver++) seed42 | 0.864 (dump) / 0.863 (C1) | 0.491 (dump) / 0.499 (C1) | 0.779 | 0.547 | 0.574 |
| A4 三种子 mean±std | 0.859±0.003 | 0.516±0.012 | — | — | — |

AP_S 极不稳定：同一模型 (A4 seed42) 多次推理 AP_S 范围 0.491-0.547-0.583，跨种子 0.499-0.527。

### C4: 测试集评估 ✅ 完成（重跑）

- 使用 `a4_test_eval_24obj.py`（修复 config bug）
- A4 seed42, Dataset 2 test: **mAP=0.859, AP_S=0.577**

### C7: Shift 消融推理 ✅ 完成

| 配置 | mAP | AP_S | AP_M | AP_L |
|------|------|------|------|------|
| C7-1 Linear (shift=0) | 0.857 | 0.528 | 0.853 | 0.907 |
| C7-2 Shifted (shift=3.0) | 0.856 | 0.542 | 0.853 | 0.897 |
| 差异 | -0.001 | +0.014 | 0.000 | -0.010 |

结论：shift 消融对 mAP 影响可忽略 (-0.001)，符合论文声明。

### C3: 统计显著性检验 ✅ 完成（500 images, val, seed42）

#### mAP@0.5:0.95

| 对比 | Δ均值 | Wilcoxon p | paired t p | 显著性 |
|------|-------|-----------|-----------|--------|
| A3 vs A2 (StochOT 效应) | +0.0001 | 7.97e-01 | 9.44e-01 | 不显著 |
| A4 vs A3 (DPM-Solver++ 效应) | +0.0056 | **2.54e-07** | **8.44e-07** | ***显著*** |
| A4 vs A2 (联合效应) | +0.0057 | **4.53e-04** | **4.88e-05** | ***显著*** |

#### ⚠️ 关键矛盾：DPM-Solver++ 显著提升 mAP

- 论文 L161: "with purely computational advantage and no precision gain"
- 论文 L162: "The +0.005 mAP gap in the main table stems from checkpoint selection, not solver precision"
- **但 C3 检验显示**: A4 vs A3 (4-step vs 4-step) Δ=+0.0056 mAP, Wilcoxon p=2.54e-07 (高度显著)
- **不能归结为 checkpoint selection** — 这是 500 张图的配对检验

#### AP_S (60 images with small objects)

| 对比 | Δ均值 | Wilcoxon p | 显著性 |
|------|-------|-----------|--------|
| A3 vs A2 | +0.0012 | 7.83e-01 | 不显著 |
| A4 vs A3 | -0.0031 | 8.55e-01 | 不显著 (A4 略低) |
| A4 vs A2 | -0.0019 | 6.91e-01 | 不显著 |

结论：AP_S 差异均不显著，"小目标优势"叙事缺乏统计支撑。

### test.py per-image AP 功能 ✅ 已完成

- 已添加 `--dump-per-image PATH` 选项
- 输出 JSON: `[{image_id, image_path, AP, AP50, AP75, AP_S, AP_M, AP_L}, ...]`
- 已生成 3 个 per-image AP 文件: a2/a3/a4 (各 500 条)

### 用户决策记录（2026-07-18）

1. **C3 统计检验方法**：Wilcoxon signed-rank + paired t-test 两者都报告
2. **附加约束**："显著性实验都做，但若有一个结果不好则仅记录在草稿中，声明不放进正文"
3. **执行顺序**：立即开始 C1+C4+C7（即可执行的推理任务）

### GPU 状态（2026-07-18 11:05）

- ⚠️ **A1 multi-seed 训练 (seed 123) 正在运行**, ETA ~28 小时 (1 天 4 小时 42 分)
  - 进程 PID 2862623, 从 04:47 开始, epoch 24/150
  - 配置: `a1_rf_heun_24obj_multiseed.py`, work_dir: `multi_seed/a1_rf_heun_24obj_multiseed_20260718_044730/seed_123`
  - GPU 利用率 100%, 显存 12929 MiB
- C2 (9 次推理) 无法在此 GPU 上运行，需等待训练完成或使用远程 GPU

### 待执行任务

| 任务 | 状态 | 依赖 | 预计耗时 |
|------|------|------|---------|
| ~~C4 重跑~~ | ✅ 完成 | — | — |
| ~~A2/A3 补充推理~~ | ✅ 完成 | — | — |
| ~~C3 统计检验~~ | ✅ 完成 | — | — |
| ~~C7 Shift 消融~~ | ✅ 完成 | — | — |
| ~~论文更新 (SOTA表+DPM++声明)~~ | ✅ 完成 | 用户决策 | — |
| ~~C6 Per-class 多种子~~ | ✅ 完成 | C1 logs | — |
| ~~C2 耦合多种子~~ | ✅ 完成 | 注册 hard_ot + AsyncCheckpointHook | 25min |

### ✅ 用户决策记录（2026-07-18，已全部解决）

1. **论文 SOTA 表 A3/A4 标签串列** → **决策: 用多种子独立推理**
   - A4 (DPM++) = 0.859±0.003 (3-seed mean), A2 (Heun+StochOT) = 0.857, A3 (Heun) = 0.857
   - 已更新 main.tex 主消融表 (tab:main-ablation) 和 SOTA 表 (tab:sota)

2. **DPM-Solver++ "no precision gain" 声明矛盾** → **决策: 修正声明，承认精度优势**
   - C3 Wilcoxon p=2.54e-07 (A4 vs A3, mAP +0.0056) 高度显著
   - 已修正 abstract/contributions/novelty boundary/SOTA text/conclusion/limitations 全部声明
   - 已添加 tab:stat-tests 统计检验表

3. **C2 耦合多种子推理 GPU** → **决策: 通过 ssh 查看 workstation GPU**
   - workstation (100.99.131.26) SSH 超时不可达, txyun SSH 权限拒绝
   - 本地 GPU 与 A1 训练并行运行 C2 (A6000 49GB, A1 仅占 ~6GB, 推理 ~1GB)

### C2: 耦合多种子推理 ✅ 完成（2026-07-18 12:53）

#### 阻塞修复
- **hard_ot 耦合未注册**: 在 `ldmdet/coupling/ot_flow_coupling.py` 添加 `HardOTCoupling` (OTFlowCoupling 子类, epsilon=1e-3, coupling_mode='argmax'), 注册为 `hard_ot`
- **AsyncCheckpointHook 未注册**: 在 `experiments/mmdet_bridge/hooks.py` 添加 `AsyncCheckpointHook` 作为 mmengine `CheckpointHook` 别名 (推理路径足够, 训练路径退化为同步)
- 更新 `experiments/mmdet_bridge/__init__.py` 导出 `AsyncCheckpointHook`

#### COCO 聚合 mAP (3-seed mean ± std, Dataset 1 val, 440 images, 推理 seed=42)

| 策略 | mAP | AP50 | AP75 | AP_S | AP_M | AP_L |
|------|------|------|------|------|------|------|
| Hard OT | 0.705 ± 0.002 | 0.896 | 0.793 | 0.415 | 0.701 | 0.552 |
| Random | 0.713 ± 0.005 | 0.907 | 0.800 | 0.429 | 0.710 | 0.594 |
| **StochOT ε=5** | **0.747 ± 0.003** | **0.942** | **0.834** | **0.511** | **0.740** | **0.631** |

#### 配对统计检验 (pooled across 3 seeds, n=1320)

| 对比 | Δ mAP | Wilcoxon p | 显著性 |
|------|-------|-----------|--------|
| StochOT − Random | +0.0308 | 9.0e-126 | *** |
| Hard OT − Random | -0.0061 | 1.2e-8 | *** |
| StochOT − Hard OT | +0.0369 | 9.0e-155 | *** |

#### ⚠️ 关键发现：StochOT mAP 增益是数据集依赖的

| 数据集 | StochOT vs Random Δ mAP | Wilcoxon p | 显著性 |
|--------|-------------------------|-----------|--------|
| Dataset 1 (1540 images) | +0.034 | <10⁻¹²⁰ | *** 高度显著 |
| Dataset 2 (5000 images) | +0.0001 | 0.80 | 不显著 |

- 论文 L749-752 原错误声明 "all coupling methods produce statistically equivalent mAP (within ±0.002)" — **基于错误的附录表数据** (Hard OT mAP=0.747, 实为 StochOT 值串列)
- C2 修正: StochOT 在小数据集 (Dataset 1) 上有 +0.034 mAP 增益 (p<10⁻¹²⁰), 在大数据集 (Dataset 2) 上消失 (+0.0001, p=0.80)
- 这一发现支持论文核心论点: StochOT 在低数据 regime 最有价值, 正是医学影像场景

#### Hard OT vs Random: 多样性坍缩理论验证

- Hard OT (确定性 OT) 比 Random 更差 (-0.008 mAP, p<10⁻⁸)
- 验证了 Section method-ot 的理论预测: 确定性 OT 分配导致 H(V|X_t)→0 (多样性坍缩), 损害训练
- 这是论文理论分析的有力实证支持

### 论文 C2 相关修订（2026-07-18, LaTeX 编译通过, 13 页）

1. **附录表 tab:per-seed-coupling**: 修正错误数据 (Hard OT 0.747→0.705), 添加 Random 行, 添加 mean±std 汇总行
2. **Coupling Ablation 段落 (L749-765)**: "statistically equivalent" → "large and highly significant gain (+0.034, p<10⁻¹²⁰)"
3. **新增表 tab:stat-tests-coupling**: 6 行 (3 对比 × 2 指标), Dataset 1 pooled 检验
4. **主消融讨论 (L600-610)**: 添加 Dataset 1 vs Dataset 2 对比说明
5. **统计显著性段落 (L707-724)**: 澄清 StochOT 增益的 dataset-specific 性质
6. **Abstract (L85-90)**: "mAP gain is marginal" → "large, highly significant gain in low-data regime (+0.034, p<10⁻¹²⁰)"
7. **Cross-Dataset Summary (L917-926)**: 修正 "Hard OT ≈ Random on Dataset 1" (实际 Hard OT 显著更差)
8. **Analysis subsection (L937-957)**: 标题 "Smoothness, Not mAP" → "Dataset-Dependent mAP Gain Plus Smoothness"
9. **Conclusion (L1012-1024)**: 添加 dual role 描述 (低数据 mAP 增益 + 大数据稳定性)
10. **Limitations (L1028-1049)**: "statistically zero" → "strongly dataset-dependent"

### 论文修订完成记录（2026-07-18）

**main.tex 修订清单** (LaTeX 编译通过, 12 页):

1. **Abstract (L89-93)**: "two-step... purely computational rather than a precision advantage" → "four-step... small but statistically significant precision advantage (Wilcoxon p<0.001)"

2. **Contributions (L160-167)**: "two-step... purely computational advantage and no precision gain" → "four-step... statistically significant precision advantage (+0.006 per-image mAP, Wilcoxon p<0.001)"

3. **Novelty boundary (L169-186)**: "higher-order solvers perform no better" → "DPM-Solver++ is slightly but significantly better, 1.71x faster in NFE"

4. **Main ablation table (tab:main-ablation)**: 全部 mAP/AP_S 更新为独立推理值, A3 报告 3-seed mean±std

5. **Discussion text (L596-602)**: StochOT "+0.002" → "no measurable gain (Wilcoxon p=0.80)"; DPM++ → "small but significant precision advantage (+0.006, p<0.001)"

6. **SOTA table (tab:sota)**: 全部值更新, caption 从 "leading on small objects" → "AP_S differences not statistically significant"

7. **SOTA text (L617-625)**: "matches RTMDet-L at 0.863" → "trails RTMDet-L by only -0.004, DINO by -0.010"

8. **Small-object paragraph (L657-674)**: 移除虚假 AP_S=0.583 声明, 改为 "statistically indistinguishable... competitive"

9. **Per-class AP (L676-693)**: Y AP 0.776→0.779 (独立推理), 移除 AP_S=0.577

10. **Statistical tests table (tab:stat-tests)**: 新增表, 6 行 (3 对比 × 2 指标), Wilcoxon + paired t-test

11. **Solver analysis (L793-814)**: 添加 matched-step 精度优势说明; test set AP_S 从 "drops -0.059" 改为 "swings 0.499→0.577 (high-variance)"

12. **FPS table caption (L827)**: 添加 "seed 42" 和 latency 测量说明

13. **Cross-dataset summary (L865-871)**: "DPM-Solver++ matches Heun" → "matches at lower NFE and slightly but significantly more accurate at matched step count"

14. **Conclusion (L945-951)**: "two-step... purely computational" → "four-step... both computational and a small but statistically significant precision gain"

15. **Limitations (L979-980)**: StochOT "+0.002" → "statistically zero (+0.0001, Wilcoxon p=0.80)"

16. **Appendix Y analysis (L1246-1248)**: Y AP 0.776→0.779 (seed42 独立推理)

17. **Per-class AP (C6 更新)**: 添加 Y 染色体 3-seed 稳定性 (0.771±0.006), C-group spread 跨种子稳定性 (0.027-0.032)

18. **Appendix Y (C6 更新)**: Y AP 添加 3-seed mean (0.771±0.006)

### C6: Per-class AP 多种子稳定性 ✅ 完成

**方法**: 从 C1 推理日志提取 3 个训练种子的 24 类 per-class mAP@0.5:0.95, 计算 mean ± std

**关键结果**:

| 统计量 | 值 |
|--------|---|
| 全部 24 类 mean per-class std | 0.0035 |
| 最稳定类 | C9 (std=0.0009), D14 (std=0.0009) |
| 最不稳定类 | Y (std=0.0060), G21 (std=0.0058), X (std=0.0058) |
| Y AP 3-seed mean ± std | 0.771 ± 0.006 (范围 0.765-0.779) |
| C-group spread 跨种子 | 0.027-0.032 (论文报告 0.029, 在范围内) |

**Size group 3-seed 均值**:

| 组 | 类数 | 3-seed Mean ± Std |
|----|------|-------------------|
| Large | 5 | 0.904 ± 0.004 |
| Medium | 11 | 0.874 ± 0.003 |
| Small | 8 | 0.811 ± 0.004 |

**结论**: Per-class AP 跨种子稳定 (mean std=0.0035), Y 染色体方差最大但仍可接受 (±0.006)。论文已更新添加 3-seed Y AP 和 C-group spread 稳定性说明。

**输出**: `/tmp/c_class_results/c6_perclass_stability.{md,json}`

---

### 文档归档完成（2026-07-18）

**所有 C 类任务已完成并归档**：

| 任务 | 状态 | 归档位置 |
|------|------|---------|
| C1: A4 多种子推理 | ✅ 完成 | EXPERIMENT_CATALOG.md §7.2 |
| C2: 耦合多种子推理 | ✅ 完成 | EXPERIMENT_CATALOG.md §7.3 |
| C3: 统计显著性检验 | ✅ 完成 | EXPERIMENT_CATALOG.md §7.4 |
| C4: 测试集评估 | ✅ 完成 | EXPERIMENT_CATALOG.md §7.5 |
| C5: Dataset 1 SOTA 核实 | ✅ 已解决 | EXPERIMENT_CATALOG.md §7.1 |
| C6: Per-class 多种子 | ✅ 完成 | EXPERIMENT_CATALOG.md §7.8 |
| C7: Shift 消融验证 | ✅ 完成 | EXPERIMENT_CATALOG.md §7.6 |
| C8: FPS 复测 | ⏸️ 跳过 | EXPERIMENT_CATALOG.md §7.7（决策记录） |

**归档操作**：
1. 合并 `PAPER_RESULTS.md` + `EXPERIMENT_RESULTS.md` 内容到 `EXPERIMENT_CATALOG.md` §七（全部内容已在 CATALOG 中更详细存在，跳过）
2. 独立校验 subagent 验证数据一致性（9 PASS / 2 FAIL → 修复后全部 PASS）
3. 修复 2 项数据缺失（C4 AP_S=0.577 / C7 AP_S/AP_M/AP_L 列）+ 1 项建议（§7.3.2b COCO 聚合 mAP 表）
4. 删除冗余文档 `PAPER_RESULTS.md`、`EXPERIMENT_RESULTS.md`
5. 保留备份 `EXPERIMENT_CATALOG.md.bak.20260718`

**关键文件**：
- 最终存档: `/home/linkst/workspace/projects/chromosome-kd/docs/EXPERIMENT_CATALOG.md`（973 行）
- 校验报告: `/tmp/c_class_results/verification_report.md`（480 行）
- 合并日志: `/tmp/c_class_results/merge_log.md`（201 行）


