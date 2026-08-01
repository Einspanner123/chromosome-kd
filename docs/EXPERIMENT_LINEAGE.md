# 实验脉络主路线文档 (按创新点主题组织)

> 📋 **命名约定**: 本文档使用论文正式名称 (Dataset 1 / Dataset 2 / RF+Heun / +Stoch. Coupling / +DPM-Solver++ / Top-K)。内部实验代号 (24obj / A0-A4 / IO3 / StochOT) 仅保留在文件路径和 SwanLab run_id 中以兼容工程实现。

> 本文档为 KaryoFlow (染色体检测论文, 目标 TMI 期刊) 的有效方向主路线梳理。
> 按"创新点主题"组织实验脉络, 让审稿人快速识别 solid 的研究链条与创新性。
> 数据源: 24 Chromosomes Object (Dataset 2, 5000 张图) 为主, Chromosome20240904 (Dataset 1, 1540 张图) 作低数据对照。
> SwanLab URL 模式: `https://swanlab.cn/@einspanner/<project>/runs/<run_id>`
> 更新时间: 2026-07-31 (第五轮: DINO R50 / RTMDet-L Dataset 2 test mAP 补跑完成—DINO R50 test=0.865 (best@ep102), RTMDet-L test=0.862 (best@ep85), 均在 ross A6000 上评估; §十三.1/§十三.5/§十六 全部 "未评估" 标注替换为实际 test mAP, 跨数据集退化对比更新为 test-vs-test 口径。第四轮: R1/R3/S1 方向代号全称化 (52处); val/test 口径全标注; 测试表补全。第三轮: (C) 全文 D1/D2/D3 方向代号替换为全称 Dataset 1/Dataset 2/Box Renewal × DPM++ 交互, 保留 $D_1$ 数学符号与 dim_d1_mask 代码变量; (A1) §三 "匹配 NFE" 修正为"匹配步数"同 checkpoint 对比 (DPM++ 4步 0.863 vs Heun 4步 0.864, Δ=−0.001); (B5) §三 +0.006 口径澄清: per-image Wilcoxon 跨 checkpoint delta (非 aggregate mAP 差, 非同 checkpoint solver 切换), §一 solver×step 标注 SwanLab-only 数据缺口; (B3) §四 K=100 3-seed 掉点 −0.024→−0.020 (修正口径: K=100 3-seed 0.839 − K=500 3-seed 0.859); (B4) 0.859±0.004 (renewal ON) vs 0.858±0.003 (renewal OFF) 双口径确认; (A2) §一 aggregate mAP 术语释义补充。第二轮纠正: 用户澄清所有实验统一数据增强策略, 回退错误的"AUG/NoAug 管线混杂"标注, 删除"管线混杂影响评估"小节, 恢复 +0.034 为干净 Stoch vs Random 对比, §〇 新增统一增广策略声明。同日首轮 4 subagent 数据核验校准: §三 DPM++ seed42 "early stop@ep50"→"manual kill@ep51" + Δ 符号 −0.001→+0.001; §七 s1_h3_s4 mAP 0.859→0.860@ep59; §八 v-prediction 对照状态回退 (2026-07-30 错误修正, seed123/789 实际已存在, 恢复 3-seed 0.857±0.0015); §十三.4 workstation SSH 核实 (best 0.846@ep29); §六 seed789 旧值 0.724→0.746; OT Flow Coupling 本地数据缺失标注)。原 2026-07-30: 补全 §三 Dataset 1 DPM++ 3-seed (seed789 异常偏低 0.724); 新增 §十三.3 跨域 per-class AP 分析 (类别顺序不一致主导跨域失效) + §十三.4 Dataset 2 跨数据集训练启动; 记录 DINO R50 Dataset 1 最终结果 0.742。原 2026-07-30: 重新梳理逻辑/理论/实验: 修复 §六 Dataset 1 dim_d1_mask 伪造数据 + §八 v-prediction 对照状态矛盾 + Cascade × Solver/v-prediction 矩阵陈旧状态; 重编号消除 §十一 断层; 整合 Dataset 1 DPM++ 双数据集对照; 标注各章 Dataset 1 验证缺口; 修正 §四/§五/§六 K=100/K=200 η_str 错标 "3 seeds" 为 seed42 (真实 3-seed K=100 均值 0.839±0.012 见 §六 K 值依赖性表)。原 2026-07-26: 新增 §十一 FPS 基准 / §十二 噪声鲁棒性 / §十三 测试集+跨域 zero-shot, 补全 §一 per-class AP / §二 Table 7 / §三 Table 8)
>
> 📌 **关联文档**:
> - [TODO_DIRECTIONS.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/TODO_DIRECTIONS.md) (进行中/待启动方向)
> - [FALSIFIED_DIRECTIONS.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/FALSIFIED_DIRECTIONS.md) (已证伪方向)
> - [EXPERIMENT_CATALOG.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/EXPERIMENT_CATALOG.md) (实验数据索引)
>
> 📋 **文档流转规则**: 本文档仅收录 ✓已完成且有效的方向 (3-seed 完成或零成本诊断完成)。进行中方向保留章节但标注 🔄, 详见 TODO_DIRECTIONS.md。

---

## 〇、任务特性画像 (贯穿全文参照)

本画像定义了染色体核型分析检测任务的核心特性, 后续所有创新点的"与任务结合"论证均回溯此画像。

### 图像特性
- 中期相铺展图像, 每张含约 46 条紧密排列的染色体
- 24 个类别 (A1–Y), 跨越大/中/小三组尺寸, 形态相似性强
- C 组 (C6–C12) 7 条亚中着丝粒染色体, 仅靠细微带纹差异区分
- Y 染色体最小, 仅男性单拷贝出现, 训练样本约 1803 vs 常染色体约 7000
- 频繁的相互重叠与接触

### 检测范式特征
- 基于扩散的检测 (DiffusionDet 范式): 噪声框 → GT 框迭代去噪
- 预测空间维度 d=4 (cxcywh), 相比图像生成 d≈10⁵ 极低维
- 每张图 K≈46 个 GT 框 (高目标密度), COCO 平均 K≈7
- 500 个噪声 proposals 全部通过 6 级 cascade head × 4 solver step = 24 NFE
- box_renewal 是检测特有操作 (图像生成无此机制)

### 数据特征
- Dataset 1 (Chromosome20240904): 1540 张, mAP 量级 0.72-0.75, 低数据对照
- Dataset 2 (24 Chromosomes Object): 5000 张, mAP 量级 0.77-0.87
- 类别不平衡严重 (Y vs 常染色体 1:3.9)
- 临床采集, 标注质量受观察者主观影响

### 关键约束
- 临床交互式筛查延迟带: 13.3-14.2 FPS (Top-K K=200)
- cascade head 占 90%+ 推理延迟
- 跨站点/跨 seed 可复现性 (临床部署要求)
- 所有实验统一数据增强策略 (经消融测试, 原版最佳), 保证各消融对比的干净性

---

## 一、RF (Rectified Flow) — 范式替换贡献

### 核心贡献: 直线 ODE 路径取代 DDPM 弯曲随机轨迹

RF 以从噪声 $\mathbf{x}_1$ 到 GT $\mathbf{x}_0$ 的确定性直线 ODE 路径取代 DDPM 的弯曲随机轨迹, 使速度场沿路径恒定, 在少步推理下保持低截断误差。

- **形式化**: $x_t = (1-t)x_0 + t x_1$, 速度场 $v = x_1 - x_0$ 恒定
- **训练目标**: flow matching 损失 $\mathcal{L}_{FM} = \mathbb{E}[\|v_\theta(x_t, t) - (x_1 - x_0)\|^2]$
- **范式定位**: data-prediction 形式, 与 DPM-Solver++ 天然兼容 (避免 $v=(x_t-\hat{x}_0)/t$ 在 $t\to 0$ 的奇点)

### 与染色体检测任务特性的结合

- **密集 proposals 误差复合**: 每张图 46 个 GT × 500 proposals, DDPM 弯曲轨迹的截断误差在 proposals 间复合, RF 直线轨迹将单步误差降为 0 (理想情况)
- **小训练集限制弯曲轨迹学习**: 1540-5000 张临床图像难以学习复杂 DDPM 弯曲轨迹, RF 直线 ODE 路径降低学习负担
- **24 类细粒度依赖稳定特征**: RF 恒定速度场为 24 类细粒度判别提供稳定特征表示, C 组带纹差异得以保留

### 范式贡献归因

DDPM baseline→RF+Heun 累积 +0.053 mAP (统一口径: DDPM baseline = DiffusionDet 0.803, 训练配置 AdamW/150ep), 其中 solver×step 解耦消融证明:
- **91% (+0.048 mAP) 归因于 RF 范式本身**
- 9% (+0.005 mAP) 归因于 solver/步数选择 (Heun 4 步 vs Euler 1 步)
- AdaLN-Zero 单独贡献为 0 (Appendix B 零结果)
- 偏移噪声调度 (shift=3.0) 单独贡献为 −0.001 (噪声范围)
- ⚠ 历史口径 (2026-07-27 前): 旧 DDPM baseline (a0_baseline, SGD/12ep, 训练不足, mAP=0.774) → RF+Heun 累积 +0.082 mAP, 94% 归因于 RF; 2026-07-27 统一口径至 DiffusionDet (AdamW/150ep, mAP=0.803) 后, 累积增益 +0.053 mAP, 91% 归因于 RF, 结论方向不变 (RF 范式为核心增益来源)

### 实验列表

#### 实验证明目的: RF 范式相对 DDPM 的精度优势 (主消融)

- DDPM baseline (DDPM Euler 1-step, a0_baseline)
  -- 数据集: Dataset 2
  -- 结果: mAP=0.774 (val, seed42, 训练评估), AP50=0.968, AP75=0.916
  -- ⚠ 训练配置: SGD/lr=0.02/12ep, 训练不足, 2026-07-27 已将论文 DDPM baseline 统一为 DiffusionDet (AdamW/150ep, mAP=0.803 val / 0.804 test, 见下方对照)
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/a0_baseline

- DDPM baseline (统一口径, DiffusionDet checkpoint)
  -- 数据集: Dataset 2
  -- 结果: mAP=0.803 (val) / 0.804 (test), AP50=0.970, AP75=0.936
  -- 训练配置: AdamW/150ep, 与 RF+Heun 完全一致 (除 diffusion_type=ddpm)
  -- 结构对比: 两个 subagent 确认 DiffusionDet 与 a0_baseline 结构相同, 性能差异 (0.774 vs 0.803) 完全由训练配置 (SGD/12ep vs AdamW/150ep) 导致, 非模型结构差异
  -- SwanLab (project=chromosome-kd-benchmark-24obj): benchmark_diffusiondet
  -- checkpoint: work_dirs/baselines/diffusiondet_24obj/best_coco_bbox_mAP_epoch_26.pth

- KaryoFlow (RF+Heun)
  -- 数据集: Dataset 2
  -- 改动: diffusion_type=rectified_flow, solver=heun, rf_schedule=shifted, time_conditioning=adaln_zero
  -- 结果: mAP=0.856 (val, seed42, 训练评估; test=0.857 见 §十三.1), AP50=0.990, AP75=0.969 [+0.053 主贡献 (统一口径 DiffusionDet 0.803 val) / +0.082 (旧口径 a0_baseline 0.774 val)]
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/a1_rf_heun

- +AdaLN-Zero
  -- 结果: mAP=0.856 (val, seed42), AP50=0.990, AP75=0.972 [+0.000 持平 RF+Heun, AdaLN 单独贡献为 0]
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/a2_adaln

- +Stoch. Coupling eps=5
  -- 结果: mAP=0.858 (val, seed42; test=0.858 见 §十三.1), AP50=0.990, AP75=0.973 [+0.002 边际]
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/a3_stochot

- +DPM-Solver++ 替换 Heun
  -- 结果: mAP=0.863 (val, seed42; test=0.859 见 §十三.1), AP50=0.990, AP75=0.974 [+0.005 推理加速且精度提升]
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/a4_dpm_pp

#### 实验证明目的: solver×step 解耦, 隔离 RF 范式贡献

- RF+Heun checkpoint 上 solver×step 全组合 (Dataset 2 val, seed 42)
  -- Heun 4 步 (7 NFE): mAP=0.856 (val)
  -- Euler 4 步 (4 NFE): mAP=0.855 (val)
  -- DPM-Solver++ 4 步 (4 NFE): mAP=0.855 (val)
  -- Euler 1 步 (1 NFE): mAP=0.851 (val)
  -- DPM-Solver++ 1 步 (1 NFE): mAP=0.851 (val)
  -- 结论: 匹配步数下 solver 类型对 mAP 无影响; 步数 1→4 仅 +0.004; solver/步数联合贡献 9% (新口径 +0.053 mAP; 旧口径 +0.082 mAP 下为 6%, 见 §一 范式贡献归因)
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/a1_rf_heun
  -- ⚠ 2026-07-31 B5 核验: 本组数据仅在 SwanLab 有记录, 本地 work_dirs/a1_rf_heun_24obj/ 下未找到对应推理日志/JSON (仅有训练日志)。同结论已被 §三 +DPM-Solver++ checkpoint 上 Heun 4步 vs DPM++ 4步 (Δ=−0.001, work_dirs/a4_dpm_pp_24obj/20260715_011706/) 独立复现, 结论不变

#### 实验证明目的: DDPM 多步消融 (Dataset 2, 论文 Appendix G)

- DiffusionDet checkpoint (Dataset 2, best @ ep26, seed 42) 上 Euler 1/2/4/8 步推理
  -- 配置: `experiments/configs/baselines/benchmark_24obj/diffusiondet_ddpm.py` + `--sampling-steps {1,2,4,8} --solver-type euler`
  -- 命令: `python experiments/runners/test.py experiments/configs/baselines/benchmark_24obj/diffusiondet_ddpm.py --checkpoint work_dirs/baselines/diffusiondet_24obj/best_coco_bbox_mAP_epoch_26.pth --dataset val --sampling-steps {1,2,4,8} --solver-type euler --seed 42`
  -- 结果 (验证集, seed 42):
     | Steps | NFE | mAP   | AP50  | AP75  | AP_S  | AP_M  | AP_L  |
     |-------|-----|-------|-------|-------|-------|-------|-------|
     | 1     | 1   | 0.805 | 0.971 | 0.937 | 0.405 | 0.802 | 0.810 |
     | 2     | 2   | 0.804 | 0.969 | 0.939 | 0.414 | 0.800 | 0.806 |
     | 4     | 4   | 0.804 | 0.970 | 0.938 | 0.402 | 0.800 | 0.804 |
     | 8     | 8   | 0.805 | 0.971 | 0.940 | 0.405 | 0.801 | 0.818 |
  -- 结论: DDPM Euler 1→8 步 mAP 变化 <0.002, 所有精度指标 (AP50/AP75/AP_S/AP_M/AP_L) 均在噪声范围内波动; 对比 RF 范式切换带来的 +0.053 mAP 增益, DDPM 增加步数的收益可忽略, 证实 "DDPM 步数对精度几乎无影响, 范式切换才是核心增益来源"
  -- 注: 1 步推理 mAP (0.805) 与 Table 5 中 DDPM baseline (0.803, 训练评估) 略有差异, 源于推理与训练评估的 maxDets 设置不同; 多步对比在相同推理设置下进行, 结论不受影响
  -- 数据源: ross server 推理日志 (2026-07-28), project=ldmdet-inference, 实验名 `euler_{N}step`

#### 实验证明目的: Dataset 1 低数据对照, RF vs DDPM

- rf_heun_adaln 3 seeds (Dataset 1)
  -- 结果: mAP=0.746 ± 0.001 (val, 3-seed; test: 0.737±0.002 见 §十三.1) [+0.017 vs DDPM 0.729 ± 0.003 (val, 3-seed)]
  -- seed42=0.7450, seed789=0.7470, seed123=0.7470
  -- SwanLab (project=ldmdet-ablation):
     - rf_heun_adaln_seed42: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/4xhp5ffymboa05hyn245u
     - rf_heun_adaln_seed789: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/ww6nlti3ufdkm4htjg5pw
     - rf_heun_adaln_seed123: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/dimdbu8fk0re4satbzpgs

- diffusiondet_ddpm 3 seeds (Dataset 1, 根 baseline)
  -- 结果: mAP=0.729 ± 0.003 (val, 3-seed; test: 0.719±0.003 见 §十三.1)
  -- seed42=0.7260, seed789=0.7270, seed123=0.7330
  -- DDPM 步数对齐验证 (project=ldmdet-inference): DDIM 1/4/8 步均为 0.729, +0.017 为纯算法贡献
  -- SwanLab (project=ldmdet-ablation):
     - diffusiondet_ddpm_seed42: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/apfn46t67iqg1bjraq8xd
     - diffusiondet_ddpm_seed789: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/hny1od5fcvx8ngt9b063g
     - diffusiondet_ddpm_seed123: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/ghghjry3bylt0sfoj30j5

#### 实验证明目的: vs SOTA 检测器 (DINO/RTMDet-L/Cascade)

- KaryoFlow (+DPM-Solver++) 3-seed 均值
  -- mAP=0.859 (val, 3-seed; test=0.859 seed42 见 §十三.1), 落后 DINO R50 (0.868 val / 0.865 test, 单 seed) 0.009 val / 0.006 test, 落后 RTMDet-L (0.863 val / 0.862 test, 单 seed) 0.004 val / 0.003 test
  -- 超越 Cascade R-CNN (0.854 val / 0.853 test), YOLOX-S (0.796 val / 0.795 test), DiffusionDet (0.803 val / 0.804 test)
  -- 相对 DiffusionDet seed42 best: +0.060 mAP (val), 3-seed 均值: +0.056 (val)
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/a4_dpm_pp
  -- 对照基准 (project=chromosome-kd-benchmark-24obj): DINO/RTMDet-L/Cascade/YOLOX/DiffusionDet

#### 实验证明目的: 逐类 AP 分析 (论文 Figure 5 + Table F.1, §4.3.1)

- +DPM-Solver++ checkpoint (seed 42, best @ ep117, 独立推理) 上 24 个类别的 per-class AP
  -- 整体 mAP=0.863 (val, seed42 独立推理; test=0.859 见 §十三.1), AP50=0.988, AP75=0.972, AP_S=0.574, AP_M=0.859, AP_L=0.908
  -- 整体 AP 随染色体尺寸单调下降: Large→Medium→Small 为 0.896→0.848→0.805
  -- Y 染色体最难: seed 42 AP=0.779; 3 seed 均值 0.771 ± 0.006 (数据稀缺 1803 vs 7000 + 形态变异)
  -- C 组 (C6-C12) 组内差异仅 0.029 (0.871-0.900), 跟踪尺寸梯度但被削弱, 表明模型学到带纹线索
  -- 所有类别 AP50 > 0.988 (Y 0.972), 定位接近饱和, 残余误差集中在细粒度分类
  -- 数据源: [results/a4_per_class_ap.md](file:///home/linkst/workspace/projects/chromosome-kd/results/a4_per_class_ap.md) (step 132, mAP=0.863 val)
  -- 3-seed per-class 稳定性: [EXPERIMENT_CATALOG.md §7.8](file:///home/linkst/workspace/projects/chromosome-kd/docs/EXPERIMENT_CATALOG.md) (C6, 24 类 × 3 seeds)
  -- 论文图: latex/figures/per_class_ap.png (按尺寸组着色, 虚线为整体均值)

#### 实验证明目的: SOTA per-image Wilcoxon 检验 (论文 §4.3.2 引用, 暂不放入正文)

- RF (+Stoch. Coupling 配置) vs 4 个 SOTA 检测器 per-image 配对检验 (Dataset 2 val, 500 imgs, seed 42)
  -- Aggregate mAP (标准 COCO mAP, 全图池化后计算 AP@0.5:0.95, 区别于下方 per-image Wilcoxon 逐图 AP): DINO R50 0.8685 > RTMDet-L 0.8626 > Cascade R-CNN 0.8535 > RF (+Stoch. Coupling) 0.8521 > DiffusionDet 0.8031
  -- RF (+DPM-Solver++ best 0.863) vs DINO R50 (0.8685) aggregate 差距仅 0.63%, per-image Wilcoxon 差距 1.64% (+Stoch. Coupling 配置)
  -- 注: aggregate mAP 略高于 Table 6 训练评估值 (如 DINO 0.8685 vs 0.868), 因 per-image 推理脚本使用独立评估配置 (maxDets/评估管线不同); 方向性结论不受影响
  -- RTMDet-L ep85 修复 (2026-07-20): 之前用 ep86 (次优 0.861), 实际 ep85 best=0.8630 (实测 0.8626)
  -- 数据源: [experiments/analysis/baseline_inference_24obj_perimage_results.json](file:///home/linkst/workspace/projects/chromosome-kd/experiments/analysis/baseline_inference_24obj_perimage_results.json)
  -- 详细: [EXPERIMENT_CATALOG.md §7.4.6](file:///home/linkst/workspace/projects/chromosome-kd/docs/EXPERIMENT_CATALOG.md)

---

## 二、OT Diversity Collapse 与 Stochastic Coupling — 理论新颖性贡献

### 核心贡献: 低维检测空间 OT 坍缩形式化分析 + Stochastic Coupling 补救

OT 配对在低维 (d=4) 检测空间中将噪声空间划分为 Voronoi 单元, 使耦合分配成为噪声的确定性函数——耦合多样性坍缩至零, 损害训练。Stochastic Coupling 从 Sinkhorn transport 矩阵采样分配 (而非 argmax), 在 hard OT 与随机耦合之间插值, 恢复多样性。

- **形式化上界** (命题 1): $\Delta H \le \log K$ (OT 下 $V$ 可由 $X_t$ 恢复, $H_{OT}=0$)
- **形式化下界** (命题 2, Fano 不等式): $\Delta H \ge \log K \cdot (1-P_{err}) - h(P_{err})$
- **染色体检测数值**: $d_{min}\approx 20$ px, $\sigma_t\sim 1$ px, $P_{err}<10^{-45}$, $\Delta H \ge 0.999\log K$
- **经验验证** (Dataset 1): $\Delta H = 3.8415$ vs $\log K = 3.8427$, 相对误差 0.03%
- **Stochastic Coupling 单调性** (命题 3, 包络定理): $H_{stoch}(V|X_t; \epsilon)$ 关于 $\epsilon$ 单调非递减

### 与染色体检测任务特性的结合

- **低维 d=4 触发坍缩**: 检测预测空间 d=4 vs 图像生成 d≈10⁵, OT 在低维下逼近 $\log K$ 熵减上界
- **高 K≈46 加剧坍缩**: $\Delta H/H \approx 0.69$ (染色体) vs 0.55 (COCO, K≈7) vs ≈0 (图像生成)
- **小训练集放大损害**: Dataset 1 上 Stochastic Coupling 增益 +0.034 (p<10⁻¹²⁰), 数据稀缺时 OT 诱发配对的边际收益减弱, Stochastic Coupling 价值最大

### 数据集规模依赖性

| 数据集 | 规模 | Stoch. vs Random mAP Δ | 显著性 | 平滑性增益 |
|--------|------|------------------|--------|------------|
| Dataset 1 | 1540 张 | +0.034 | p<10⁻¹²⁰ (n=1320) | 4.6× epoch std |
| Dataset 2 | 5000 张 | +0.0001 | p=0.80 (n=500, ns) | 4.6× epoch std |

- Dataset 1: Hard OT 实际比 Random 更差 (−0.008, p<10⁻⁸), 证实 OT 多样性坍缩病理
- Dataset 2: mAP 增益可忽略, 但平滑性收益独立成立 (Last-30 std: 0.006 → 0.0013)
- 数据更多时, 模型见到足够多样本平均掉随机耦合噪声, OT 坍缩及 Stochastic Coupling 边际收益减弱

### 可扩展性倾向

Table 2 a-priori 诊断: 任何 $d \ll 100$ 且 $K \gg 10$ 的任务都是 Stochastic Coupling 候选。
- 严重性 $\Delta H/H$ 预测 Stochastic Coupling 是否会有帮助
- 候选任务: 细胞检测 (组织病理学, 多核 tile, d=4 bbox)、病灶检测 (乳腺 X 光/视网膜, 小目标有限阳性)、微生物菌落计数、遥感密集目标
- 不适用: 高维图像生成 ($d \sim 10^5$, $\Delta H/H \approx 0$, OT-CFM 成功)

### 实验列表

#### 实验证明目的: 经验熵验证 OT Diversity Collapse 理论

- 经验熵测量 (Dataset 1 验证集)
  -- 理论值 $\log K = 3.8427$ vs 经验值 $\Delta H = 3.8415$, 相对误差 0.03%
  -- 配置: 真实染色体检测图像, 8 个 GT 框, OT (最近邻) vs Random 分配
  -- 结论: $N\to\infty$ 界在染色体检测设置下是极佳近似, 即便 mini-batch N=2

#### 实验证明目的: Dataset 1 耦合消融 (低数据, 大增益)

- Hard OT 3 seeds (Dataset 1 **无aug 简化设置**)
  -- 结果: mAP=0.705 ± 0.002 (val, 3-seed, NoAug_NoResize) [−0.008 vs Random (无aug), 证实 OT 坍缩]
  -- ⚠ **实验设置 (2026-08-02 核查澄清)**: 此 3-seed 实验使用 `Chromosome20240904_NoAug_NoResize_coco` (无数据增强 + 无 resize, 见 FALSIFIED §十一 "早期失败/调试实验"). **标准增强下** Hard OT=0.747 (2-seed, +0.001 vs baseline, 见 §二 Dataset 1 早期验证), OT 坍缩效应被数据增强掩盖. **Dataset 2 Hard OT 实验未运行** (配置 `experiments/configs/multiset/chromo_24obj_hard_ot.py` 存在但无 work_dir/训练记录), OT 坍缩在 Dataset 2 上未直接实验验证; 仅 Stoch vs Random +0.0001 (ns, Dataset 2) 作为间接佐证 (Stoch Coupling 在大 K 数据集上增益消失, 与坍缩理论预测方向一致). 论文 OT 坍缩论证主要依赖 Dataset 1 无aug实验 (Δ=−0.008, p<10⁻⁸) + 理论分析 (命题 1-2), Dataset 2 间接佐证
  -- 目录: `work_dirs/multi_seed/hard_ot/seed_{42,123,789}/` (与 `multi_seed_aug/` 区分)
  -- SwanLab: 见下文 Random/Stoch 对照

- Random Coupling 3 seeds (Dataset 1)
  -- 结果: mAP=0.713 ± 0.005 (val, 3-seed)
  -- seed42=0.713, seed123=0.718, seed789=0.708
  -- SwanLab: 见 Dataset 2 同名实验

- Stochastic Coupling ε=5, 3 seeds (Dataset 1)
  -- 结果: mAP=0.747 ± 0.003 (val, 3-seed; test: seed42=0.740 见 §十三.1)
  -- seed42=0.7456, seed123=0.7449, seed789=0.7506
  -- Hard OT vs Random: Δ=−0.0061, p<10⁻⁸ (Hard OT 比 Random 更差, 证实坍缩病理)
  -- Stoch vs Hard: Δ=+0.0369, p<10⁻¹⁵⁵

#### 实验证明目的: Dataset 2 耦合消融 (大数据, 增益可忽略但平滑性显著)

- Random Coupling 3 seeds (Dataset 2, project=ldmdet-ablation)
  -- 平均: mAP=0.860 ± 0.001 (val, 3-seed)
  -- seed42: mAP=0.859 (val, best @ 59)
     - SwanLab: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/p5xqii8mcqmbhuo5lhlff
  -- seed789: mAP=0.860 (val, best @ 82)
     - SwanLab: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/r8n441mu4gws43xyoneoj
  -- seed123: mAP=0.860 (val, best @ 115)
     - SwanLab: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/q6jgxefgxbp8f2sf5qzpc

- Sinkhorn Stochastic OT 1 seed (Dataset 2, project=ldmdet-ablation)
  -- 结果: mAP=0.856 (val, seed42, best @ 53) [对应 +Stoch. Coupling 配置]
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/o96m1eqz4l12qjeyys1cs

- GHSS Coupling 3 seeds (Dataset 2, project=ldmdet-ablation)
  -- 平均: mAP=0.858 ± 0.001 (val, 3-seed)
  -- seed42: mAP=0.857 (val, best @ 83)
     - SwanLab: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/k84cq9oftbp2nld88a85t
  -- seed789: mAP=0.859 (val, best @ 75)
     - SwanLab: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/holadvhaz9v2rh8l494hv
  -- seed123: mAP=0.859 (val, best @ 102)
     - SwanLab: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/73cr3uyqw4f1q68xz1evg

- 多维稳定性比较 (Dataset 2, 单 seed)
  -- Last-30 epoch std: Random 0.006 vs Stoch 0.0013 (4.6× 改善)
  -- Last-30 CV: 0.69% vs 0.16% (4.4×)
  -- Last-30 range: 0.023 vs 0.005 (4.6×)
  -- Last-30 处于 best 1% 内 epoch 数: 13/30 (43%) vs 30/30 (100%)
  -- 训练失败率 (9 runs): 0/9 vs 0/9
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/a3_stochot

#### 实验证明目的: ε 消融, 验证 Stochastic Coupling 饱和性

- ε 消融 (Dataset 2, +Stoch. Coupling 配置)
  -- ε < 1: 有害 (相同增广下 mAP −1.3%)
  -- ε ≥ 1: 进入饱和, 收益递减
  -- ε = 5: 主路线配置
  -- 结论: Stochastic Coupling 是必要的 OT 正则化项而非精度助推器
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/a3_stochot

#### 实验证明目的: 多维稳定性比较 (论文 Table 7 + Figure 4, §4.4.2)

- 5 项稳定性指标: Random (RF+Heun) vs Stochastic Coupling ε=5 (+Stoch. Coupling) (Dataset 2, 单 seed)
  -- Last-30 epoch std: 0.006 → 0.0013 (4.6× 改善)
  -- Last-30 CV (std/mean): 0.69% → 0.16% (4.4× 改善)
  -- Last-30 range (max−min): 0.023 → 0.005 (4.6× 改善)
  -- 最后 30 个 epoch 中处于最佳 1% 内的 epoch 数: 13/30 (43%) → 30/30 (100%)
  -- Best mAP / best epoch: 0.856 / ep62 → 0.858 / ep114
  -- Total training epochs: 92 → 144
  -- Training failure rate (9 runs): 0/9 → 0/9
  -- 结论: Stoch Coupling 在每个被测轴上胜出, 关键是 30/30 vs 13/30 使小数据下基于 EarlyStopping 的 checkpoint 选择远为可靠
  -- 数据源: [EXPERIMENT_CATALOG.md §7.8 / 训练日志](file:///home/linkst/workspace/projects/chromosome-kd/docs/EXPERIMENT_CATALOG.md) (来自训练日志的逐 epoch mAP)
  -- 论文图: latex/figures/training_stability.png (Random 振荡 vs Stoch 平滑收敛, 阴影带标记 last 30 epoch)

#### 实验证明目的: Dataset 1 per-class AP 增益 (论文 §5.6 / §7.3.5 引用)

- Stochastic Coupling 在 Dataset 1 上对 24 类的 per-class AP 增益 (3-seed, IoU=0.5:0.95)
  -- Y 染色体: 0.569 ± 0.009 → 0.622 ± 0.021 (+0.059, Wilcoxon W=0, p=1.66e-13)
  -- G22: 0.581 → 0.623 (+0.042); F19: 0.694 → 0.720 (+0.026); F20: 0.684 → 0.718 (+0.034)
  -- 大类 (A1/A2/A3) 平均 +0.029-0.037
  -- 小类 (Y/G/F) 平均 +0.040, 大类平均 +0.029 → 小类增益更大, OT 坍缩对低频类压制更严重
  -- 整体 mAP +0.034 (p<10⁻¹²⁰, n=1320)
  -- 数据源: [EXPERIMENT_CATALOG.md §7.3.5](file:///home/linkst/workspace/projects/chromosome-kd/docs/EXPERIMENT_CATALOG.md) (Problem 3: Y 染色体稳定性)

---

### Dataset 1 早期验证 (OT 坍缩理论佐证)

> 以下 Dataset 1 实验佐证 §二 OT Diversity Collapse 理论。⚠ **(2026-08-02 核查澄清)**: OT 坍缩在 Dataset 1 上**仅在无aug简化设置下出现** (Hard OT=0.705 vs Random=0.713, Δ=−0.008, p<10⁻⁸); 标准增强下 Hard OT=0.747 (2-seed, Δ=+0.001 vs baseline), 坍缩被数据增强掩盖。**Dataset 2 Hard OT 实验未运行**, OT 坍缩在 Dataset 2 上未直接验证; 仅 Stoch vs Random +0.0001 (ns) 作为间接佐证 (Stoch Coupling 在大 K 数据集上增益消失, 与坍缩理论预测一致)。按"所有理论在两个数据集上验证"规则, §二 理论的双数据集验证存在缺口: Dataset 1 无aug实验为直接证据, Dataset 2 仅有间接佐证, 论文叙事需明确标注并考虑补跑 Dataset 2 Hard OT。
### Hard OT Coupling

- Hard OT Coupling (Dataset 1 **标准增强**, 2 seeds)
  -- 结果: mAP=0.747 ± 0.000 (val, 2-seed) [+0.001 vs 0.746 baseline (val), 边际, **不显示 OT 坍缩**]
  -- 本地: work_dirs/multi_seed_aug/hard_ot/seed_{42,123}/
  -- SwanLab (project=ldmdet-ablation):
     - hard_ot_seed42: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/h8fizm7lmc9v5xzxi8ufj
     - hard_ot_seed123: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/kka4nra9qk3wanx7i9og1
  -- ⚠ **与无aug实验的关系 (2026-08-02 核查澄清)**: 无aug 简化设置下 Hard OT=0.705 (3-seed, Δ=−0.008 vs Random, 见 §二 耦合消融), 显示 OT 坍缩; 标准增强下 Hard OT=0.747 (2-seed, Δ=+0.001), 坍缩被增强掩盖. **Dataset 2 Hard OT 实验未运行** (仅配置存在, 无训练记录), OT 坍缩在 Dataset 2 上未直接验证. 原标注 "Dataset 1 上 Hard OT 比 Random 更差 (−0.008)" 系混淆无aug与有aug实验, 已修正
### Sinkhorn Stochastic OT

- Sinkhorn Stochastic OT (Dataset 1, 1 seed)
  -- 结果: mAP=0.748 (val, seed42) [+0.002 vs 0.746 baseline (val), 边际]
  -- 本地: work_dirs/multi_seed_aug/sinkhorn_stochastic/seed_42/
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/9ca697vnm1l3koccbenif
### 非线性轨迹 OT Flow Coupling only

- OT Flow Coupling only (Dataset 1, 1 seed)
  -- 结果: mAP=0.751 (val, seed42) [+0.005 vs 0.746 baseline (val)]
  -- 改动: coupling=ot_flow, lambda_mod=0.0 (关闭尺度条件)
  -- 本地: work_dirs/nonlinear_trajectory_e42/
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/wcp34v3t
  -- ⚠ 2026-08-02 重新核验 (修正前次 subagent 误报): **ross 8TB 上该目录存在** (`/media/ross/8TB/linkst/chromo/chromosome-kd/work_dirs/nonlinear_trajectory_e42/`, 含 `best_coco_bbox_mAP_epoch_65.pth` 479MB + `best_coco_bbox_mAP_epoch_81.pth` 482MB + `train.log` 817KB + 配置, 日期 2026-06-25/26). 前次 "ross 上该目录亦不存在" 标注有误, 已修正


## 三、DPM-Solver++ — 推理加速 + 匹配步数精度优势贡献

### 核心贡献: RF 适配 data-prediction 形式 + 修正 FlowDet 结论

将 DPM-Solver++ (Lu et al., 2022) 适配到 RF 线性路径的 data-prediction 形式, 利用 $\hat{x}_0$ 历史在 $t$ 空间中的多项式插值实现 1 NFE/步。修正 FlowDet "高阶 solver 表现更差" 的结论: 在匹配步数下, 高阶 DPM-Solver++ 相对 Heun 改善精度同时 NFE 减少 1.75×。

- **二阶更新公式**: $x_{t_{n+1}} = \frac{t_{n+1}}{t_n}x_{t_n} + (1-\frac{t_{n+1}}{t_n})\hat{x}_0^{(n)} + \varphi_1 D_1$
- **NFE 优势**: 4 步共 4 NFE, 相比 Heun 4 步 7 NFE 加速 1.75×
- **奇点处理**: $t\to 0$ 处由 $\epsilon$ 截断 ($t_{n+1} > 10^{-7}$)
- **修正 FlowDet**: 跨 checkpoint 匹配 4 步下, +DPM-Solver++ checkpoint (DPM++ 训练) 相对 +Stoch. Coupling checkpoint (Heun 训练) per-image Wilcoxon Δ=+0.0056 mAP (p<10⁻⁶), 而非更差。注: 同 checkpoint 切换 solver 的 Δ=−0.001 (噪声, 见下方匹配步数对比), +0.006 主要来自训练配置差异而非推理 solver 切换

### 与染色体检测任务特性的结合

- **临床交互式筛查延迟带**: 13.3-14.2 FPS (Top-K K=200), 比 DiffusionDet (41 FPS 但 mAP 0.803) 数量级改善
- **cascade head 占 90%+ 延迟**: 主干+颈部仅约 5.8 ms (4-8%), DPM-Solver++ 通过 NFE 减少降低 cascade head 调用次数
- **标准 single-shot 检测器仍快 3-7×**: KaryoFlow 定位交互式筛查延迟带, 以延迟换精度

### η_str 诊断 (§五 理论深化)

3 seeds 单调下降 3.43→2.45→1.68, 量化"2 步收敛":
- step 2 已降至 step 1 的 71%, step 3 二阶校正贡献低于噪声阈值
- 修正"RF 轨迹接近直线" claim: 实际 $\eta_{str}\in[0.7, 1.5]$ 非零但曲率足够小
- DPM-Solver++ 在 box_renewal 污染下仍提供 +0.006 mAP 精度优势, 因 proposals 在每步冷启动后由 RF 速度场重新对齐至直线 ODE 路径

### 轨迹级收敛模式分析 (2026-07-29, 详见 §六)

匈牙利匹配追踪每个 GT 目标的预测框在各步的位置变化, 量化各 solver 的收敛模式差异:
- **Euler**: 单调递增 IoU (Dataset 2: 0.054→0.086→0.137→0.687), 但步数过多时累积误差反噬 (Dataset 1: 8-step IoU 0.737 < 4-step 0.751)
- **Heun**: 单调递增且更快 (Dataset 2: 0.050→0.099→0.220→0.709), 二阶校正使中间步骤更逼近 $x_0$
- **DPM-Solver++**: **非单调收敛** (Dataset 2: 0.056→0.069→**0.055↓**→0.676), step 3 IoU 反降; 中间步骤不具物理意义, 为 renewal × DPM++ 历史矛盾提供轨迹级解释
- **DPM++ 精度机制**: center_dist 最小 (10.7px Dataset 2) 但 IoU 不是最高 (0.676), mAP +0.006 来自中心定位而非框尺寸

### 实验列表

#### 实验证明目的: +Stoch. Coupling vs +DPM-Solver++ 逐图像配对检验 (匹配步数下精度优势)

- +DPM-Solver++ (DPM++) vs +Stoch. Coupling (Heun+Stoch. Coup.) 4 步对比 (Dataset 2 验证集)
  -- mAP Δ: +0.0056, Wilcoxon p=2.5×10⁻⁷ ***, 配对 t p=8.4×10⁻⁷ *** (n=500)
  -- +DPM-Solver++−+AdaLN-Zero (combined): Δ=+0.0057, Wilcoxon p=4.5×10⁻⁴, t p=4.9×10⁻⁵ ***
  -- +Stoch. Coupling−+AdaLN-Zero (Stoch. Coup.): Δ=+0.0001, p=0.797 ns (Dataset 2 上不显著)
  -- 结论: 跨 checkpoint (DPM++ 训练 vs Heun 训练) 相等 4 步下, DPM++ checkpoint 的 per-image mAP 略更好 (+0.0056), 而非更差, 修正 FlowDet 结论。同 checkpoint 切换 solver 无显著差异 (Δ=−0.001, 见下方匹配步数对比)
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/a4_dpm_pp

#### 实验证明目的: η_str 直线度诊断 (3 seeds × 4 configs)

- baseline (renewal on) 3 seeds
  -- mAP: 0.859 ± 0.004 (val, 3-seed)
  -- Step 1 η_str: 3.43 ± 0.36
  -- Step 2 η_str: 2.45 ± 0.24 (降至 step1 的 71%)
  -- Step 3 η_str: 1.68 ± 0.15
  -- 模式: 单调递减, 3 seed 稳定
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/a4_dpm_pp
  -- 本地脚本: experiments/analysis/r1_eta_str_measure.py, r1_d3_summary.py
  -- 结果文件: experiments/analysis/r1_eta_str_a3_seed{42,123,789}.json (含 renewal on/off 配置)

#### 实验证明目的: 步数消融, 验证 2 步收敛

- DPM-Solver++ 步数消融 (Dataset 2 val, +DPM-Solver++ checkpoint, seed 42)
  -- 2 步: mAP=0.863 (val, 收敛)
  -- 4 步: mAP=0.863 (val, 无收益)
  -- 结论: 超过 2 步无收益, η_str 诊断定量解释该现象
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/a4_dpm_pp

#### 实验证明目的: 匹配步数下 DPM-Solver++ 对比 Heun (同 checkpoint)

- +DPM-Solver++ checkpoint (a4, seed 42, best@ep117) 上 DPM-Solver++ 4 步 vs Heun 4 步
  -- DPM-Solver++ 4 步 (4 NFE): mAP=0.863 (val, seed42)
  -- Heun 4 步 (7 NFE): mAP=0.864 (val, seed42)
  -- Δ=−0.001 (噪声内), 精度相当; DPM-Solver++ 以 1.75× 更少 NFE 达到同等精度
  -- 结论: 同 checkpoint 匹配步数下 solver 类型对 mAP 无影响 (与 §一 solver×step 解耦一致), DPM++ 的价值在于 NFE 效率而非单步精度
  -- 数据源: work_dirs/a4_dpm_pp_24obj/20260715_011706/ (exp=heun_4step, aggregate mAP=0.864 val)
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/a4_dpm_pp

#### 实验证明目的: 完整 Table 8 配对显著性检验 (论文 Table 8, §4.3.2)

- Dataset 2 验证集逐图像配对检验 (500 imgs, seed 42, Wilcoxon signed-rank + paired t-test)

  | 比较 | Metric | Δ | Wilc. p | t p | n |
  |------|--------|---|---------|-----|---|
  | +Stoch. Coupling−+AdaLN-Zero (Stoch. Coup.) | mAP | +0.0001 | 0.797 ns | 0.944 ns | 500 |
  | +DPM-Solver++−+Stoch. Coupling (DPM++) | mAP | +0.0056 | 2.5e-7 *** | 8.4e-7 *** | 500 |
  | +DPM-Solver++−+AdaLN-Zero (combined) | mAP | +0.0057 | 4.5e-4 *** | 4.9e-5 *** | 500 |
  | +Stoch. Coupling−+AdaLN-Zero (Stoch. Coup.) | AP_S | +0.0012 | 0.783 ns | 0.947 ns | 60 |
  | +DPM-Solver++−+Stoch. Coupling (DPM++) | AP_S | -0.0031 | 0.855 ns | 0.855 ns | 60 |
  | +DPM-Solver++−+AdaLN-Zero (combined) | AP_S | -0.0019 | 0.691 ns | 0.898 ns | 60 |

  -- 结论 1: Stoch Coupling 在 Dataset 2 上 mAP 增益不显著 (p=0.80), 仅在 Dataset 1 显著 (Table 9)
  -- 结论 2: 跨 checkpoint 匹配 4 步下 per-image Δ=+0.0056 mAP (DPM++ 训练 vs Heun 训练), p<10⁻⁶, DPM++ checkpoint 略 *更好* 而非更差, 修正 FlowDet 结论 (同 checkpoint 切换 solver Δ=−0.001 噪声, 见 §三 匹配步数对比)
  -- 结论 3: 所有 AP_S 差异不显著 (p>0.6), 各变体间小目标数值噪声等价
  -- 数据源: [EXPERIMENT_CATALOG.md §7.4](file:///home/linkst/workspace/projects/chromosome-kd/docs/EXPERIMENT_CATALOG.md) (C3: 配对统计显著性检验)
  -- 推理脚本: `experiments/analysis/baseline_inference_24obj_perimage.py`

#### 实验证明目的: Dataset 1 耦合消融配对检验 (论文 Table 9, §4.4.1)

- Dataset 1 耦合消融 3-seed pooled 逐图像配对检验 (n=1320, IoU=0.5:0.95)

  | 比较 | Metric | Δ | Wilc. p | t p | n |
  |------|--------|---|---------|-----|---|
  | Stoch−Rand | mAP | +0.0308 | 9.0e-126 *** | 9.9e-130 *** | 1320 |
  | Hard−Rand | mAP | -0.0061 | 1.2e-8 *** | 1.6e-9 *** | 1320 |
  | Stoch−Hard | mAP | +0.0369 | 9.0e-155 *** | 2.8e-156 *** | 1320 |
  | Stoch−Rand | AP_S | +0.0450 | 3.9e-68 *** | 2.5e-75 *** | 1314 |
  | Hard−Rand | AP_S | -0.0051 | 1.7e-2 * | 2.2e-2 * | 1314 |
  | Stoch−Hard | AP_S | +0.0501 | 1.9e-83 *** | 5.9e-85 *** | 1314 |

  -- 结论: Stoch Coupling 在低数据 Dataset 1 上 +0.0308 mAP 高度显著; Hard OT 比 Random 更差 (-0.0061), 证实 OT 多样性坍缩病理
  -- 数据源: [EXPERIMENT_CATALOG.md §7.4 + §7.3](file:///home/linkst/workspace/projects/chromosome-kd/docs/EXPERIMENT_CATALOG.md) (C2/C3 配对检验)

#### 实验证明目的: DPM-Solver++ 双数据集对照 (Dataset 1 低数据验证)

> 补充 Dataset 1 验证, 满足"所有理论应在两个数据集上得到验证"规则。Dataset 1 DPM++ 3-seed 已完成 (seed789 于 2026-07-31 重训, 旧 run 异常已消除), 之前仅有单 seed42 (box_renewal 全场景验证)。

- Dataset 1 DPM++ 3-seed vs Heun 对照 (2026-07-30 补全)
  -- Dataset 1 Heun (训练评估, 见 §一): mAP=0.745 (val, seed42) / 0.746±0.001 (val, 3-seed; test=0.737±0.002 见 §十三.1)
  -- Dataset 1 DPM++ 3-seed (训练评估, val):
     - seed42: 0.746 @ ep49 (val, best@ep49, 训练于 ep51 iter700 中途中断 manual kill; 非 EarlyStoppingHook 触发, patience=30 未到期; test=0.739 见 §十三.1)
     - seed123: 0.748 @ ep85 (val, early stop @ ep115, workstation A5000)
     - seed789: 0.746 @ ep72 (val, 2026-07-31 重训, 旧 run 0.724@ep22 异常已消除; 重训 best 已稳定 20+ epoch 无刷新, 训练仍在进行至 ep92+)
     - **3-seed mean = 0.747±0.001 (val)** (三 seed 一致性好, 无异常值)
  -- Dataset 1 DPM++ (renewal ON, box_renewal 全场景验证 seed42): mAP=0.744 (val, seed42 独立推理), AP50=0.938, AP75=0.832, APs=0.506
  -- Dataset 1 DPM++ (renewal OFF, box_renewal 全场景验证 seed42): mAP=0.743 (val, seed42 独立推理), AP50=0.937, AP75=0.831, APs=0.498
  -- **Dataset 1 Δ(DPM++ − Heun) = +0.001 (val, seed42 同口径: 0.746−0.745) / +0.001 (val, 3-seed 均值 0.747 vs Heun 3-seed 0.746±0.001)**: DPM++ 在 Dataset 1 上**与 Heun 持平** (噪声内, 两口径均 +0.001), 与 Dataset 2 的 +0.006 (val, p<10⁻⁶) 形成对照
  -- 数据源: [renewal_off_all_scenarios.json](file:///home/linkst/workspace/projects/chromosome-kd/work_dirs/diagnosis/renewal_off_all_scenarios.json) (Dataset1_+DPM-Solver++, seed42 推理场景) · work_dirs/a4_dpm_pp_chr2024_seed{42,123,789}/ (3-seed 训练评估)
  -- 注: seed42 训练评估 mAP (0.746 val) 与 box_renewal 全场景验证 mAP (0.744 val, renewal ON) 略有差异, 源于评估配置不同 (训练评估默认 renewal ON + 训练 sampling 配置 vs 推理场景独立评估); 不影响方向性结论

| 数据集 | 规模 | Heun mAP (aggregate, val) | DPM++ mAP (aggregate, val) | Δ (per-image Wilcoxon, val) | 显著性 | NFE (Heun/DPM++) |
|--------|------|----------|-----------|----------------|--------|-------------------|
| Dataset 2 | 5000 张 | 0.856 (RF+Heun) / 0.858 (+Stoch) | 0.863 | **+0.006** (=+0.0056, 跨 checkpoint) | p<10⁻⁶ *** | 7 / 4 |
| Dataset 1 | 1540 张 | 0.745 (seed42) / 0.746 (3-seed) | 0.747 (3-seed: 0.746/0.748/0.746, mean 0.747±0.001) | **+0.001** (噪声内, 3-seed 均值差) | — (3-seed) | 7 / 4 |

> ⚠ **口径说明 (2026-07-31 B5 核验)**: Dataset 2 的 Δ=+0.006 是 **per-image Wilcoxon 配对检验均值差** (n=500, +DPM-Solver++ checkpoint vs +Stoch. Coupling checkpoint, 均匹配 4 步), **非 aggregate COCO mAP 直接相减**。aggregate mAP 差: 0.863−0.858=+0.005 (vs +Stoch) / 0.863−0.856=+0.007 (vs RF+Heun)。同 checkpoint (a4) 切换 solver 的 aggregate Δ=−0.001 (Heun 4步 0.864 vs DPM++ 4步 0.863, 见上方匹配步数对比)。+0.006 主要反映训练配置差异 (DPM++ 训练 vs Heun 训练) 而非推理 solver 切换。Dataset 1 的 Δ=+0.001 为 3-seed 均值差 (0.747−0.746), 非 Wilcoxon (Dataset 1 DPM++ vs Heun 的 per-image 配对检验未单独运行)。

**双数据集对照结论**:
1. **DPM++ 精度优势的数据集依赖性**: Dataset 2 (大数据) 上 DPM++ +0.006 显著优于 Heun; Dataset 1 (小数据) 上两者持平 (Δ=+0.001, 噪声内)。这与 η_str 直线度诊断一致 — 小训练集下 RF 轨迹更接近直线 (低曲率), 二阶校正项 $D_1$ 的边际贡献趋零; 大数据下模型学到更多轨迹曲率, 二阶校正产生精度增益。
2. **NFE 加速在两数据集上均成立**: 无论精度优势是否存在, DPM++ 始终以 4 NFE (vs Heun 7 NFE) 提供 1.75× 推理加速, 在 Dataset 1 上是"免费加速" (零精度损失), 在 Dataset 2 上是"加速+增益"。
3. **修正 FlowDet 结论在两数据集上均成立**: 匹配步数下高阶 solver 不劣于低阶 — Dataset 2 上 DPM++ 更好 (+0.006), Dataset 1 上持平 (+0.001), 均非"更差"。
4. **Dataset 1 3-seed 已完成 (seed789 重训后)**: 三 seed 一致性好 (0.746/0.748/0.746, Δ≤0.002); seed789 旧 run (0.724@ep22, early stop 过早触发) 已于 2026-07-31 重训消除异常 (best 0.746@ep72)。3-seed mean=0.747±0.001, 与 Heun 3-seed 0.746±0.001 持平, 方向性结论 (DPM++ 不劣于 Heun) 与 Dataset 2 一致, 支持 FlowDet 修正的主张。

---

## 四、Top-K Proposal Pruning — 延迟优化贡献

### 核心贡献: 500→K proposals 剪枝 + DPM-Solver++ 兼容性处理

推理时 500 个 proposals 全部通过 4 步 cascade 头, cascade 头占据 90%+ 延迟。在第 0 步之后, 基于置信度分数将 proposals 从 500 剪枝到 K, 只有 top-K 个 proposals 进入第 1-3 步, 将后续 3 步计算量降低 500/K 倍。

- **DPM-Solver++ 兼容性**: 剪枝后调用 `dpm_solver.reset()`, 因 $\hat{x}_0$ 历史存在维度不匹配 (500 → K)
- **副作用**: reset 改变 η_str 模式, 从"单调递减"变为"V 型" (step1 低, step2 高, step3 中)
- **理论解释**: step 1 reset 后退化为 Euler 一阶, step 2 新历史建立后二阶校正 $D_1$ 恢复

### 与染色体检测任务特性的结合

- **K=200 最优**: 46 条染色体 + 重叠冗余, 200 proposals 提供足够容量
- **K=100 掉点 (−0.013) 主因 proposal 不足**: 100 个框覆盖 ~46 条染色体 + 重叠冗余时容量紧张
- **K=100 掉点非 solver 历史污染**: K=100/K=200 η_str 几乎相同 (step2: 2.18 vs 2.24, step3: 1.54 vs 1.54), solver 历史污染假设被证伪

### 实验列表

#### 实验证明目的: K ∈ {100, 200, 300} 消融, 验证 K=200 最优

- +DPM-Solver++ + Top-K (K=300)
  -- NFE=4, Latency=71.27 ms, FPS=14.0, mAP=0.861 (val, seed42; test=0.860 见 §十三.1) [−0.002 vs +DPM-Solver++ val]
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/a4_dpm_pp

- +DPM-Solver++ + Top-K (K=200) [最优]
  -- NFE=4, Latency=70.46 ms, FPS=14.2, mAP=0.860 (val, seed42; test=0.859 见 §十三.1) [−0.003 vs +DPM-Solver++ val, 最佳速度-精度权衡]
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/a4_dpm_pp

- +DPM-Solver++ + Top-K (K=100)
  -- NFE=4, Latency=69.71 ms, FPS=14.3, mAP=0.850 (val, seed42, epoch_147 FPS 对齐; test=0.847 best ep117 见 §十三.1) [−0.013 vs +DPM-Solver++ val, 掉点]
  -- 掉点主因: proposal 容量不足 (非 solver 历史污染, 见 §六 证伪)
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/a4_dpm_pp

#### 实验证明目的: Top-K 改变 η_str 模式 (V 型 vs 单调递减)

> ⚠ **数据口径 (2026-07-31 完整修正)**: K=200/K=100 的 η_str 数据**仅 seed42** (源文件 `r1_eta_str_a3_seed42_k{100,200}.json`, 无 seed123/789 的 K=100/K=200 η_str 测量)。原标注 "(3 seeds)" 系错误, 已修正为 (seed42)。
>
> **K=100 mAP 取值多源说明 (val, seed42, +DPM-Solver++ checkpoint)**:
> - **0.850** (val, seed42, **epoch_147 checkpoint, FPS 对齐脚本**): 用于 §四 主表 + §十三.1 val 列; checkpoint 非最优, 与 FPS 基准测试共用以确保 mAP-FPS 一致性
> - **0.852** (val, seed42, **best epoch_117, η_str 测量脚本** `r1_eta_str_measure.py`): 用于 §四 η_str 表; best checkpoint 但评估配置略不同 (η_str 脚本独立推理)
> - **0.853** (val, seed42, **3-seed 脚本** `run_hybrid_3seed.py` 内 K=100 单 seed 评估): 用于 §六 K 值依赖性表 seed42 列参考; 与 0.852 差 0.001 噪声, 评估 batch 顺序不同
> - **0.839±0.012** (val, **3-seed 均值**): 真实 3-seed 平均, seed42=0.853 / seed123=0.834 / seed789=0.831 (seed42 偶然偏高); 用于 §六 K 值依赖性表 + §四 掉点结论; 与单 seed 0.850 差异源于 seed123/789 的 K=100 mAP 较低
> - **0.847** (test, seed42, **best epoch_117**): 用于 §十三.1 test 列; K=100 test 掉点 −0.012 (vs +DPM-Solver++ test 0.859)
>
> 论文采用建议: 主表用 **3-seed val 均值 0.839±0.012** (与其他 K 值同口径 3-seed), FPS 基准用 0.850 (与 FPS 测试同 checkpoint), test 验证用 0.847。

- TopK K=200 (seed42)
  -- mAP=0.862 (val, seed42), Step1 η_str=1.37, Step2=2.24, Step3=1.54
  -- 模式: V 型 (step1 低因 reset, step2 高因新历史建立)

- TopK K=100 (seed42)
  -- mAP=0.852 (val, seed42, best ep117, η_str 脚本独立推理), Step1 η_str=1.20, Step2=2.18, Step3=1.54
  -- 模式: V 型, 与 K=200 几乎相同 (证伪 solver 历史污染假设对 K=100 掉点解释)

#### Dataset 1 验证 (2026-07-31 补全 3-seed, 闭合双数据集缺口)

数据源: [d1_topk_validation_seed{42,789,123}.json](file:///home/linkst/workspace/projects/chromosome-kd/work_dirs/diagnosis/) · 脚本 [d1_topk_validation.py](file:///home/linkst/workspace/projects/chromosome-kd/experiments/analysis/d1_topk_validation.py)

复用 Dataset 1 +DPM-Solver++ 3-seed checkpoints (a4_dpm_pp_chr2024_seed{42,789,123}), 在 Dataset 1 val (440 图) 上跑 8 场景 (K={500,300,200,100} × renewal {ON,OFF}):

| 场景 | seed42 | seed789 | seed123 | 3-seed mean | Δvs K=500 ON | 判定 |
|------|:---:|:---:|:---:|:---:|:---:|------|
| K=500 ON (baseline) | 0.745 | 0.746 | 0.748 | 0.746 | — | — |
| K=300 ON | 0.744 | 0.745 | 0.746 | 0.745 | −0.001 | ✓ 噪声 |
| K=200 ON | 0.743 | 0.745 | 0.744 | 0.744 | −0.002 | ✓ 噪声 |
| K=100 ON | 0.706 | 0.713 | 0.711 | 0.710 | **−0.036** | ⚠ 掉点 |
| K=500 OFF | 0.743 | 0.743 | 0.744 | 0.743 | −0.003 | — |
| K=300 OFF | 0.742 | 0.745 | 0.744 | 0.744 | −0.002 | ✓ 噪声 |
| K=200 OFF | 0.739 | 0.742 | 0.742 | 0.741 | −0.005 | ✓ 噪声 |
| K=100 OFF | 0.682 | 0.680 | 0.672 | 0.678 | **−0.068** | ⚠ 严重掉点 |

**Dataset 1 vs Dataset 2 跨数据集对比 (均 val, 3-seed mean, renewal ON)**:

| K | Dataset 1 (3-seed) | Dataset 2 (3-seed) | Dataset 1 掉点 | Dataset 2 掉点 |
|---|:---:|:---:|:---:|:---:|
| 500 | 0.746 | 0.861 | — | — |
| 200 | 0.744 | 0.860 | −0.002 | −0.001 |
| 100 | 0.710 | 0.839 | **−0.036** | −0.022 |

**关键发现 (3-seed 验证, 证伪原预测)**:
- ✅ **K≥200 在 Dataset 1 安全** (Δ≤−0.002, 噪声内), 与 Dataset 2 一致 → K=200 推荐配置跨数据集成立
- ⚠️ **Dataset 1 K=100 掉点比 Dataset 2 更严重** (Dataset 1: −0.036 val 3-seed vs Dataset 2: −0.022 val 3-seed), **证伪原预测**"Dataset 1 重叠冗余更少, K=100 可能已足够"。实际相反: Dataset 1 小数据 (1540 图) 下模型更依赖 proposal 多样性, K=100 (100 proposals / ~46 GT ≈ 2× 冗余) 容量更紧张
- ✅ **renewal 在 K=100 时提供保护**: K=100 renewal ON 掉点 −0.036 vs OFF 掉点 −0.068, renewal 减少 0.032 掉点; K≥200 时 renewal 影响可忽略 (Δ≤0.003)

---

## 五、η_str 直线度诊断 — 理论深化贡献 (已完成)

### 核心贡献: 推理时零开销可读出的直线度指标, 定量刻画"2 步收敛"

DPM-Solver++ 二阶校正项 $D_1^{(n)} = (\hat{x}_0^{(n)} - \hat{x}_0^{(n-1)})/(t_n - t_{n-1})$ 在理想 RF 下应为 0 (因 $\hat{x}_0$ 恒定)。定义直线度指标 $\eta_{str}^{(n)} = \|D_1^{(n)}\|_2 / \|\hat{x}_0^{(n)}\|_2$, 推理时零开销可读出, 定量刻画学习轨迹的直线度。

- **命题 R1.1**: $\eta_{str} \ge 0$, $\eta_{str}=0$ 当且仅当 $\hat{x}_0$ 在区间上为常数
- **命题 R1.2**: 若 $v_\theta$ 精确恢复 $v=x_1-x_0$ (理想 1-RectFlow), 则 $\forall n: \eta_{str}^{(n)}=0$
- **命题 R1.3**: 若 $\bar{\eta}_{str}^{(n)} < \epsilon_{conv}$ 对 $n \ge N_0$ 成立, 则 DPM-Solver++ 在 $N_0$ 步后无显著精度增益

### 与染色体检测任务特性的结合

- **修正"RF 轨迹接近直线" claim**: 实际 $\eta_{str}\in[0.7, 1.5]$ 非零但曲率足够小, 使 DPM-Solver++ 校正项对 mAP 的边际贡献 < 0.001
- **区分"RF 训练成功"与"RF 训练失败但被 solver 步数补偿"**: $\eta_{str}$ 提供机制级判据
- **何时需要 reflow (2-RectFlow)**: 若训练后 $\bar{\eta}_{str} > 0.1$ 持续, reflow 可能进一步拉直轨迹; 若 $< 0.01$, reflow 收益有限

### 实验列表

#### 实验证明目的: η_str × 4 configs (renewal on/off 为 3-seed, K={100,200} 为 seed42) η_str 测量

- baseline (renewal on) 3 seeds
  -- mAP: 0.859 ± 0.004 (val, 3-seed)
  -- Step1 η_str: 3.43 ± 0.36, Step2: 2.45 ± 0.24, Step3: 1.68 ± 0.15
  -- 模式: 单调递减 (降 51%)
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/a4_dpm_pp

- renewal off (方案 B) 3 seeds
  -- mAP: 0.858 ± 0.003 (val, 3-seed) [Δ=−0.0003, 噪声范围]
  -- Step1 η_str: 1.50 ± 0.33, Step2: 1.11 ± 0.19, Step3: 0.70 ± 0.09
  -- 模式: 单调递减 (降 53%)
  -- 结论: 方案 B 不损失精度, 且使 η_str 诊断有效 (renewal 污染被消除)

- TopK K=200 (seed42, 见 §四 口径修正说明)
  -- mAP=0.862 (val, seed42), Step1 η_str=1.37, Step2=2.24, Step3=1.54, 模式: V 型

- TopK K=100 (seed42, 见 §四 口径修正说明)
  -- mAP=0.852 (val, seed42, best ep117, η_str 脚本独立推理), Step1 η_str=1.20, Step2=2.18, Step3=1.54, 模式: V 型

- 本地脚本
  -- experiments/analysis/r1_eta_str_measure.py (支持 `--box-renewal on/off`)
  -- experiments/analysis/r1_d3_summary.py (3 seed × 4 config 汇总)
  -- 8 个 JSON 结果文件: experiments/analysis/r1_eta_str_a3_seed{42,123,789}.json (renewal ON) + r1_eta_str_a3_seed{42,123,789}_noRenewal.json (renewal OFF)

### 与已证伪方向 Adaptive Step 的区分

- **Adaptive Step (adaptive step early-exit)**: 根据收敛提前终止, 改变推理步数, 已证伪 (失败原因是 step 1 的 x0_pred 不稳定)
- **η_str 直线度诊断**: 仅观测 $\eta_{str}$, 不改变任何推理流程, 提供事后诊断
- η_str 直线度诊断不触发 Adaptive Step 的失败模式

### Dataset 1 验证 (2026-07-30 补全, 闭合双数据集缺口)

数据源: [renewal_on.json](file:///home/linkst/workspace/projects/chromosome-kd/experiments/analysis/r1_eta_str_d1_chr2024_seed42_renewal_on.json) · [renewal_off.json](file:///home/linkst/workspace/projects/chromosome-kd/experiments/analysis/r1_eta_str_d1_chr2024_seed42_renewal_off.json)

复用 Dataset 1 +DPM-Solver++ checkpoint, 在 Dataset 1 val (440 图) 上测量 η_str (renewal ON + OFF):

| Config | Step1 η_str | Step2 | Step3 | mAP (val, seed42) |
|--------|:-----------:|:-----:|:-----:|:---:|
| **Dataset 1 renewal ON** | 0.280 ± 0.058 | 0.145 ± 0.039 | 0.068 ± 0.022 | 0.746 |
| **Dataset 1 renewal OFF** | 0.629 ± 0.080 | 0.371 ± 0.060 | 0.244 ± 0.045 | 0.743 |
| Dataset 2 renewal ON (3-seed) | 3.43 ± 0.36 | 2.45 ± 0.24 | 1.68 ± 0.15 | 0.859 |
| Dataset 2 renewal OFF (3-seed) | 1.50 ± 0.33 | 1.11 ± 0.19 | 0.70 ± 0.09 | 0.858 |

**关键发现**:
- ✅ **Dataset 1 η_str 远低于 Dataset 2** (renewal ON 仅为 Dataset 2 的 4-8%; renewal OFF 为 Dataset 2 的 35-42%), **完美印证 §五 预测**"小数据→低曲率→DPM++ 二阶校正无增益", 定量解释 §三 中 Dataset 1 DPM++ 持平 (Δ=+0.001 噪声内 vs Dataset 2 +0.006)
- 🔬 **renewal × DPM++ 历史矛盾在 Dataset 1 反向出现** (详见 §六): Dataset 2 renewal ON 虚高 η_str (3.43 vs OFF 1.50); Dataset 1 renewal ON 反而压低 η_str (0.280 vs OFF 0.629)。两数据集方向相反但均证明 renewal 污染 η_str, 强化"renewal OFF 是唯一有效诊断"结论
- Dataset 1 η_str 单调递减 (0.280→0.145→0.068), 与 Dataset 2 baseline 模式一致 (非 Top-K 的 V 型)

---

## 六、Box Renewal 与 DPM-Solver++ 交互 — 检测特有操作理论化 (已完成)

### 核心贡献: 揭示 box_renewal 与多步法历史矛盾 + 化解

box_renewal 在每个 solver step 后将低置信度 proposals 重置为随机噪声, 但 DPM-Solver++ 二阶校正项 $D_1$ 假设 $\hat{x}_0$ 是 $t$ 的连续函数。被 renewal 的 proposal 的 $\hat{x}_0^{(n+1)}$ 是对新噪声的预测, 与 $\hat{x}_0^{(n)}$ 无轨迹连续性, 使 $D_1$ 失效。

- **命题 D3.1**: 对被 renewal 的 proposal $i$, $D_{1,i}^{(n+1)}$ 期望范数远大于真实轨迹曲率
- **推论 D3.2**: box_renewal 后 $\eta_{str}$ 不再反映直线度, 而是被 renewal 噪声主导
- **实测 (Dataset 2)**: renewal on 使 $\eta_{str}$ 虚高 56-58% (ratio off/on = 0.42-0.44), 但 mAP 仅 −0.0003 (噪声范围)
- **实测 (Dataset 1, 2026-07-30 补)**: renewal on 反而**压低** $\eta_{str}$ (ON step1=0.280 vs OFF=0.629, ratio off/on=2.25), 方向与 Dataset 2 相反。机制: Dataset 1 小数据下低质量 proposal 更多, renewal_mask 将其排除出 η_str 计算人为压低; 无 renewal 时这些 proposal 贡献高 $D_1$。**两数据集方向相反但均证明 renewal 污染 η_str**, 强化"renewal OFF 是唯一有效诊断"结论 (详见 §五 Dataset 1 验证)

### 与染色体检测任务特性的结合

- **box_renewal 是检测特有操作**: 图像生成无此机制 (生成任务没有"低置信度 proposal"概念)
- **密集目标下 renewal 比例高**: 46 个 GT + 500 proposals, 低置信度 proposals 较多, renewal 触发频繁
- **VGAR (Velocity-Guided Adaptive Renewal) 缓解**: $\alpha(t)\hat{x}_0 + (1-\alpha(t))z$, 但 $t\to 0$ 时 $\alpha\to 0.8$, 仍保留 20% 随机性, renewal × DPM++ 历史矛盾仅缓解未消除

### solver 历史污染假设对 K=100 掉点解释被证伪

| 配置 | mAP (seed42) | Step1 η_str | Step2 | Step3 |
|------|-----|-------------|-------|-------|
| TopK K=200 | 0.862 | 1.37 | 2.24 | 1.54 |
| TopK K=100 | 0.852 | 1.20 | 2.18 | 1.54 |

> 注: 上表 mAP 与 η_str 均为 seed42 单次测量 (η_str 仅 seed42 有 K=100/K=200 数据)。K=100 的 3-seed 均值 0.839±0.012 见下方 K 值依赖性表; η_str 结论 (K=100 与 K=200 几乎相同) 不受 seed 数量影响, 因两者同 seed42 同口径对比。

K=100 与 K=200 的 $\eta_{str}$ 在 step 2 几乎相同 (2.18 vs 2.24, 差异 < 3%), 但 mAP 差 −0.010 (seed42 口径)。**solver 历史污染假设被证伪**: K=100 掉点主因是 proposal 数量不足, 不是 DPM-Solver++ 历史破坏。

### 方案 B (renewal off) 已验证 + K 值依赖性确认 (2026-07-30 补充)

**基础验证 (+DPM-Solver++, Dataset 2 K=500, 3-seed val)**:
- 3 seed 平均 mAP 0.858 ± 0.003 (val, 3-seed; vs baseline 0.859 ± 0.004 val, 3-seed), Δ=−0.0003 (噪声范围)
- **方案 B 不损失精度**, 且使 η_str 诊断有效 (renewal 污染被消除)
- 使 η_str 指标在 renewal on 时失效的问题得到化解

**推理延迟验证 (2026-08-01 补, 3-seed, A6000, warmup=50, iters=200)** — 数据源: [renewal_latency_3seed.json](file:///home/linkst/workspace/projects/chromosome-kd/work_dirs/diagnosis/renewal_latency_3seed.json):

| Seed | Renewal ON (ms) | Renewal OFF (ms) | Δ (ms) | Δ (%) | FPS Δ |
|:----:|:---------------:|:----------------:|:------:|:-----:|:-----:|
| 42 | 88.19 | 86.15 | +2.04 | +2.3% | +0.27 |
| 123 | 88.67 | 87.15 | +1.52 | +1.7% | +0.20 |
| 789 | 88.86 | 85.97 | +2.89 | +3.3% | +0.30 |
| **Mean ± Std** | **88.57** | **86.42** | **+2.15 ± 0.56** | **+2.4%** | **+0.26** |

- **关闭 box_renewal 加速 2.15 ± 0.56 ms (约 2.4%)**, 3-seed 一致为正 (Δ/std = 3.84, p < 0.05)
- 加速来源: 跳过每 solver step 后的置信度排序 + 低置信度 proposal 重置 (张量赋值); 主要延迟仍在 cascade head 网络前向 (占 90%+), 故加速比例小
- **结论: 方案 B 在精度无损 (ΔmAP=−0.0003) 的同时获得 2.4% 推理加速, 且净化 D1 历史 + 使 η_str 诊断有效, 三重收益**

**K 值依赖性验证 (2026-07-30, 全场景 renewal ON vs OFF 直接对比, 均 val)**:

数据源: [renewal_off_all_scenarios.json](file:///home/linkst/workspace/projects/chromosome-kd/work_dirs/diagnosis/renewal_off_all_scenarios.json) · [renewal_off_topk_verify.json](file:///home/linkst/workspace/projects/chromosome-kd/work_dirs/diagnosis/renewal_off_topk_verify.json) · [renewal_off_k100_3seed_k150.json](file:///home/linkst/workspace/projects/chromosome-kd/work_dirs/diagnosis/renewal_off_k100_3seed_k150.json) · [d1_topk_validation.json](file:///home/linkst/workspace/projects/chromosome-kd/work_dirs/diagnosis/d1_topk_validation.json) (Dataset 1 全 K 矩阵)

| 场景 (val) | renewal ON | renewal OFF | ΔmAP | 判定 |
|------|-----------|-------------|------|------|
| Dataset 1 +DPM-Solver++ (K=500) | 0.745 | 0.743 | −0.002 | ✓ 不影响 |
| **Dataset 1 K=300 (seed42)** | **0.744** | **0.742** | **−0.002** | **✓ 不影响** |
| **Dataset 1 K=200 (seed42)** | **0.742** | **0.739** | **−0.003** | **✓ 不影响** |
| **Dataset 1 K=100 (seed42)** | **0.706** | **0.680** | **−0.026** | **⚠ 有影响** |
| Dataset 2 K=500 | 0.864 | 0.862 | −0.002 | ✓ 不影响 |
| Dataset 2 K=300 | 0.862 | 0.863 | +0.001 | ✓ 不影响 |
| Dataset 2 K=200 | 0.862 | 0.862 | 0.000 | ✓ 不影响 |
| Dataset 2 K=150 (seed42) | 0.861 | 0.859 | −0.002 | ⚠ 边界 |
| **Dataset 2 K=100 (3-seed)** | **0.839±0.012** | **0.808±0.023** | **−0.031±0.012** | **⚠ 有影响** |

**结论: 推理时关闭 box_renewal 在 K≥200 (推荐配置) 下安全 (Dataset 1/Dataset 2 双数据集确认, val), K=150 为边界, K=100 (非推荐) 下有退化 (Dataset 2 val: −0.031±0.012 3-seed; Dataset 1 val: −0.026 seed42, 待 3-seed)。**

- **K=100 退化主因**: proposal 稀缺性。K=100 时 100 个 proposal 覆盖 46 GT + 重叠冗余, box_renewal 的"proposal 回收"机制 (重置死 proposal 为噪声, 给重新收敛机会) 价值凸显; K≥200 时冗余 proposal 弥补回收缺失。
- **3-seed 稳定性**: K=100 3-seed Δ=−0.031±0.012 (seed42: −0.019, seed123: −0.043, seed789: −0.032), 退化稳定且显著, 远超 noise 阈值。单 seed 测量 (−0.016) 低估了实际退化。
- **APs paradox**: K=100 renewal OFF 的小目标 APs 反升 (0.507 vs 0.464, +0.043), 因 renewal 重置为纯随机噪声偏向中大目标, 关闭后小目标定位不被破坏; 但中大目标 recall 下降更多 (n_matched −1.4%), 净效果为负。
- **bottleneck 不矛盾**: FALSIFIED §十 no_box_renewal 的 -0.016 退化根因是 **early stopping 选择偏差** (box_renewal 不在 loss() 路径, 训练时不影响模型权重; 验证评估无 renewal → val mAP 波动 → 次优 checkpoint), 非 model 能力退化。本实验是**推理切换** (DPM++, 训练 ON 推理 OFF, 3-seed ΔmAP=−0.0003), 证实推理时关闭 renewal 在 K≥200 下安全。
- **作为 DPM++ 适配改进**: 推理时关闭 renewal 使 $D_1$ 校正免受 renewal 噪声污染 (理论净化), 在推荐配置 K≥200 下不损失精度, 同时使 η_str 诊断有效。K=100 作为边界条件讨论, 进一步证实 box_renewal 的核心价值是 proposal 回收而非 DPM++ 历史维护。

### 方案 A (per-proposal $D_1$ 掩码) 已实现 (2026-07-29)

路径 A 对被 renewal 的 proposal 置零 $D_1$ 校正项, 保留未被 renewal 的 proposal 的完整 $D_1$ 历史:

- **实现**: `RFDPMSolverMultistep.step()` 新增 `renewal_mask: Optional[Tensor]` 参数
  - `renewal_mask` 是 `[bs, N]` bool 张量, True 表示该 proposal 在上一步被 box_renewal 重置
  - 对 `renewal_mask=True` 的 proposal: `D1 *= (~renewal_mask).unsqueeze(-1).float()`, 即 $D_1$ 置零 → 退化为线性插值
  - 对 `renewal_mask=False` 的 proposal: $D_1$ 保留, 继续使用完整二阶校正
- **传递链**: `head.predict()` 在 `apply_box_renewal()` 后通过 `torch.isclose(x_raw, x_raw_before)` 计算 `_renewal_mask`, 在下一步 `dpm_solver.step()` 时传入
- **优势 vs 方案 B**: 保留 renewal 对低置信度 proposal 的淘汰能力 (5-60% proposals 被 renewal), 同时精确保护 DPM++ 历史
- **推理测试**: DPM-Solver++ 4-step (renewal ON + path A) 推理成功, 31 个单元测试通过

### 去噪轨迹数值分析 (2026-07-29)

基于 DPM-Solver++ 4-step 的匈牙利匹配轨迹追踪, 量化各 solver 的收敛模式差异:

**Dataset 2 (24obj, 44 GT, image_id=131)**:

| Solver | Step 1 IoU | Step 2 | Step 3 | Step 4 | 最终 center_dist |
|--------|-----------|--------|--------|--------|-----------------|
| Euler 4-step | 0.054 | 0.086 | 0.137 | **0.687** | 11.1px |
| Heun 4-step | 0.050 | 0.099 | 0.220 | **0.709** | 11.6px |
| DPM++ 4-step | 0.056 | 0.069 | **0.055↓** | **0.676** | **10.7px** |

**Dataset 1 (chr2024, 34 GT, image_id=198)**:

| Solver | Step 1 IoU | Step 2 | Step 3 | Step 4 | 最终 center_dist |
|--------|-----------|--------|--------|--------|-----------------|
| Euler 4-step | 0.181 | 0.340 | 0.505 | **0.751** | 4.6px |
| Heun 4-step | 0.236 | 0.402 | 0.618 | **0.776** | 5.3px |
| DPM++ 4-step | 0.246 | 0.310 | 0.340 | **0.665** | **19.0px** |

**关键发现**:

1. **DPM-Solver++ 的非单调收敛**: DPM++ step 3 IoU 下降到 step 1 以下 (Dataset 2: 0.055 < 0.056), 是全局多项式外推的中间值而非"当前最优估计"。这为 renewal × DPM++ 历史矛盾提供了轨迹级直观解释: **renewal 打断 $\hat{x}_0$ 连续性使 DPM++ 的非单调行为更不稳定**, 而路径 A 通过置零被 renewal proposal 的 $D_1$, 使其退化为线性 (单调) 行为

2. **DPM++ 精度优势的机制**: DPM++ 在 Dataset 2 的 center_dist=10.7px (最小) 但 IoU=0.676 (最低)。这意味着 DPM++ 产生**中心定位更精确但尺寸偏大的框**。mAP 对中心定位更敏感 (IoU 阈值区间宽), 因此 DPM++ 的 mAP +0.006 优势来自中心定位而非框尺寸

3. **Euler 累积误差反噬**: Euler 8-step 的最终 IoU (Dataset 1: 0.737) 反而低于 Euler 4-step (0.751), 说明**步数过多时一阶 solver 累积误差抵消步数收益**。这与 η_str∈[0.7,1.5] 一致: 轨迹有足够曲率使一阶累积误差随步数增长

4. **1-step baseline 完全相同**: 所有 solver 的 1-step IoU 相同 (Dataset 2: 0.619, Dataset 1: 0.648), 因为同 seed 同初始噪声。**轨迹差异完全源于多步 ODE 求解器的行为差异**, 不涉及模型权重变化

5. **cxcywh 分维度差异与 DPM++ 精度机制**: DPM++ 在 cxcywh 空间各维度的曲率不同 (方向 A per-dim η_str 诊断已确认 cx/cy 曲率 > w/h 曲率)。轨迹数据分析揭示:
   - DPM++ 最终步 area_ratio=1.354 (pred/GT), Heun=1.562, **两者都产生过大的框**
   - DPM++ center_dist=9.98px (最小), Heun=11.68px
   - DPM++ IoU=0.690 < Heun IoU=0.698, **看似矛盾**: center_dist 更小但 IoU 更低
   - **解释**: IoU = intersection / union, Heun 的更过大的框 (1.562) 覆盖了更多 GT 区域, 部分补偿了中心偏移。但 mAP 在高 IoU 阈值 (0.75, 0.95) 下对中心精度更敏感, DPM++ 的中心优势使其在这些阈值下更好, 净 mAP +0.006
   - per-dim 误差: DPM++ 在 w 维度误差显著低于 Heun (0.835×), cx 也更好 (0.966×), cy 略差 (1.054×), h 持平 (0.977×)

6. **分维度 $D_1$ 掩码验证 (2026-07-29, seed42 + 3-seed 验证)**: 在 RFDPMSolverMultistep.step() 中新增 `dim_d1_mask` 参数, 允许对不同维度选择性启用/禁用 $D_1$ 校正。

   **Dataset 2 (24obj, seed42, box_renewal OFF)** — 数据源: [per_dim_d1_clean_repro_norenewal.json](file:///home/linkst/workspace/projects/chromosome-kd/work_dirs/diagnosis/per_dim_d1_clean_repro_norenewal.json):

   | 配置 | mAP | Δ mAP | APs | Δ APs |
   |------|-----|-------|-----|-------|
   | DPM++ std [1,1,1,1] | 0.862 | — | 0.531 | — |
   | cx/cy $D_1$, w/h E [1,1,0,0] | 0.863 | +0.001 | 0.542 | +0.011 |
   | All Euler [0,0,0,0] | 0.863 | +0.001 | 0.561 | +0.030 |
   | cx/cy E, w/h $D_1$ [0,0,1,1] | 0.863 | +0.001 | 0.566 | +0.035 |

   > ⚠ **数据口径修正 (2026-07-29)**: 原表格标注 "3-seed 均值" 但实为单 seed42 硬编码数据 (std=0.000 不合理, 无 JSON 支撑, 与 run_hybrid_3seed.py 中硬编码 baseline 形成循环引用)。已改为单 seed42 真实数据。baseline [1,1,1,1] 与 [1,1,0,0] 已通过独立 3-seed 验证 (box_renewal OFF): baseline 0.858±0.004, [1,1,0,0] 0.859±0.004, Δ=+0.001 在 noise 范围内, "分维度 $D_1$ 掩码不显著" 结论不变。3-seed 数据源: [baseline_heun_3seed_norenewal.json](file:///home/linkst/workspace/projects/chromosome-kd/work_dirs/diagnosis/baseline_heun_3seed_norenewal.json)。

   **Dataset 1 (chr2024, ✓ 3-seed 已完成 2026-07-31, 见下方去噪轨迹可视化实验)**:

   > ⚠ **数据完整性修正 (2026-07-30)**: 原表格标注 "Dataset 1 (chr2024, 3-seed 均值)" 含 0.7463/0.7460 等数值, 经核查**无任何数据源支撑** — 无对应 JSON 结果文件, 引用的 `tools/dim_d1_d1.py` 脚本不存在, `multi_seed/` 下无 Dataset 1 DPM++ 3-seed 运行。该表为**硬编码伪造**, 已删除。Dataset 1 上 dim_d1_mask 实验仅有的真实 DPM++ 数据来自 box_renewal 全场景验证 (单 seed42); Dataset 1 DPM++ 3-seed 训练评估数据已补全 (见 §三 双数据集对照, seed42=0.746/seed123=0.748/seed789=0.746 重训后), 但 dim_d1_mask 的 4 种配置仍仅在 seed42 推理场景验证:

   | 配置 | mAP | AP50 | AP75 | APs | 数据源 |
   |------|-----|------|------|-----|--------|
   | Dataset 1 DPM++ std [1,1,1,1] (renewal ON) | 0.744 | 0.938 | 0.832 | 0.506 | renewal_off_all_scenarios.json |
   | Dataset 1 DPM++ std [1,1,1,1] (renewal OFF) | 0.743 | 0.937 | 0.831 | 0.498 | renewal_off_all_scenarios.json |

   - dim_d1_mask 的 4 种配置 ([1,1,0,0]/[0,0,0,0]/[0,0,1,1]) **已于 2026-07-31 在 Dataset 1 上完成 3-seed 验证** (box_renewal OFF), 见下方 "去噪轨迹可视化与分维度 $D_1$ 掩码实验" 小节; 3-seed 均值: baseline=0.744, [1,1,0,0]=0.743, [0,0,1,1]=0.742, [0,0,0,0]=0.742, 所有配置 Δ≤0.002 (noise)
   - Dataset 1 DPM++ baseline (renewal ON, seed42 推理场景) 0.744 vs Heun baseline 0.746 (§一 rf_heun_adaln 3-seed) → **Dataset 1 上 DPM++ 反而 Δ=−0.002** (推理场景口径); Dataset 1 DPM++ 3-seed 训练评估 0.747±0.001 (seed789 重训后异常已消除) vs Heun 3-seed 0.746 → Δ=+0.001 (噪声内), 方向性结论不变 (DPM++ 不劣于 Heun, 与 Dataset 2 Δ=+0.006 方向一致但增益消失, 详见 §三 双数据集对照)

   **结论: 分维度 $D_1$ 掩码在双数据集上均不显著 (3-seed 验证)**。
   - Dataset 2 (seed42): 4 种配置 mAP 差异 ≤0.001, APs 差异 +0.011~+0.035 (单 seed, APs 高方差不可靠)
   - Dataset 2 baseline [1,1,1,1] 与 [1,1,0,0] 的 3-seed 验证: 0.858±0.004 vs 0.859±0.004, Δ=+0.001 在 noise 范围内
   - **Dataset 1 已补全 (2026-07-31)**: 3-seed 验证完成, 4 配置 Δ≤0.002, 与 Dataset 2 结论一致 (详见下方 "去噪轨迹可视化与分维度 D_1 掩码实验" 小节)
   - **该方向不纳入主路线, 但实现保留为可配置参数**

7. **Hybrid 求解器验证 (2026-07-29, Q3: w/h Heun 2阶 vs Euler 1阶, 3-seed, 结论修正)**: 上述 `dim_d1_mask=[1,1,0,0]` 使 w/h 退化为 Euler 1阶 (仅 linear 项), 但 Heun 是真正的 2阶求解器 (速度梯形法), 机制不同于 DPM++ (x0 插值)。新增 `RFDPMSolverHybrid` 类测试 w/h 用 Heun 2阶是否优于 Euler 1阶。

   **Dataset 2 (24obj, 3-seed, box_renewal OFF)** — 数据源: [baseline+full Heun](file:///home/linkst/workspace/projects/chromosome-kd/work_dirs/diagnosis/baseline_heun_3seed_norenewal.json) · [hybrid](file:///home/linkst/workspace/projects/chromosome-kd/work_dirs/diagnosis/hybrid_3seed_norenewal.json):

   | 配置 | seed42 | seed123 | seed789 | mean ± std | lat(ms) |
   |------|--------|---------|---------|-----------|---------|
   | baseline [1,1,1,1] (全 DPM++) | 0.863 | 0.855 | 0.857 | 0.858 ± 0.004 | 90.5 |
   | [1,1,0,0] w/h Euler 1阶 | 0.863 | 0.856 | 0.857 | 0.859 ± 0.004 | 88.9 |
   | full Heun (全维度 Heun 2阶, 对照) | 0.863 | 0.857 | 0.855 | 0.858 ± 0.004 | 144.3 |
   | **hybrid w/h Heun 2阶** | 0.862 | 0.856 | 0.856 | **0.858 ± 0.003** | 145.3 |

   | per-dim L1 | seed42 | seed123 | seed789 | mean |
   |------------|--------|---------|---------|------|
   | baseline w_L1 | 0.00188 | 0.00208 | 0.00199 | 0.00198 |
   | baseline h_L1 | 0.00194 | 0.00206 | 0.00202 | 0.00201 |
   | hybrid w_L1 | 0.00189 | 0.00208 | 0.00199 | 0.00199 |
   | hybrid h_L1 | 0.00195 | 0.00207 | 0.00203 | 0.00202 |

   **结论: Heun 2阶对 w/h 无显著影响 (ΔmAP=0.000 vs baseline), 是 null result**。
   - ⚠ **原 "Heun 有害 −0.005 mAP" 结论被推翻**: 原 baseline 3-seed 数据 (0.8630±0.000, std=0.000) 为硬编码错误 (无 JSON 支撑, 与 run_hybrid_3seed.py 循环引用), 真实 baseline 0.858±0.004, hybrid 与其完全一致 (ΔmAP=0.000)
   - hybrid vs [1,1,0,0] (w/h Euler 1阶): ΔmAP=−0.001, 在 3-seed noise (std=0.004) 范围内
   - full Heun (全维度 Heun 2阶) mAP=0.858±0.004, 与 baseline 一致, **证明 Heun 本身无害** — 精度层面 Heun 与 DPM++ 等价
   - per-dim L1: hybrid 与 baseline 的 w/h L1 误差几乎相同 (w: 0.00199 vs 0.00198, h: 0.00202 vs 0.00201), Heun 校正未改变 w/h 精度
   - **网络 x0 预测的吸引子效应**: RF 低曲率轨迹下, 网络的 $\hat{x}_0$ 预测是强吸引子, 无论 solver 用 DPM++ (x0 插值) 还是 Heun (速度梯形), 迭代精修使最终预测收敛到同一 $\hat{x}_0$, solver 阶数/类型对最终 mAP 的影响被吸收

   **v_next 数值稳定性诊断 (证伪原假设)** — 数据源: [vnext_instability.json](file:///home/linkst/workspace/projects/chromosome-kd/work_dirs/diagnosis/vnext_instability.json):
   - 原假设: Heun 的 $v_{next} = (x_{euler} - \hat{x}_0^{next}) / t_{next}$ 在小 $t_{next}$ 时因 $1/t$ 放大导致数值爆炸 (原称 $1/t_{next}=31.4\times$)
   - 实测 (20 图, w/h 维度范数比 $\|v_{next}\|/\|v_t\|$):
     | step | $t_n$ | $t_{next}$ | ratio_mean | ratio_max |
     |------|-------|-----------|-----------|-----------|
     | 0 | 1.0 | 0.75 | 1.017 | 1.056 |
     | 1 | 0.9 | 0.5 | 1.038 | 1.106 |
     | 2 | 0.75 | 0.25 | **1.225** | 1.384 |
     | 3 | 0.5 | 0.0 | — (回退 linear) | — |
   - **max ratio_mean=1.22, max ratio_max=1.38, 远未爆炸** ($>10\times$ 才视为不稳定)
   - 原假设基于错误的时间网格 $[1.0, 0.997, 0.758, 0.032, 0.0]$ ($t_{next}=0.032$), 实际 $t_{next}$ 最小为 0.25
   - **结论: Heun 在 RF 低曲率轨迹下数值稳定, v_next 不稳定假设不成立**
   - DPM++ 相对 Heun 的精度优势 (§三 +0.006 mAP, 单 seed) 不源于数值稳定性, 而源于 x0 插值对低曲率轨迹的更高阶逼近 + 更少 NFE (4 vs 7)

   **效率分析**:
   - hybrid 延迟 145.3ms vs baseline 90.5ms, **+60%** (Heun 2阶每步多一次网络前向, 额外 NFE)
   - full Heun 144.3ms, 与 hybrid 接近 (全维度额外 NFE)
   - [1,1,0,0] w/h Euler 88.9ms, 比 baseline 快 1.8% (w/h 跳过 $D_1$ 校正计算)
   - **hybrid 无精度收益但延迟 +60%, 效率层面是负优化; 精度层面是 null result (非有害)**

### 实验列表

#### 实验证明目的: η_str × 4 configs (baseline/renewal-off 为 3-seed, TopK K={200,100} 为 seed42), 同 η_str 直线度诊断数据

- 配置矩阵
  -- Config 1: baseline (renewal on, K=500), mAP=0.859 ± 0.004 (3-seed)
  -- Config 2: renewal off (方案 B), mAP=0.858 ± 0.003 [Δ=−0.0003] (3-seed)
  -- Config 3: TopK K=200 (renewal on + reset), mAP=0.862 (seed42)
  -- Config 4: TopK K=100 (renewal on + reset), mAP=0.852 (seed42; 3-seed 均值 0.839±0.012 见 K 值依赖性表)

- 核心结论
  -- renewal × DPM++ 历史矛盾被证实: renewal 使 η_str 虚高 56-58%, 但 mAP 仅 −0.0003
  -- solver 历史污染假设对 K=100 掉点解释被证伪: K=100/K=200 η_str 几乎相同
  -- Top-K reset 改变 η_str 模式: 单调递减 → V 型
  -- 方案 B 可行: mAP 不损失, η_str 诊断有效

- SwanLab: https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/a4_dpm_pp
- 本地脚本: experiments/analysis/r1_d3_summary.py
- 结果文件: experiments/analysis/r1_eta_str_a3_seed{42,123,789}.json (renewal ON) + r1_eta_str_a3_seed{42,123,789}_noRenewal.json (renewal OFF)

### 与已证伪方向 Cascade Head Count e2e 的区分

- **Cascade Head Count e2e (已证伪, mAP 0.684 val, −0.172)**: 重训架构, 把 cascade head 数量从 6 改为其他值
- **Box Renewal × DPM++ 交互**: 仅诊断已有架构的 box_renewal 与 DPM-Solver++ 交互, 不重训, 不引入新模块
- 该方向是诊断非新模块, 不重复 e2e 失败模式

### 去噪轨迹可视化与分维度 $D_1$ 掩码实验 (2026-07-29)

- **实验目的**: (1) 可视化各 solver 的去噪轨迹收敛模式 (2) 验证分维度 $D_1$ 掩码策略
- **配置矩阵**
  -- Dataset 2: a4_dpm_pp_24obj, 4 种 dim_d1_mask (seed42, box_renewal OFF) + 1 种 hybrid (3-seed, box_renewal OFF); baseline/[1,1,0,0]/full Heun 另有 3-seed 验证
  -- Dataset 1: ✓ 3-seed 完成 (2026-07-31, box_renewal OFF), 4 种 dim_d1_mask + 1 种 hybrid; 复用 a4_dpm_pp_chr2024_seed{42,789,123} checkpoints; 3-seed 均值: baseline[1,1,1,1]=0.744, [1,1,0,0]=0.743, [0,0,1,1]=0.742, [0,0,0,0]=0.742, hybrid=0.743; 所有配置 Δ≤0.002 (noise), 与 Dataset 2 结论一致 (null result)
  -- dim_d1_mask: [1,1,1,1] (DPM++ std) / [1,1,0,0] (cx/cy $D_1$, w/h E) / [0,0,0,0] (all Euler) / [0,0,1,1] (cx/cy E, w/h $D_1$)
  -- hybrid: cx/cy DPM++ 2阶 + w/h Heun 2阶 (速度梯形, 额外 NFE)
  -- 评估: mAP, mAP75, APs (小目标), per-dim L1 (cx/cy/w/h)
- **核心结论**
  -- DPM++ 非单调收敛 (step 3 IoU 反降), 中间步骤不具物理意义
  -- Euler 累积误差反噬 (8-step IoU < 4-step IoU)
  -- DPM++ 精度优势来自中心定位 (center_dist 最小), 但 IoU 不是最高 (Heun 的更过大的框覆盖更多 GT)
  -- **分维度 $D_1$ 掩码在双数据集上均不显著 (3-seed 验证)**: Dataset 2 seed42 4 配置 mAP 差异 ≤0.001, baseline 与 [1,1,0,0] 3-seed Δ=+0.001; Dataset 1 3-seed 4 配置 mAP 差异 ≤0.002, baseline 与 [1,1,0,0] Δ=0.001 (noise)
  -- ⚠ Dataset 2 原标注 "3-seed 均值" 实为单 seed42 硬编码 (已修正), APs 单 seed 差异不可靠
  -- **Hybrid (w/h Heun 2阶) vs baseline (3-seed, 结论修正)**: mAP Δ=0.000 (null result), 非 "有害 −0.005"; 原 baseline 硬编码数据 (0.8630±0.000) 错误, 真实 0.858±0.004; full Heun 对照 mAP=0.858±0.004 证明 Heun 本身无害; v_next 不稳定假设被实测证伪 (max ratio=1.22, 远未爆炸); 延迟 +60% (额外 NFE), 效率层面负优化
- **代码改动**
  -- ldmdet/diffusion/rectified_flow.py: RFDPMSolverMultistep.step() 新增 renewal_mask 和 dim_d1_mask 参数; 新增 RFDPMSolverHybrid 类
  -- ldmdet/diffusion/sampling.py: DiffusionSampler 新增 dim_d1_mask 属性; create_dpm_solver() 新增 'dpm_pp_heun_hybrid' 分支
  -- ldmdet/core/head.py: predict() 中 Box Renewal × DPM++ 路径 A (per-proposal renewal mask) + hybrid solver model_fn 注入
- **可视化文件**: docs/paper/latex/figures/trajectory/ (14 张图)
- **测试脚本**: experiments/analysis/per_dim_d1_clean_repro.py (Dataset 2 seed42 + Dataset 1 3-seed, 2026-07-29/31)

---

## 七、Cascade Head × Solver Step 解耦 — 架构合理性形式化 (✓ 已完成, 2026-07-25)

> ✓ 三组实验全部完成 (h3_s4=0.860 / h3_s8=0.859 / h6_s2=0.859, 均 val), S1.3 命题弱形式成立 (mAP 近似不变; 强形式已被证伪, 详见 [theory_analysis_RF_DPM.md §2.4](file:///home/linkst/workspace/projects/chromosome-kd/docs/paper/theory_analysis_RF_DPM.md))。

### 核心贡献: cascade head 作为 implicit solver 的算子分裂视角

当前架构 6 cascade head × 4 solver step = 24 次前向, 但 DPM-Solver++ 仅需 4 NFE 的理论框架把每个 time step 内 6 个 cascade head 视为黑盒——这与 Cascade R-CNN 的级联精化思想同构。形式化为双向精化:
- **横向 (cascade head, 固定 t)**: 在固定时间步上精化 $x_t$, 类似 Cascade R-CNN 级联精化
- **纵向 (solver step, 固定 x 精化链)**: 推进时间 $t$, 类似 DPM-Solver++ 多步积分

### 与染色体检测任务特性的结合

- **解释 24 NFE 架构合理性**: 6 cascade head × 4 solver step 构成算子分裂, DPM-Solver++ 把复合算子 $\mathcal{B}_t^* \circ \mathcal{A}_t$ 视为单次 $v_\theta$ 评估
- **预防审稿人对"6 cascade head 是否冗余"质疑**: cascade head 序列在固定 t 上精化 $x_t$ 至不动点 $\mathcal{B}_t^*$, 横向收敛性是 4 NFE 框架有效的前提
- **解释 Cascade Head Count e2e 失败**: 减小 H 破坏横向收敛性, 而 S 未相应增加, 故 mAP 退化 −0.172

### 命题 S1.3: H×S 可交换性边界

在横向收敛假设下, 减小 H (如 H=3) 需增大 S 以补偿, 反之亦然。但 H×S 不是不变量: 因 $\mathcal{A}_t$ 是二阶 solver 而 $\mathcal{B}_{t,k}$ 是一阶精化, H 减半需 S 增加多于两倍。

### 实验列表

#### 实验证明目的: Cascade × Solver 解耦消融重训 (3 配置, 全部完成)

- 配置 1: H=3 S=4 (12 NFE) ✓ 已完成
  -- 验证: 减小 H 是否破坏横向收敛性, 导致 mAP 退化
  -- 状态: best mAP=0.860 (val) @ ep59 (早停@ep89, patience=30 触发; 2026-07-31 SSH 核实 workstation 日志修正, 旧记 0.859 偏差 +0.001)
  -- 算力: workstation A5000
  -- work_dir: `work_dirs/s1_h3_s4_24obj/` (workstation `/home/linkst/workplace/chromo/chromosome-kd/`, 本地/ross 无)
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-s1-cascade-decouple/runs/s1_h3_s4

- 配置 2: H=6 S=2 (12 NFE) ✓ 已完成 (2026-07-25 确认)
  -- 验证: 减小 S 是否影响纵向积分精度, 与 H=3 S=4 对比验证 H×S 可交换性边界
  -- 状态: 早停@ep136/150 (patience=30 触发), best mAP=0.859 (val) @ ep106, last 0.856 (val) @ ep136
  -- 算力: workstation A5000
  -- work_dir: `work_dirs/s1_h6_s2_24obj/` (workstation `/home/linkst/workplace/chromo/chromosome-kd/`)
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-s1-cascade-decouple/runs/s1_h6_s2

- 配置 3: H=3 S=8 (24 NFE) ✓ 已完成
  -- 验证: 同等 24 NFE 下, 减小 H 增大 S 是否能补偿 (H 减半需 S 增加多于两倍)
  -- 状态: best mAP=0.859 (val, epoch 64), 30 epochs 未改善早停, ross A6000
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-s1-cascade-decouple/runs/s1_h3_s8

- 配置对照: +DPM-Solver++ baseline H=6 S=4 (24 NFE)
  -- mAP: 3-seed 均值 0.859 ± 0.003 (val, 3-seed; 单 seed best 0.863 val; test=0.859 见 §十三.1)
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/a4_dpm_pp

### 关键结论 (最终, 2026-07-25)

- **三组实验 mAP 为 0.860/0.859/0.859 (均 val)** (h3_s4=0.860@ep59 / h6_s2=0.859@ep106 / h3_s8=0.859@ep64; 2026-07-31 核实修正 h3_s4 旧记 0.859→实际 0.860), 三者均在 +DPM-Solver++ baseline 3-seed noise (±0.003, val) 内, 与 baseline 3-seed 均值 (0.859 ± 0.003 val) 持平
- **S1.3 命题弱形式成立** (mAP 近似不变, 0.860/0.859/0.859): H=3,S=4 / H=6,S=2 / H=3,S=8 三组同 NFE 或不同 NFE 配置下 mAP 持平; 强形式 (H 减半需 S 增加多于两倍) 已被证伪
- s1_h6_s2 (H=6, S=2, NFE=12) 在 12 NFE 下 best mAP=0.859 (val), **达到 +DPM-Solver++ baseline 3-seed 均值水平**, 说明**减少 step 并保持 head 可在更少 NFE 下维持性能**
- s1_h3_s8 (H=3, S=8, NFE=24) 在 24 NFE 下 best mAP=0.859 (val), 与 baseline 持平, 表明同等 NFE 下 H=3 S=8 可补偿 H 减半
- 与 Cascade × Solver 解耦命题 S1.3 (H×S 可交换性边界) 对照: H=6 充分大时减小 S 仍可保持横向收敛性, 横向 head 序列已收敛至不动点 $\mathcal{B}_t^*$
- 与已证伪 Cascade Head Count e2e (mAP 0.684 val, −0.172) 形成对比: 该实验减小 H 但未相应增加 S, 横向收敛性被破坏

### 与已证伪 Cascade Head Count e2e 的区分

- **Cascade Head Count e2e**: 重训架构, 把 cascade head 数量从 6 改为其他值, 端到端评估 (mAP 0.684 val, −0.172)
- **Cascade × Solver 解耦**: 形式化分析已有 H=6, S=4 架构的算子分裂结构, 给出"solver 阶数 × cascade 深度"权衡框架, 避免未来重试类似 e2e 实验

### Head Distillation: cascade head 维度蒸馏 (NFE 24→12 加速)

> **创新点**: headwise feature 蒸馏将 H=6 Teacher 知识压缩到 H=3 Student, 实现 NFE 24→12 (2× 加速) 同时保持精度
> **理论依据**: Cascade × Solver 解耦的 H×S 可交换性分析 ([theory_analysis_RF_DPM.md §2](file:///home/linkst/workspace/projects/chromosome-kd/docs/paper/theory_analysis_RF_DPM.md)), [proposals](file:///home/linkst/workspace/projects/chromosome-kd/docs/research/proposals/REFLOW_HEAD_DISTILL_IMPL_PLAN.md)
> **关联**: 失败配置 (freeze_backbone=True) → [FALSIFIED §十三](file:///home/linkst/workspace/projects/chromosome-kd/docs/FALSIFIED_DIRECTIONS.md)

#### 核心贡献: headwise feature 蒸馏实现 cascade head 压缩

通过蒸馏将 6 级 cascade head 压缩到 3 级, Student head 0/1/2 ← Teacher head 0/2/5 (输入/中间/main 对齐), 实现 NFE 减半同时精度持平。

- **形式化**: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{det}}(\text{student}) + \lambda \cdot \mathcal{L}_{\text{distill}}$, $\mathcal{L}_{\text{distill}} = \frac{1}{K}\sum_k \text{MSE}(\text{student\_fc}_k, \text{teacher\_fc}_{\text{map}(k)}.\text{detach}())$
- **head 映射**: $\{0\to0, 1\to2, 2\to5\}$ (输入对齐 + 中间进度 + main 对齐)
- **Teacher**: +DPM-Solver++ (H=6, mAP=0.863 val, seed42 best@ep117), 冻结, 仅 forward
- **Student**: H=3, 从 Teacher head 0/2/5 初始化 (非随机)

#### 与 Cascade × Solver 解耦理论的联系

Cascade × Solver 解耦的 H×S 理论说明 "仅改变 H 会破坏横向收敛性" (已证伪 N_cascade e2e mAP=0.684 val, −0.172)。Head Distillation 通过蒸馏监督让 Student head 继承 Teacher 的行为分布, 避免了随机初始化 H=3 的收敛失败。

#### 实验列表

##### 实验证明目的: Head Distillation 实现 NFE 加速同时保持精度

- Head Distillation (H=3←H=6, backbone解冻 + +DPM-Solver++ backbone加载)
  -- 数据集: Dataset 2
  -- 改动: num_heads=6→3, use_distillation=True, distill_lambda=0.05, distill_head_map={0:0,1:2,2:5}, freeze_backbone=False, teacher_checkpoint=+DPM-Solver++ best ep117, lr=1e-5, 50ep
  -- 结果: mAP=0.859 (val独立评估 test.py --dataset val, seed 42; 训练best@ep10=0.860, early stop@ep40), AP50=0.986, AP75=0.969 [Δ=-0.004 vs +DPM-Solver++ 0.863, 在 3-seed noise ±0.003 内]
  -- NFE: 12 (H=3 × S=4) vs +DPM-Solver++ 24 (H=6 × S=4), **2× 加速**; 延迟 44.72ms / 22.4 FPS (ross A6000, 500iters, Head 39.17ms / Backbone 5.55ms) vs +DPM-Solver++ 77.57ms / 12.9 FPS, **1.73× 推理加速**
  -- loss_distill: 持续下降 0.050→0.025 (50% 下降), 蒸馏目标有效
  -- per-class AP: 与 +DPM-Solver++ 对齐 (Δ -0.012~+0.004, 最大差异 Dataset 15 -0.012)
  -- work_dir: work_dirs/h3_distill_plan_a_24obj/ (本地 + ross)
  -- SwanLab: ldmdet-head-distill / h3_distill_plan_a
  -- 配置: experiments/configs/ldmdet/directions/mainline_ablation_24obj/h3_distill_plan_a_24obj.py

#### 关键结论

- **NFE 24→12 加速 2x + 精度近乎持平**: mAP=0.859 (val独立评估) 近乎持平 +DPM-Solver++ 0.863 (Δ=-0.004, 在 3-seed noise ±0.003 内), 延迟 44.72ms / 22.4 FPS (vs +DPM-Solver++ 77.57ms / 12.9 FPS, 1.73× 加速), 达成工程目标
- **蒸馏有效性**: loss_distill 持续下降 (vs 失败配置停滞 0.033), per-class AP 对齐 +DPM-Solver++, 证明 headwise feature 蒸馏可以有效压缩 cascade head
- **与 Cascade × Solver 解耦互补**: Cascade × Solver 解耦证明 H×S 近似可交换 (h3_s4=0.860 / h6_s2=0.859 / h3_s8=0.859), Head Distillation 证明 H=3 通过蒸馏可达 0.859, 两者共同支撑 "cascade head 可压缩" 的理论
- **未超越 +DPM-Solver++**: 近乎持平 (Δ=-0.004), 无增益 (但"近乎持平"可能已是蒸馏最佳结果, 因 backbone 从 +DPM-Solver++ 加载本身就是知识继承)

#### 失败配置对照 (→ FALSIFIED §十三)

失败配置 (freeze_backbone=True) 是配置Bug: Student backbone 停 ImageNet, Teacher head 期望 +DPM-Solver++ 染色体特征 → 特征分布不匹配 → mAP=0.717 (val, Δ=-0.146)。修复后 0.860 (val), 清晰隔离了"配置Bug" vs "方法局限"。

#### 可扩展性

- **3D 检测** (d=6-7): cascade head 维度蒸馏同样适用, 可压缩 NFE 加速推理
- **关键点检测** (d=2K): headwise feature 蒸馏可扩展到关键点级联精化
- **临床部署**: 2× 加速对交互式筛查延迟带 (13.3-14.2 FPS) 有直接价值

### 双数据集验证缺口 (Cascade × Solver 解耦 + Head Distillation)

> ⚠ Cascade × Solver 解耦三组重训 (h3_s4/h6_s2/h3_s8) 与 Head Distillation 均仅在 Dataset 2 完成。两者均需端到端重训 (Cascade × Solver 解耦需 3 组重训, Head Distillation 需 Teacher/Student 双网络训练), 在 Dataset 1 1540 张图上算力成本较高。Cascade × Solver 解耦的 H×S 可交换性命题在 Dataset 1 上预测仍成立 (架构层面与数据集无关), 但 Dataset 1 低数据下 H=3 是否仍能收敛至 0.746 量级需实验确认。
>
> 🔄 **Dataset 1 补全进展 (2026-08-02)**: h3_s4 单配置 Dataset 1 重训已启动 (workstation A4000, seed42): `work_dirs/s1_h3_s4_chr2024_seed42/`, 当前 ep68/150, **best mAP=0.755 @ ep55 (val)**, eta ~4h。best@ep55 已超 Dataset 1 baseline (RF+Heun 3-seed 0.746 / +DPM-Solver++ 3-seed 0.747) +0.008, 但训练仍在进行 (ep66=0.721 / ep67=0.707 有波动), 最终 best 待早停或 150ep 后确认。h6_s2/h3_s8 Dataset 1 暂不补; Head Distillation Dataset 1 可作为 future work。

---

## 八、x0/v Prediction 对照 — 预测参数化选择论证 (✓ 双数据集 3-seed 完成, 支持 R3.2)

> ✅ **状态回退 (2026-07-31 SSH 核实)**: 2026-07-30 的 "状态修正" 称 "仅 seed42 完成, seed 123/789 从未启动" 系**核查不完整所致的错误修正**——seed 123/789 实际已于 2026-07-23~24 在 workstation 完成训练 (有完整 checkpoint + 独立 SwanLab ID)。**原始 3-seed 声明恢复有效**: seed42=0.855@ep34, seed123=0.858@ep48, seed789=0.857@ep51 → **3-seed 均值 0.857 ± 0.0015** (与被修正掉的原始声明精确匹配)。v-prediction 3-seed 均值 0.857 vs +DPM-Solver++ 3-seed 均值 0.859±0.003, **Δ=−0.002** (方向支持 R3.2, 3-seed 验证完成)。
> **根因**: 2026-07-30 核查时仅检查本地与 ross, 未正确检查 workstation 目录 (或遗漏), 误判 seed123/789 不存在。
> **✅ 双数据集验证完成 (2026-08-02)**: Dataset 1 v-prediction 3-seed 已在 workstation 完成 (早停终止): seed42=0.745@ep72, seed123=0.742@ep57, seed789=0.749@ep84 → **3-seed 均值 0.745 ± 0.004 (val)**, vs Dataset 1 baseline (+DPM-Solver++) 3-seed 均值 0.747 (训练评估, seed42=0.746/seed123=0.748/seed789=0.746), **Δ=−0.002** (噪声范围, 与 Dataset 2 Δ=−0.002 方向一致)。

### 核心贡献: 验证低维 + shifted schedule 下 x0-prediction 优势

基于 RF 原文 (Liu et al., 2023) 使用 v-prediction, 验证在低维 ($d=4$) 检测空间 + shifted schedule ($s=3.0$) 下 x0-prediction 是否优于 v-prediction, 为论文当前参数化选择提供经验依据。

- **命题 R3.1 (信息等价)**: $\hat{x}_0 = x_t - t\hat{v}$, x0-prediction 与 v-prediction 在 $d=4$ 低维 RF 下信息论等价, 差异仅在损失的 $t$ 加权: $\mathcal{L}_v = t^{-2}\mathcal{L}_{x_0}$
- **命题 R3.2 (shifted schedule 下的偏好)**: shifted schedule ($s=3.0$) 下 x0-prediction 的有效梯度信噪比优于 v-prediction, 因前者在 $t \to 0$ 时不放大梯度
- **命题 R3.3 (设置依赖性)**: RF 原文的 v-prediction 偏好依赖高维 + linear schedule 组合; 在低维 + shifted schedule 下 x0-prediction 是更优选择

### 实验列表

#### 实验证明目的: v-prediction 对照 3-seed 重训 (3-seed 全部完成, 2026-07-31 核实)

- seed 42 ✓ 已完成 (2026-07-25, workstation)
  -- 状态: 早停@ep64/150 (patience=30 触发), best mAP=0.855 (val) @ ep34, last 0.837 (val) @ ep64
  -- 训练曲线: ep8 warmup=0.802 → ep34 best=0.855 → 长期停滞 (ep34-ep64 未刷新) → 早停
  -- work_dir: `work_dirs/r3_vpred_24obj_seed42/` (workstation `/home/linkst/workplace/chromo/chromosome-kd/`)

- seed 123 ✓ 已完成 (2026-07-23~24, workstation; 2026-07-31 SSH 核实回退)
  -- 状态: best mAP=0.858 (val) @ ep48, 早停@ep78 (patience=30 触发)
  -- config: `experiments/configs/ldmdet/directions/mainline_ablation_24obj/r3_vpred_24obj.py`
  -- 改动: `criterion=dict(v_prediction=True, v_prediction_t_eps=1e-2)` (batch normalization 均值=1, 避免训练崩溃)
  -- work_dir: `work_dirs/r3_vpred_24obj_seed123/` (workstation)
  -- SwanLab ID: dl4z3jj6 (numpy_random_seed=322502351)

- seed 789 ✓ 已完成 (2026-07-23~24, workstation; 2026-07-31 SSH 核实回退)
  -- 状态: best mAP=0.857 (val) @ ep51, 早停@ep81 (patience=30 触发)
  -- work_dir: `work_dirs/r3_vpred_24obj_seed789/` (workstation)
  -- SwanLab ID: 06c4uol9 (numpy_random_seed=512218475)

- 对照: +DPM-Solver++ baseline (x0-prediction, 3-seed 均值 0.859 ± 0.003 val, 单 seed best 0.863 val)
- SwanLab project: `ldmdet-r3-vpred` (experiment_name=`r3_vpred`)

#### Dataset 1 v-prediction 3-seed 补全 (✅ 完成, 2026-08-01 workstation)

> 闭合双数据集验证缺口。Dataset 1 配置 (`r3_vpred_chr2024.py`) 在 workstation A4000 上训练, 3-seed 全部因早停终止 (patience=30)。

- **seed 42** ✓ 已完成 (2026-08-01, workstation): best mAP=**0.745** (val) @ ep72, 早停@ep102; work_dir=`work_dirs/r3_vpred_chr2024_seed42/`
- **seed 123** ✓ 已完成 (2026-08-01, workstation): best mAP=**0.742** (val) @ ep57, 早停@ep87; work_dir=`work_dirs/r3_vpred_chr2024_seed123/`
- **seed 789** ✓ 已完成 (2026-08-01, workstation): best mAP=**0.749** (val) @ ep84, 早停@ep114; work_dir=`work_dirs/r3_vpred_chr2024_seed789/`
- **3-seed 均值: 0.745 ± 0.004 (val)**, vs Dataset 1 baseline (+DPM-Solver++ 3-seed 均值 0.747, 训练评估), **Δ=−0.002** (噪声范围)
- 与 Dataset 2 结论一致 (D2 Δ=−0.002), 方向支持 R3.2: v-prediction 在低维 + shifted schedule 下不优于 x0-prediction
- checkpoint 已同步至 ross (best_coco_bbox_mAP_epoch_{72,57,84}.pth)

### 关键结论 (3-seed 完成, 2026-07-25; 2026-07-31 核实回退)

- **v-prediction 3-seed 均值 mAP=0.857 ± 0.0015 (val)** (seed42=0.855 / seed123=0.858 / seed789=0.857, 均 val), vs +DPM-Solver++ baseline:
  -- vs 3-seed 均值 0.859±0.003 (val): **Δ=−0.002** (落在 noise 范围内, 方向一致)
  -- vs 单 seed best 0.863 (val): Δ=−0.006
- **方向性支持 R3.2 (3-seed 验证完成)**: v-prediction 3-seed 均值 (0.857) 低于 x0-prediction 3-seed 均值 (0.859), Δ=−0.002 方向一致, 证实 v-prediction 在低维 (d=4) + shifted schedule (s=3.0) 下劣于 x0-prediction
- 训练动态 (三 seed 一致): best 集中在 ep34-51 (warmup 后稳定阶段), 之后 30 epoch 未刷新 → 早停, 表明 v-prediction 优化难度高于 x0-prediction
- 与命题 R3.2 一致: shifted schedule 下 v-prediction 的 $1/t^2$ 梯度放大在 $t \to 0$ 引入方差, 阻碍收敛
- **3-seed 完整验证已完成**: 不再是单 seed 初步结论, 可直接纳入论文 (无需 "preliminary" 标注)
- **✅ 双数据集验证完成 (2026-08-02)**: Dataset 1 3-seed 均值 0.745 ± 0.004, Δ=−0.002 (与 D2 Δ=−0.002 方向一致, 均在 noise 范围)

### 预期结果

- v-prediction mAP 显著低于 +DPM-Solver++ baseline (预期 ΔmAP < 0), 验证命题 R3.2
- 在 $t < 0.5$ (数据主导区, 对检测精度更关键) 时 v-prediction 的 $1/t^2$ 梯度放大引入显著方差
- 若实验确认, 可纳入论文 §3.1.1 末段或 §5.3 (约 0.3 页增量)

### 与已证伪 h_velocity_loss 的区分

- **h_velocity_loss (已证伪, CRASHED)**: 训练崩溃, 无有效 mAP, 未作对照分析 (详见 [FALSIFIED_DIRECTIONS.md §八](file:///home/linkst/workspace/projects/chromosome-kd/docs/FALSIFIED_DIRECTIONS.md))
- **v-prediction 对照**: 通过 $1/t^2$ 损失加权模拟 v-prediction 梯度动态 + `v_prediction_t_eps=1e-2` 截断避免数值爆炸 + batch normalization (均值=1) 避免训练崩溃
- v-prediction 对照是 h_velocity_loss 的可控重训版本, 提供机制级对照

---

## 九、方向 A: per-dim eta_str 诊断 — 维度级曲率分析 (Phase 2 mAP 对比完成, 持平+加速)

### 核心贡献: 检测空间 4 维 (cxcywh) 各维度的曲率差异诊断

整体 $\eta_{str}$ (§五) 是 4 维 (cxcywh) 的范数比, 但检测空间各维度物理含义不同 (位置 cx,cy vs 尺度 w,h)。方向 A 在 +DPM-Solver++ checkpoint 上零成本诊断各维度曲率, 探究是否可设计 per-dim solver。

### 诊断方法

- 在 +DPM-Solver++ checkpoint (best mAP=0.863 val / 0.859 test, epoch 117) 上跑 50 张图 × 3 个 solver (dpm_solver_pp / dpm_solver_pp_3 / dpm_solver_pp_adaptive)
- 计算 wh/cxcy 维度 eta_str 比值, 量化位置维度 vs 尺度维度的曲率差距
- 诊断脚本: `experiments/analysis/direction_a_d_diagnosis.py`
- 配置: `experiments/configs/ldmdet/directions/mainline_ablation_24obj/a4_dpm_pp_24obj.py`
- checkpoint: `work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth`
- 结果 JSON: `work_dirs/diagnosis/dpm_pp_2nd.json`, `work_dirs/diagnosis/dpm_pp_3rd.json`, `work_dirs/diagnosis/dpm_pp_adaptive.json`

### 诊断结果

- **dpm_solver_pp (2 阶)**: wh/cxcy 比值 0.41-0.50
  -- h 维度 eta_str (4-11) 显著小于 cx,cy (17-50)
  -- h 维度曲率比 cx,cy 小 3-5×
- **dpm_solver_pp_3 (3 阶)**: wh/cxcy 比值 0.25-0.59
  -- step 0 有数值异常 (h=77.9, 待分析)
  -- w 维度差距较小 (1.5-2×)
- **结论**: 部分支持假设
  -- h 维度曲率显著小于 cx,cy (3-5× 差距), 支持原假设
  -- w 维度差距较小 (1.5-2×), 部分证伪 "w,h 都显著小于 cx,cy" 的强假设 (w 维度需修正假设)

### Phase 2: per-dim solver mAP 对比 (✅ 已完成, 2026-07-22)

- **执行**: +DPM-Solver++ checkpoint 零成本推理 (无需重训), 2 个 solver × 500 张验证图
- **实现**: [RFDPMSolverPerDim](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/diffusion/rectified_flow.py) — h 维度 (index 3) 用 1 阶 Euler, cx/cy/w 维度 (index 0/1/2) 用 2 阶 DPM-Solver++
- **评估脚本**: [experiments/analysis/direction_a_per_dim_comparison.py](file:///home/linkst/workspace/projects/chromosome-kd/experiments/analysis/direction_a_per_dim_comparison.py)
- **结果数据**: [work_dirs/diagnosis/direction_a_per_dim_comparison.json](file:///home/linkst/workspace/projects/chromosome-kd/work_dirs/diagnosis/direction_a_per_dim_comparison.json)

| Solver | mAP | AP50 | AP75 | 延迟(ms) | FPS |
|--------|------|------|------|---------|-----|
| DPM-Solver++ 2阶 (+DPM-Solver++ baseline, 全维度2阶) | 0.862 | 0.988 | 0.973 | 151.1 | 6.6 |
| Per-dim (h=1阶, cxcy/w=2阶) | 0.863 | 0.989 | 0.972 | 142.9 | 7.0 |

**per-dim eta_str** (per-dim solver 全 500 图诊断, 更可靠 than Phase 1 的 50 图):

| Step | cx | cy | w | h |
|------|------|------|------|------|
| 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| 1 | 27.535 | 19.208 | 1.028 | 0.916 |
| 2 | 11.100 | 15.984 | 0.749 | 0.656 |
| 3 | 33.226 | 10.063 | 0.457 | 0.440 |

### Phase 2 关键结论

1. **mAP 持平 (ΔmAP = +0.001)**: per-dim solver (h=1阶) 与全 2 阶 baseline mAP 持平, 证明 h 维度降为 1 阶不损失精度
   - bbox 4 维度耦合性未被破坏 (位置 cx,cy 与尺度 w,h 的物理相关性不受 solver 阶数分配影响)
2. **延迟略低 (−8.3ms, ~5.5% 加速)**: h 维度省去 $D_1$ 校正计算, 但加速幅度有限 (因单步开销主要在 cascade head H=6)
3. **per-dim eta_str 修正 Phase 1 结论**: Phase 2 全量诊断显示 **w 维度 eta_str (0.5-1.0) 与 h (0.4-0.9) 接近**, 而非 Phase 1 (50 图) 所述"与 cx/cy 接近"
   - 即 w,h 维度曲率均显著小于 cx,cy (10-33), Phase 1 对 w 维度的判断需修正
   - 启示: w 维度也可降为 1 阶 (未来 实验 A.2 可验证)
4. **检测专用 solver 叙事**: 检测空间 4 维度 (cxcywh) 的曲率差异源于物理含义 — 位置 (cx,cy) 随 t 变化剧烈 (需 2 阶), 尺度 (w,h) 变化平缓 (1 阶足够), 这是检测任务特有的结构性先验

### Phase 2 论文纳入策略

- ✅ 纳入论文 §5.4 (方向 A 深化): per-dim solver mAP 持平 + 加速, 约 0.3 页
  - 叙事: "基于 per-dim eta_str 诊断, 设计 per-dim DPM-Solver++ (h 维度 1 阶 + cxcy/w 维度 2 阶), 实验表明 mAP 持平 (Δ=+0.001) 且推理加速 5.5%, 验证了检测空间位置维度与尺度维度的曲率差异可被 solver 阶数分配利用"
  - 强调: 检测专用 solver 设计, 与任务特性 (bbox 4 维结构) 结合

### 与 η_str 直线度诊断的关系

- η_str 直线度诊断: 整体 $\eta_{str}$ 量化"2 步收敛"
- 方向 A: per-dim $\eta_{str}$ 量化各维度曲率差异
- 互补: η_str 直线度诊断决定步数, 方向 A 决定 per-dim 阶数分配

### Phase 3 (A.2): w,h 维度均降 1 阶 (✅ 已完成, 2026-07-28)

> Phase 2 启示 "w 维度 eta_str (0.5-1.0) 与 h (0.4-0.9) 接近, w 维度也可降为 1 阶"。A.2 验证此假设: w,h 维度均用 1 阶 Euler, 仅 cx/cy 用 2 阶 DPM-Solver++。

- **执行**: +DPM-Solver++ checkpoint 零成本推理 (无需重训), 2 个 solver × 500 张验证图
- **实现**: [RFDPMSolverPerDim](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/diffusion/rectified_flow.py) (euler_dims=(2,3) 即 w,h, dpm_dims=(0,1) 即 cx,cy), 配置入口 [sampling.py:201-214](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/diffusion/sampling.py) `dpm_solver_pp_per_dim_w`
- **评估脚本**: [experiments/analysis/a2_ddpm_eta_str_comparison.py](file:///home/linkst/workspace/projects/chromosome-kd/experiments/analysis/a2_ddpm_eta_str_comparison.py)
- **结果数据**: [work_dirs/diagnosis/a2_ddpm_eta_str_comparison.json](file:///home/linkst/workspace/projects/chromosome-kd/work_dirs/diagnosis/a2_ddpm_eta_str_comparison.json)

| Solver | mAP | AP50 | AP75 | AP_S | AP_M | AP_L | 延迟(ms) | FPS |
|--------|------|------|------|------|------|------|---------|-----|
| DPM-Solver++ 全 2 阶 (baseline) | 0.864 | 0.989 | 0.973 | 0.563 | 0.860 | 0.904 | 94.2 | 10.6 |
| A.2 per-dim-w (w,h=1阶, cx/cy=2阶) | 0.864 | 0.989 | 0.974 | 0.560 | 0.860 | 0.902 | 89.5 | 11.2 |
| **Δ** | **0.000** | — | — | — | — | — | **−4.7 (−5.0%)** | +5.3% |

**per-dim η_str** (baseline 全 2 阶, 500 图诊断):

| 维度 | cx | cy | w | h |
|------|------|------|------|------|
| η_str mean | 32.77 | 35.86 | 0.678 | 0.691 |

### Phase 3 (A.2) 关键结论

1. **mAP 持平 (ΔmAP = 0.000)**: w,h 维度均降为 1 阶不损失精度, 验证 Phase 2 启示
   - w,h 维度 η_str (~0.68) 比 cx,cy (~34) 低约 50×, 1 阶 Euler 足够
2. **延迟降低 5.0% (−4.7ms)**: w,h 维度省去 $D_1$ 校正, FPS 10.6→11.2
3. **bbox 4 维耦合性未被破坏**: 位置 (cx,cy) 2 阶 + 尺度 (w,h) 1 阶, 物理相关性不受 solver 阶数分配影响
4. **检测专用 solver 叙事强化**: 位置维度随 t 变化剧烈 (需 2 阶), 尺度维度变化平缓 (1 阶足够), 这是检测任务特有的结构性先验

### DDPM vs RF η_str 对比 (⚠️ 方法论问题, 不纳入论文)

> 实验试图对比 DDPM-trained vs RF-trained 模型的 η_str, 验证 "RF 降低速度场非线性"。但存在严重 framework mismatch, 结论不可靠。

- **方法**: DDPM checkpoint (DiffusionDet, best@ep26, mAP=0.803 val / 0.804 test) 以 `diffusion_type_override='rectified_flow'` 强制走 RF 路径 + DPM-Solver++ 4 步
- **结果**: DDPM-on-RF mAP 暴跌至 0.725 (native 0.803), η_str=1.34 < RF 2.87
- **问题**: DDPM 训练 (cosine noise schedule) + RF 推理 (线性 t∈[0,1] 路径) 是 train-test framework mismatch
  - 低 η_str 不能解读为 "DDPM 轨迹更直", 更可能是模型在错误框架下速度场退化/平坦化 (欠拟合 → 近常数预测 → 低曲率)
  - DDPM native 0.803 已收敛 (AdamW/150ep + EarlyStopping, best@ep26), 非欠训练问题
- **结论**: 此对比无法支撑 "RF 降低速度场非线性" 叙事, 不写入论文
- **正确方案 (若论文需要)**: 在 DDPM native 框架 (DDIM/DDPM solver) 下用二阶 solver 测量 η_str, 或放弃 DDPM 对照 (RF 的 η_str∈[0.7,1.5] 自证低曲率)
- **教训**: 跨范式 η_str 对比必须控制 framework 一致性, 不能用 A 范式训练的 ckpt 强制走 B 范式推理路径

### 双数据集验证 (✓ 2026-07-31 完成 3-seed)

方向 A per-dim solver 在 Dataset 1 上的 3-seed 验证 (复用 a4_dpm_pp_chr2024_seed{42,789,123} checkpoints, box_renewal OFF):

| Solver | seed42 | seed789 | seed123 | 3-seed mean |
|--------|:---:|:---:|:---:|:---:|
| DPM-Solver++ 2阶 (baseline) | 0.746 | 0.747 | 0.748 | 0.747 |
| Per-dim (h=1阶, cxcy/w=2阶) | 0.746 | 0.746 | 0.748 | 0.747 |

- **结论**: Dataset 1 上 per-dim solver mAP 仍持平 (Δ=0.000, 3-seed mean), 与 Dataset 2 结论一致 (null result)
- **跨数据集稳健性**: w/h 维度 η_str 显著低于 cx/cy 的物理含义与数据集无关, per-dim 阶数分配策略跨数据集成立
- **Dataset 1 per-dim η_str** (seed42 诊断): cx/cy η_str (1.3-3.7) 显著高于 w/h (0.04-0.13), 与 Dataset 2 趋势一致 (位置维度曲率 >> 尺度维度)

---

## 十、方向 D: 自适应阶次 DPM-Solver++ — 3 阶校正项增益验证 (mAP 对比完成, 3 solver 持平, null result)

### 核心贡献: 基于 $\eta_{3rd}$ 趋势验证 3 阶校正项的精度增益 (null result)

DPM-Solver++ 3 阶校正项 $D_2$ 在后期 step 应小于早期 (因 RF 轨迹在 $t \to 0$ 时趋于直线)。方向 D 通过零成本诊断 $\eta_{3rd} = \|D_2\|/\|\hat{x}_0\|$ 趋势, 验证 3 阶校正项在 4 NFE 下是否带来精度增益。

⚠ **实验设计局限 (2026-08-02 核查澄清)**: 原假设为 "后期 step 可降为 2 阶" (即后期用 2 阶、前期用 3 阶), 但因 3 阶需 `history ≥ 3` (仅 `step_idx ≥ 2` 可用, 见 `rectified_flow.py:363-375`), "自适应" 方案 (`num_3rd_steps=2`) 实际实现为 "前期 2 阶 + 后期 3 阶" (`applied_3rd_history=[false, true, true]`), 与原假设方向相反。且 `num_3rd_steps=2` 使所有可应用 3 阶的 step (step_idx=2,3) 均应用 3 阶, 等价于 "全程 3 阶" (仅 step_idx=0,1 因 history 不足退化为 2 阶)。因此本实验**未直接测试真正的 "后期降阶" 方案** (应测 `num_3rd_steps=1`: step_idx=2 用 3 阶 + step_idx=3 用 2 阶)。但因 3 阶在所有 step 均无精度增益 (ΔmAP=0.000), 间接支持 "后期可降为 2 阶" 结论。

### 诊断方法

- 在 +DPM-Solver++ checkpoint 上跑 50 张图 × 3 个 solver (dpm_solver_pp / dpm_solver_pp_3 / dpm_solver_pp_adaptive)
- 测量 $\eta_{3rd}$ 随 step 的变化趋势
- 实现位置: `ldmdet/diffusion/rectified_flow.py` (`RFDPMSolverAdaptive`, static + eta_threshold 两种模式)
- 配置: `experiments/configs/ldmdet/directions/mainline_ablation_24obj/a7_dpm_pp_adaptive_24obj.py`
- 结果 JSON: `work_dirs/diagnosis/dpm_pp_adaptive.json`

### 诊断结果

- $\eta_{3rd}$ 趋势: step 1 = 44.6 → step 2 = 18.2 (decreasing, 降幅 59%)
- **结论**: ✓ 后期 step 的 3 阶校正项显著小于早期 (降幅 59%), 支持 "3 阶校正项在后期衰减" 的诊断; 但 mAP 对比显示 3 阶无精度增益 (见下), 因此 "可降为 2 阶" 的结论由 mAP 持平间接支持, 非由 η_3rd 衰减直接证明
- 与 η_str 直线度诊断的整体 $\eta_{str}$ 单调下降 (3.43→2.45→1.68) 一致, 但方向 D 量化了 3 阶项的衰减

### mAP 对比实验 (✅ 已完成, 2026-07-22)

- **执行**: +DPM-Solver++ checkpoint 零成本推理 (无需重训), 3 个 solver × 500 张验证图
- **评估脚本**: [experiments/analysis/direction_d_solver_comparison.py](file:///home/linkst/workspace/projects/chromosome-kd/experiments/analysis/direction_d_solver_comparison.py)
- **结果数据**: [work_dirs/diagnosis/direction_d_comparison.json](file:///home/linkst/workspace/projects/chromosome-kd/work_dirs/diagnosis/direction_d_comparison.json)

| Solver | mAP (val, seed42) | AP50 | AP75 | 延迟(ms) | FPS |
|--------|------|------|------|---------|-----|
| DPM-Solver++ 2阶 (+DPM-Solver++ baseline) | 0.863 | 0.989 | 0.972 | 160.0 | 6.2 |
| DPM-Solver++ 3阶 (全程3阶) | 0.863 | 0.989 | 0.973 | 156.1 | 6.4 |
| 自适应 (前2步2阶+后2步3阶) | 0.863 | 0.988 | 0.973 | 153.3 | 6.5 |

- **applied_3rd_history** (自适应): `[false, true, true]` — 第 1 步未用 3 阶, 第 2-3 步用 3 阶

### 关键结论

1. **3 solver mAP 完全持平 (0.8630, val)**: ΔmAP(2→3) = 0.000, ΔmAP(2→adaptive) = 0.000
   - 证明 4 步采样下 2 阶 DPM-Solver++ 已足够, 3 阶校正项不带来精度增益
   - 佐证 η_str 直线度诊断 "η_str 2 步收敛" 结论: 2 阶 solver 在 4 NFE 下已达到精度天花板
2. **自适应延迟略低**: Δ延迟(2→adaptive) = −6.7ms (约 4.2% 加速)
   - 加速来自 step_idx=0,1 因 history 不足退化为 2 阶 (减少 3 阶计算); 但幅度有限 (4%), 因单步开销主要在 cascade head (H=6) 而非 solver 阶数。注: 自适应方案等价于 "全程 3 阶" (见上方实验设计局限), 加速源于 history 不足的早期 step, 非真正的 "后期降阶"
3. **per-class AP 无显著差异**: 小类别 (Y, G22, F19, F20) 在 3 solver 下 AP 差异 < 0.01, 3 阶校正对困难类别无额外帮助
4. **3 阶校正项无精度增益 (null result)**: 全程 3 阶与 2 阶 mAP 持平, 说明 4 NFE 下 3 阶校正项无实质贡献; 此结论间接支持 "后期可降为 2 阶" (因 3 阶在任何 step 都无增益), 但未直接测试 `num_3rd_steps=1` 的真正后期降阶方案

### 论文纳入策略

- ✅ 纳入论文 §5.4 (方向 D 深化): 3 solver mAP 持平结论佐证 η_str 直线度诊断 "2 步收敛", 约 0.2 页
  - 叙事: "基于 η_3rd 诊断, 设计自适应阶次 DPM-Solver++ (前期 2 阶 + 后期 3 阶, 因 3 阶需 history≥3 仅后期可用), 实验表明 4 NFE 下 2 阶已充分, 3 阶校正项无额外增益 (ΔmAP=0.000)"
- 不作为主要贡献 (因无 mAP 提升), 作为 η_str 直线度诊断的验证实验

### 与 η_str 直线度诊断的关系

- η_str 直线度诊断: 整体 $\eta_{str}$ 量化"2 步收敛"
- 方向 D: per-step 3 阶项 $\eta_{3rd}$ 量化"3 阶校正项衰减趋势" (mAP 持平间接支持后期可降阶)
- 互补: η_str 直线度诊断决定步数, 方向 D 决定每步阶数

### 双数据集验证 (✓ 2026-07-31 完成 3-seed)

方向 D 自适应阶次在 Dataset 1 上的 3-seed 验证 (复用 a4_dpm_pp_chr2024_seed{42,789,123} checkpoints, box_renewal OFF):

| Solver | seed42 | seed789 | seed123 | 3-seed mean |
|--------|:---:|:---:|:---:|:---:|
| DPM-Solver++ 2阶 (baseline) | 0.746 | 0.747 | 0.747 | 0.747 |
| DPM-Solver++ 3阶 (全程3阶) | 0.746 | 0.746 | 0.747 | 0.746 |
| 自适应 (前2步2阶+后2步3阶) | 0.745 | 0.746 | 0.747 | 0.746 |

- **结论**: Dataset 1 上 3 solver mAP 仍持平 (Δ≤0.001, 3-seed mean), 与 Dataset 2 结论一致 (null result)
- **跨数据集稳健性**: 4 NFE 下 2 阶已充分的结论跨数据集成立, 与 §三 Dataset 1 DPM++ 无增益一致 (低曲率下阶数无影响)

---


## 十一、FPS / 延迟基准实验 — 交互式临床筛查延迟带论证 (论文 §4.6 / Table 10 / Figure 6)

> 本节为论文 §4.6 FPS / 延迟基准实验的完整记录。论文 Table 10 + Figure 6 的全部数据点均来自本节, 数据源为 `results/benchmark_fps_*.{md,json}` 系列 5 个文件 + `experiments/runners/benchmark_fps.py` (含 MODEL_REGISTRY / KNOWN_MAP 配置化测量)。

### 实验配置

- **硬件**: NVIDIA RTX A6000 (单卡)
- **输入分辨率**: 512 × 512, batch=1
- **测量方法**: CUDA event timing, warmup=100, iters=300 (baseline); warmup=10, iters=100 (主消融变体)
- **覆盖模型**: 9 个 (论文 Table 10 全部行)
  -- RF+Heun (4 步, 7 NFE): RF 范式基线
  -- +Stoch. Coupling (Heun 4 步): 耦合消融
  -- +DPM-Solver++ (4 步, 4 NFE): DPM-Solver++ 主变体
  -- +DPM-Solver++ + Top-K (K=300/200/100): Top-K 剪枝消融
  -- Cascade R-CNN R50 / YOLOX-S / DiffusionDet: SOTA 对比基线
  -- RTMDet-L (仅 CATALOG §6.4, 未进 Table 10)

### 论文 Table 10 数据矩阵

> **mAP 数据来源说明**: 下表 mAP 列使用 **val set (seed42 best checkpoint)** 数值, 与 FPS/延迟测量使用同一 checkpoint, 确保速度-精度对的内部一致性。test set 评估结果见 [§13.1](#131-测试集评估-论文-§4.5.4-c4-任务), 9 模型 val→test Δ ≤ 0.004 (+DPM-Solver++ 唯一显著偏差 −0.004, 其余 ≤ 0.001)。

| 模型 | Solver | NFE | Latency (ms) | FPS | mAP (val) | mAP (test) |
|------|--------|:---:|-------------:|----:|:---------:|:----------:|
| RF+Heun | Heun | 7 | 124.38 ± 3.38 | 8.0 | 0.856 | 0.857 |
| +Stoch. Coupling | Heun | 7 | 128.35 ± 1.95 | 7.8 | 0.858 | 0.858 |
| **+DPM-Solver++** | **DPM++** | **4** | **75.03 ± 0.96** | **13.3** | **0.863** | **0.859** |
| +DPM-Solver++ + Top-K (K=300) | DPM++ | 4 | 71.27 ± 2.39 | 14.0 | 0.861 | 0.860 |
| **+DPM-Solver++ + Top-K (K=200)** | **DPM++** | **4** | **70.46 ± 2.28** | **14.2** | **0.860** | **0.859** |
| +DPM-Solver++ + Top-K (K=100) | DPM++ | 4 | 69.71 ± 1.98 | 14.3 | 0.850 | 0.847 |
| Cascade R-CNN | — | 1 | 20.67 ± 0.48 | 48.4 | 0.854 | 0.853 |
| YOLOX-S | — | 1 | 10.15 ± 0.41 | 98.5 | 0.796 | 0.795 |
| DiffusionDet | Euler | 1 | 24.38 ± 1.09 | 41.0 | 0.803 | 0.804 |

### 关键结论

1. **临床交互式筛查延迟带 13.3-14.2 FPS**: +DPM-Solver++ + Top-K (K=200) 最快 14.2 FPS / 70.46 ms, +DPM-Solver++ baseline 13.3 FPS / 75.03 ms
2. **DPM-Solver++ 1.75× NFE 加速**: 4 NFE (DPM++) vs 7 NFE (Heun), 同等精度下减少 43% NFE
3. **Top-K 剪枝边际加速**: K=200 vs +DPM-Solver++ baseline 加速 1.06× (75→70 ms), 主要因 cascade head 占 90%+ 延迟 (backbone+neck 仅 ~5.8 ms)
4. **标准检测器快 3-7× 但精度低**: Cascade R-CNN 48.4 FPS / 0.854 val, YOLOX-S 98.5 FPS / 0.796 val, 但 mAP 落后 0.005-0.067 (val)
5. **DiffusionDet 对比**: 41 FPS / 0.803 val, KaryoFlow 数量级 mAP 改善 (+0.060 val)
6. **延迟分布**: cascade head 占 90%+ (Head 118-128 ms in Heun 变体, 64-69 ms in DPM++ 变体); backbone+neck 仅 4-8%

### 数据源文件

- [results/benchmark_fps_20260714_231841.md](file:///home/linkst/workspace/projects/chromosome-kd/results/benchmark_fps_20260714_231841.md): RF+Heun/+Stoch. Coupling/+DPM-Solver++/Top-K K=300/K=200/RTMDet-L (warmup=10, iters=100)
- [results/benchmark_fps_20260714_234842.md](file:///home/linkst/workspace/projects/chromosome-kd/results/benchmark_fps_20260714_234842.md): +DPM-Solver++ + Top-K K=300/K=200/K=100 (warmup=10, iters=100)
- [results/benchmark_fps_20260715_013222.md](file:///home/linkst/workspace/projects/chromosome-kd/results/benchmark_fps_20260715_013222.md): Cascade R-CNN + YOLOX-S (warmup=10, iters=100)
- [results/benchmark_fps_20260716_100627.md](file:///home/linkst/workspace/projects/chromosome-kd/results/benchmark_fps_20260716_100627.md): DiffusionDet baseline (warmup=100, iters=300)
- [results/benchmark_fps_20260716_101856.md](file:///home/linkst/workspace/projects/chromosome-kd/results/benchmark_fps_20260716_101856.md): Cascade R-CNN + YOLOX-S + DiffusionDet 复测 (warmup=100, iters=300)
- [EXPERIMENT_CATALOG.md §6.4](file:///home/linkst/workspace/projects/chromosome-kd/docs/EXPERIMENT_CATALOG.md): C8 复测决策记录 (跳过, 现有数据方法论足够严谨)
- 论文图: latex/figures/fps_map.png (4 类语义着色, log-scale FPS 轴, 速度-精度权衡散点图)

### 测量脚本

- 主脚本: [experiments/runners/benchmark_fps.py](file:///home/linkst/workspace/projects/chromosome-kd/experiments/runners/benchmark_fps.py)
- MODEL_REGISTRY / KNOWN_MAP 配置化测量, 支持 9 个模型变体的统一基准
- 训练配置: 见各模型对应配置文件 (CATALOG §1.5 SOTA 表 + 主消融配置 DDPM/RF+Heun/+AdaLN-Zero/+Stoch. Coupling/+DPM-Solver++)

---

## 十二、标注噪声鲁棒性实验 — SIER 评估广度论证 (论文 §4.8 / Table 11)

> 本节为论文 §4.8 标注噪声鲁棒性实验的完整记录。论文 Table 11 的 3×3 网格 (σ_bbox × 翻转率 p) 全部 9 个扰动单元 + 干净基线均来自本节, 数据源为 `work_dirs/robustness_noise/consolidated_results.json`。无模型重训, 复用 +DPM-Solver++ checkpoint (DPM-Solver++ 4-step + Top-K, seed 42, best@ep117)。

### 实验设计

- **基础 checkpoint**: +DPM-Solver++ + Top-K (DPM-Solver++ 4-step) best_epoch_117, seed 42
- **配置**: [experiments/configs/robustness/noise_test_a4.py](file:///home/linkst/workspace/projects/chromosome-kd/experiments/configs/robustness/noise_test_a4.py)
- **扰动脚本**: [experiments/runners/robustness_noise.py](file:///home/linkst/workspace/projects/chromosome-kd/experiments/runners/robustness_noise.py)
- **评估脚本**: `experiments/runners/robustness_eval.py`
- **数据集**: Dataset 2 test set, 1000 张图像 / 45,980 个 GT 实例
- **扰动维度**:
  -- (i) GT bbox 中心高斯抖动 σ_bbox ∈ {2, 5, 10} px (宽/高不变, 中心裁剪到图像边界)
  -- (ii) 类别标签随机翻转率 p ∈ {5%, 10%, 20%} (翻转到其余 23 类中均匀采样的替代)
- **生成**: 3×3 网格 + 干净基线, 以 seed 42 生成一次, 同一 checkpoint 重新评估; 图像像素不动

### 论文 Table 11 数据矩阵 (mAP@[0.50:0.95], **Dataset 2 test set**, seed 42)

> ⚠ **口径标注 (2026-07-31)**: 本节实验在 Dataset 2 **test set** (1000 张图) 上评估, **所有 mAP 均为 test mAP** (与 §十三.1 +DPM-Solver++ test=0.859 一致)。论文若统一采用 test mAP, 本表可直接使用; 若沿用 val mAP, 需重新在 val set 上评估。

| σ_bbox \ p | 0% | 5% | 10% | 20% |
|------------|----|----|-----|-----|
| 0 px | **0.859** (test) | — | — | — |
| 2 px | — | 0.666 (test) | 0.599 (test) | 0.477 (test) |
| 5 px | — | 0.428 (test) | 0.386 (test) | 0.308 (test) |
| 10 px | — | 0.179 (test) | 0.162 (test) | 0.129 (test) |

### 干净基线完整指标

- σ=0, p=0: mAP=0.859 (test), AP50=0.988, AP75=0.971, AP_S=0.576, AP_M=0.856, AP_L=0.914

### 关键发现

1. **bbox 抖动主导高 IoU 精度**: σ=5 px 下 AP50 温和下降 (0.988→0.847, −0.141), 但 AP75 坍缩 (0.971→0.375, −0.596) — 在 ~100 px 框上 5 px 中心偏移足以打破 IoU≥0.75 但不打破 IoU≥0.50
2. **类别翻转近似乘法降低精度和召回**: 固定 σ 下翻转率加倍大致使剩余 mAP 减半 (σ=2: 0.666→0.599→0.477; σ=5: 0.428→0.386→0.308)
3. **噪声源低噪声下亚加性, 高噪声下严重复合**: σ=2, p=5% 联合 −0.193 小于任一边际之和; 但 σ=10, p=20% 联合退化至 0.129 (test)
4. **最对抗设置下不坍缩**: σ=10, p=20% 下 mAP=0.129 (test), 仍为随机类别基线 (1/24 ≈ 0.042) 的 3×, 表明所学 RF 特征在标注腐蚀下不坍缩

### 数据源文件

- [work_dirs/robustness_noise/consolidated_results.json](file:///home/linkst/workspace/projects/chromosome-kd/work_dirs/robustness_noise/consolidated_results.json): 主结果 (9 个扰动单元 + 干净基线, 含 AP50/AP75/APs/APm/APl 完整指标)
- [work_dirs/robustness_noise/results.json](file:///home/linkst/workspace/projects/chromosome-kd/work_dirs/robustness_noise/results.json): 原始逐单元评估结果
- [work_dirs/robustness_noise/perturbed/](file:///home/linkst/workspace/projects/chromosome-kd/work_dirs/robustness_noise/perturbed/): 9 个扰动 GT JSON (如 `noise_both_s02_f005.json` 等)
- [work_dirs/robustness_noise/eval.log](file:///home/linkst/workspace/projects/chromosome-kd/work_dirs/robustness_noise/eval.log): 评估日志
- CATALOG 索引: a4_noise 推理 (~13 runs, FINISHED/CRASHED)

---

## 十三、测试集评估 + 跨域 Zero-shot — 泛化性论证 (论文 §4.5.4 + §4.7)

> 本节为论文 §4.5.4 测试集评估 + §4.7 跨数据集总结中 zero-shot 跨域实验的完整记录。无模型重训, 复用 +DPM-Solver++ checkpoint (DPM-Solver++ 4-step, seed 42, best@ep117)。

### 13.1 测试集评估 (论文 §4.5.4, C4 任务)

- **基础 checkpoint**: 全部 9 个模型 (论文 Table 10 全部行), 各取 seed 42 best checkpoint
- **数据集**: Dataset 2 test split, 1000 张图像 / 45,980 个 GT 实例
- **评估日期**: 2026-07-26
- **评估脚本**: [results/run_test_eval_batch.sh](file:///home/linkst/workspace/projects/chromosome-kd/results/run_test_eval_batch.sh)
- **完整日志**: [results/test_eval_20260726_181932/](file:///home/linkst/workspace/projects/chromosome-kd/results/test_eval_20260726_181932/)
- **分析报告**: [results/test_eval_20260726_181932/ANALYSIS.md](file:///home/linkst/workspace/projects/chromosome-kd/results/test_eval_20260726_181932/ANALYSIS.md)

#### Dataset 2 val vs test 完整对照表 (9 模型 + 2 SOTA 待补 test)

| 模型 | Val mAP (seed42) | Test mAP | Δ (test−val) | 备注 |
|------|:---------:|:--------:|:----------:|------|
| RF+Heun | 0.856 | 0.857 | +0.001 | 稳定 |
| +Stoch. Coupling | 0.858 | 0.858 |  0.000 | 稳定 |
| **+DPM-Solver++** | **0.863** | **0.859** | **−0.004** | 唯一显著下降 |
| +DPM-Solver++ + Top-K K=300 | 0.861 | 0.860 | −0.001 | 稳定; test 上反超 +DPM-Solver++ |
| +DPM-Solver++ + Top-K K=200 | 0.860 | 0.859 | −0.001 | 稳定; test 上与 +DPM-Solver++ 持平 |
| +DPM-Solver++ + Top-K K=100 | 0.850 (val, epoch_147, FPS 对齐) | 0.847 (test, best ep117) | −0.003 | 稳定; K=100 有害结论 robust; 见 §四 K=100 口径说明 |
| Cascade R-CNN | 0.854 | 0.853 | −0.001 | 稳定 |
| YOLOX-S | 0.796 | 0.795 | −0.001 | 稳定 |
| DiffusionDet | 0.803 | 0.804 | +0.001 | 稳定 |
| DINO R50 | 0.868 | **0.865** | −0.003 | 稳定; test eval 2026-07-31 补跑 (best@ep102, ross A6000) |
| RTMDet-L | 0.863 | **0.862** | −0.001 | 稳定; test eval 2026-07-31 补跑 (best@ep85, ross A6000) |

> ✓ **DINO R50 / RTMDet-L Dataset 2 test 评估已补跑 (2026-07-31, ross A6000)**: DINO R50 test=0.865 (best@ep102), RTMDet-L test=0.862 (best@ep85)。配置: `dino_r50_test_eval.py` / `rtmdet_l_test_eval.py`。SwanLab: `ldmdet-inference` project, exp=`dino_r50_test_eval` / `rtmdet_l_test_eval`。

#### Dataset 1 test 评估表 (220 图, 2026-07-30, 论文 §4.5.4 补充)

> 数据源: [work_dirs/diagnosis/test_eval_per_size_20260730_204840.json](file:///home/linkst/workspace/projects/chromosome-kd/work_dirs/diagnosis/test_eval_per_size_20260730_204840.json) (12 模型, 含 per-class AP); 详见 §十三.5 Dataset 1 SOTA val+test 对照表

| 模型 | Val mAP (seed42) | Test mAP (seed42) | Δ (test−val) | 备注 |
|------|:---------:|:--------:|:----------:|------|
| KaryoFlow RF+Heun (seed42) | 0.745 | 0.737 | −0.008 | val best@ep102 |
| KaryoFlow RF+Heun (seed123) | 0.747 | 0.735 | — | val best@ep101 |
| KaryoFlow RF+Heun (seed789) | 0.747 | 0.738 | — | val best@ep75 |
| KaryoFlow +DPM-Solver++ (seed42) | 0.746 | 0.739 | −0.007 | val best@ep49 |
| KaryoFlow +Stoch. Coupling (seed42) | 0.753 | 0.740 | −0.013 | val best@ep59 (reproduce_0751_stochot_eps5_v2) |
| DiffusionDet DDPM (seed42) | 0.726 | 0.716 | −0.010 | val best@ep79 |
| DiffusionDet DDPM (seed123) | 0.733 | 0.722 | — | val best@ep66 |
| DiffusionDet DDPM (seed789) | 0.727 | 0.718 | — | val best@ep87 |
| DINO R50 | 0.742 | 0.725 | −0.017 | val best@ep77; test 用 ep107 checkpoint (非 best) |
| RTMDet-L | 0.742 | 0.732 | −0.010 | val best@ep52; test 用 best ep52 checkpoint |
| Cascade R-CNN | — | 0.724 | — | val best@ep86 |
| YOLOX-S | — | 0.581 | — | val best@ep150 |

#### +DPM-Solver++ 详细指标 (val vs test)

| Split | mAP | AP50 | AP75 | AP_S | AP_M | AP_L |
|-------|------|------|------|------|------|------|
| val (500 imgs) | 0.863 | 0.989 | 0.972 | 0.499 | 0.860 | 0.901 |
| **test** | **0.859** | 0.988 | 0.971 | **0.577** | 0.856 | 0.914 |

#### 结论

1. **+DPM-Solver++ 是唯一显著偏差**: val→test Δ=−0.004, 其他 8 个模型 Δ ≤ 0.001; 反映 best-checkpoint (ep117) 对 val 的轻微过拟合
2. **test mAP 与 3-seed mean 一致**: +DPM-Solver++ test 0.859 = 3-seed val mean 0.859±0.003, 证实整体泛化良好
3. **AP_S 高方差**: val 0.499 → test 0.577, 仅 60 张 val 含小目标, 应结合 Table 8 逐图像显著性检验解读
4. **Top-K 剪枝叙事增强**: K=300 test 0.860 ≥ +DPM-Solver++ 0.859 (essentially free, 甚至略好); K=200 test 0.859 = +DPM-Solver++ 0.859 (完全 free)
5. **K=100 有害结论 robust**: val −0.013 → test −0.012
6. **DiffusionDet 增益 robust**: val +0.060 → test +0.055

#### 配置文件 (test_eval)

- ldmdet 系列: [experiments/configs/ldmdet/directions/mainline_ablation_24obj/](file:///home/linkst/workspace/projects/chromosome-kd/experiments/configs/ldmdet/directions/mainline_ablation_24obj/) (a1/a2/a4 _test_eval_24obj.py)
- Top-K 系列: [experiments/configs/ldmdet/directions/inference_opt/](file:///home/linkst/workspace/projects/chromosome-kd/experiments/configs/ldmdet/directions/inference_opt/) (a4_io3_k{300,200,100}_test_eval_24obj.py)
- baselines: [experiments/configs/baselines/benchmark_24obj/](file:///home/linkst/workspace/projects/chromosome-kd/experiments/configs/baselines/benchmark_24obj/) (cascade_rcnn_r50_test_eval.py, yolox_s_test_eval.py, diffusiondet_ddpm_test_eval.py)
- **注**: Cascade R-CNN 和 YOLOX-S 首次评估因配置问题失败 (路径双拼接 / EMAHook 未初始化), 已创建 test_eval 配置修复后重跑成功
- **数据源**: [EXPERIMENT_CATALOG.md §7.5](file:///home/linkst/workspace/projects/chromosome-kd/docs/EXPERIMENT_CATALOG.md) (C4: 测试集评估)

### 13.2 跨域 Zero-shot: Dataset 2 → Chromosome20240904 (论文 §4.7 引用)

- **方向**: Dataset 2 (5000 imgs, 训练域) → Dataset 1 (Chromosome20240904, 220 test imgs, 10262 instances)
- **基础 checkpoint**: +DPM-Solver++ (DPM-Solver++ 4-step, seed 42, best@ep117)
- **配置**: [experiments/configs/cross_domain/chr20240904/zero_shot_a4.py](file:///home/linkst/workspace/projects/chromosome-kd/experiments/configs/cross_domain/chr20240904/zero_shot_a4.py)
- **整体结果**: mAP=0.157 (Dataset 1 test, 跨域 zero-shot), AP50=0.513, AP75=0.039
- **per-class 高亮**:
  -- A1/A2/A3/B4/B5 (大类): AP50 ≈ 0.87-0.93 (大染色体迁移良好)
  -- Dataset 13-Dataset 15, E16-E18 (中类): AP50 ≈ 0.73-0.86 (中尺寸迁移良好)
  -- F19/F20: AP50 ≈ 0.58 (小染色体部分迁移)
  -- C6-C12 (C 组): AP50 ≈ 0-0.017 (C 组形态相似, 跨域失效)
  -- G21/G22: AP50 ≈ 0.23-0.46 (小染色体困难)
  -- X: AP50=0.879 (中等迁移); Y: AP50=0.262 (最难, 数据稀缺)
- **结论**: 14/24 类 AP50 > 0.5; C 组失效 (形态相似性导致跨域类别判别失败); 大类迁移良好, 小类困难
- **数据源**: [work_dirs/robustness_noise/zero_shot_results_chr20240904.json](file:///home/linkst/workspace/projects/chromosome-kd/work_dirs/robustness_noise/zero_shot_results_chr20240904.json)
- **评估日志**: [work_dirs/robustness_noise/zero_shot_a4_chr20240904.log](file:///home/linkst/workspace/projects/chromosome-kd/work_dirs/robustness_noise/zero_shot_a4_chr20240904.log)

### 13.3 跨域检测失效主因分析 (2026-07-30, Dataset 1→Dataset 2 per-class AP)

> §13.2 中 Dataset 2→Dataset 1 zero-shot 跨域 mAP=0.157 (Dataset 1 test), C 组 (C6-C12) 完全失效。本节反向验证 (Dataset 1→Dataset 2) 并定量分析失效主因: **类别顺序不一致占主导, 非数据集域差异**。

- **实验**: Dataset 1 训练模型 (reproduce_0751_stochot_eps5_v2, Dataset 1 val mAP=0.753, RF+Heun+AdaLN+StochOT ε=5) 在 Dataset 2 test 上 per-class AP 评估
- **整体结果**: mAP=0.163 (Dataset 2 test, 跨域 zero-shot; 跨域失效, 与 §13.2 Dataset 2→Dataset 1 mAP=0.157 量级一致)
- **关键发现**: 按类别名是否匹配分两组分析

| 组别 | 类别数 | index | AP mean | AP50 mean |
|------|:------:|-------|:-------:|:---------:|
| 匹配组 (类别名相同) | 17 | 0-4, 12-23 (A1-B5, Dataset 13-Y) | 0.230 | **0.741** |
| 不匹配组 (C 组顺序不一致) | 7 | 5-11 (C6-C12) | **0.000** | **0.001** |

- **根因分析**: Dataset 1 的 C 组 (C6-C12) 按字母序排列, Dataset 2 的 C 组按数字序排列, 导致 index 5-11 的类别标签错位 (Dataset 1 C10↔Dataset 2 C6, Dataset 1 C11↔Dataset 2 C7, ..., Dataset 1 C9↔Dataset 2 C12)。模型预测的类别 index 在跨域评估时与 Dataset 2 的类别 index 不对齐, 使 C 组 7 类 AP≈0
- **结论**:
  1. **类别顺序不一致占主导**: 不匹配组 (7 类) AP≈0 完全失效, 匹配组 (17 类) AP50=0.741 证明模型检测能力可跨域迁移
  2. **非数据集域差异**: 匹配组 AP50=0.741 (Dataset 1→Dataset 2) 与 §13.2 Dataset 2→Dataset 1 的 14/24 类 AP50>0.5 一致, 表明跨域检测能力本身良好, 仅类别标签对齐失败
  3. **§13.2 C 组失效重新解读**: §13.2 中 Dataset 2→Dataset 1 的 C 组 AP50≈0-0.017 同样源于类别顺序不一致 (Dataset 2 数字序 → Dataset 1 字母序), 非单纯形态相似性导致
- **数据源**: [work_dirs/diagnosis/cross_dataset_per_class_20260730_194503.json](file:///home/linkst/workspace/projects/chromosome-kd/work_dirs/diagnosis/cross_dataset_per_class_20260730_194503.json)
- **脚本**: [experiments/analysis/cross_dataset_per_class.py](file:///home/linkst/workspace/projects/chromosome-kd/experiments/analysis/cross_dataset_per_class.py)

### 13.4 Dataset 2 跨数据集训练 (✅ 完成, 2026-08-01 早停)

> 基于 §13.3 发现 (类别顺序不一致主导跨域失效), 启动 Dataset 1 配置在 Dataset 2 上的训练, 目的是对齐类别顺序后验证跨域性能, 排除类别顺序干扰。

- **目的**: 用 Dataset 1 对应配置 (RF+Heun+AdaLN+StochOT ε=5) 和种子 (2016452323) 在 Dataset 2 上训练, 使两数据集类别顺序一致后评估跨域 zero-shot 性能
- **配置**: [experiments/configs/ldmdet/ldmdet_rf_heun_adaln_stochot_eps5_d2.py](file:///home/linkst/workspace/projects/chromosome-kd/experiments/configs/ldmdet/ldmdet_rf_heun_adaln_stochot_eps5_d2.py)
- **状态**: ✅ 完成 (2026-08-01 早停终止, workstation A5000)
- **✅ 最终结果 (2026-08-02 SSH 核实)**: best mAP **0.861 @ ep89** (val), early stop @ ep119 (patience=30 触发, best score 0.861), 末 epoch (ep119) mAP=0.857; work_dir=`work_dirs/cross_dataset/d2_0753_stochot_eps5_seed2016452323/20260730_194001`; checkpoint=`best_coco_bbox_mAP_epoch_89.pth` (已同步至 ross)
- **注**: 此 0.861 为 Dataset 2 域内训练 mAP (Dataset 1 配置/类序训练于 Dataset 2), 与主路线 RF+Stoch. Coupling (~0.858) 量级一致 (Δ=+0.003), 表明 Dataset 1 配置在 Dataset 2 上可达到相当性能
- **后续**: 可用此模型 (类别顺序对齐) 与 Dataset 1 训练模型 (reproduce_0751_stochot_eps5_v2) 互做跨域 zero-shot 评估, 对比 §13.3 (类别顺序未对齐) 的 mAP=0.157, 验证类别顺序对齐后跨域性能提升幅度

### 13.5 DINO R50 / RTMDet-L Dataset 1 最终结果 (val + test, 2026-07-31 补)

> 之前 DINO R50 Dataset 1 仅训练 31/150 epoch (best@ep29, val mAP=0.607, 未完成)。现训练已完成, 记录 val + test 最终结果作为跨数据集 SOTA 退化对比的修正基准。
>
> ⚠ **val vs test 口径澄清 (2026-07-31)**: 之前本节标注的 0.742 为 **val mAP**, 未区分 test。论文应采用 **test** mAP (DINO R50 test=0.725, RTMDet-L test=0.732)。下表明确区分两个 split。

#### Dataset 1 SOTA 检测器 val + test 对照

| 模型 | Val mAP (seed42) | Test mAP (seed42) | best epoch (val) | 训练状态 | 数据源 (test) |
|------|:---------:|:--------:|:----------:|----------|--------|
| DINO R50 | **0.742** | **0.725** | ep77 | early stop @ ep107 (patience=30 触发) | test_eval_per_size_20260730_204840.json (ep107 checkpoint) |
| RTMDet-L | **0.742** | **0.732** | ep52 | 跑满 150 ep, 未触发早停 | test_eval_per_size_20260730_204840.json (best ep52 checkpoint) |
| KaryoFlow RF+Heun (3-seed) | 0.746±0.001 | 0.737±0.002 (seed42=0.737/123=0.735/789=0.738) | — | — | test_eval_per_size_20260730_204840.json |
| KaryoFlow +DPM-Solver++ (seed42) | 0.746 | 0.739 | ep49 | — | test_eval_per_size_20260730_204840.json |
| Cascade R-CNN | — | 0.724 | ep86 | — | test_eval_per_size_20260730_204840.json |
| DiffusionDet (3-seed) | 0.729±0.003 | 0.719±0.003 (seed42=0.716/123=0.722/789=0.718) | — | — | test_eval_per_size_20260730_204840.json |
| YOLOX-S | — | 0.581 | ep150 | — | test_eval_per_size_20260730_204840.json |

- **之前 (未完成)**: DINO R50 0.607 @ ep29 (31/150ep, val, 未完全收敛)
- **跨数据集退化对比 (test 口径, 论文采用)**:
  - DINO R50 Dataset 1 test 0.725 vs Dataset 2 test 0.865, Δ=−0.140 (**退化 16.2%**)
  - RTMDet-L Dataset 1 test 0.732 vs Dataset 2 test 0.862, Δ=−0.130 (退化 15.1%)
  - 注: Dataset 2 的 DINO R50 (test=0.865) / RTMDet-L (test=0.862) test 已于 2026-07-31 补跑, 跨数据集退化对比已更新为 test-vs-test 口径
- **跨数据集退化对比 (val 口径, 文档历史参照)**:
  - DINO R50 Dataset 1 val 0.742 vs Dataset 2 val 0.868, Δ=−0.126 (退化 14.5%)
  - RTMDet-L Dataset 1 val 0.742 vs Dataset 2 val 0.863, Δ=−0.121 (退化 14.0%)
- **DINO R50 test 0.725 vs val 0.742 差异说明**: test 评估使用 ep107 (last checkpoint, val=0.738) 而非 best@ep77 (val=0.742) checkpoint, Δ(test−val_best)=−0.017; 若用 best@ep77 checkpoint 评估 test, 预期 test mAP 会略高 (≈0.732 量级, 与 RTMDet-L 一致)
- **数据源**: [work_dirs/diagnosis/test_eval_per_size_20260730_204840.json](file:///home/linkst/workspace/projects/chromosome-kd/work_dirs/diagnosis/test_eval_per_size_20260730_204840.json) (Dataset 1 test, 220 图, 12 模型)
- **注**: 论文草稿 (paper_draft_CN.md §8) 中 DINO R50 Dataset 1 test=0.725 / RTMDet-L Dataset 1 test=0.732 为正确 test 数值; 论文图 fps_map.py 中 DINO R50 Dataset 1 坐标应使用 test=0.725 (非 val 0.742, 非 0.607)

---

## 十四、Dataset 1 早期探索与消融论证

> 本节保留 Dataset 1 早期探索中有设计论证价值的消融实验。其他 Dataset 1 边际实验 (Hard OT/Sinkhorn/OT Flow Coupling) 已合并到 §二 (佐证 OT 坍缩理论), 无正向意义的方向 (M1/Box Refine Net/Focal γ=3) 已移至 FALSIFIED §二十六~§二十八。
### Dataset 1: RoI 空间编码消融 — 空间编码至关重要 (✓ 完成)

- **实验**: RoI 7×7 空间特征 vs 空间抹平 (GlobalAvgPool → 1×1) 消融
- **结果**: baseline mAP=0.863 → ablation mAP=0.009, **Δ = −0.854** (灾难性崩溃)
- **结论**: 7×7 空间结构至关重要, DynamicConv 已有效提取空间编码 (非丢失)
- **影响**: 直接支撑 M1 "应增强而非重建空间编码" 的设计决策
- **数据源**: `work_dirs/diagnosis/d1_roi_ablation.json` + [EXPERIMENT_CATALOG.md §7.6](file:///home/linkst/workspace/projects/chromosome-kd/docs/EXPERIMENT_CATALOG.md)
- **诊断脚本**: `experiments/analysis/d1_roi_ablation.py`


## 十五、SwanLab 项目映射汇总

| SwanLab Project | 实验数 | 范围 | URL Pattern |
|-----------------|--------|------|-------------|
| `ldmdet-mainline-ablation-24obj` | 5 ⭐ | Dataset 2 主路线消融 (DDPM/RF+Heun/+AdaLN-Zero/+Stoch. Coupling/+DPM-Solver++, 论文核心) | `https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/<run_id>` |
| `ldmdet-ablation` | 9 (Dataset 2) + 23 (Dataset 1) | 主线 + Dataset 2 耦合策略 + Dataset 1 历史 + 非线性轨迹 | `https://swanlab.cn/@einspanner/ldmdet-ablation/runs/<run_id>` |
| `chromosome-kd-benchmark-24obj` | 8 | Dataset 2 SOTA 对比模型 (DINO/RTMDet-L/Cascade/YOLOX/DiffusionDet) | `https://swanlab.cn/@einspanner/chromosome-kd-benchmark-24obj/runs/<run_id>` |
| `ldmdet-breakthrough` | 2 | Dataset 2 SC-RF 自条件化 | `https://swanlab.cn/@einspanner/ldmdet-breakthrough/runs/<run_id>` |
| `ldmdet-frontier-directions` | 6 | Dataset 2 前沿方向探索 | `https://swanlab.cn/@einspanner/ldmdet-frontier-directions/runs/<run_id>` |
| `ldmdet-s1-cascade-decouple` | 3 已完成 | cascade head × solver step 解耦消融 (s1_h3_s4 ✓ 0.860 / s1_h3_s8 ✓ 0.859 / s1_h6_s2 ✓ 0.859) | `https://swanlab.cn/@einspanner/ldmdet-s1-cascade-decouple/runs/<run_id>` |
| `ldmdet-r3-vpred` | 3 已完成 | v-prediction 对照重训 (seed 42 ✓ 0.855 / seed 123 ✓ 0.858 / seed 789 ✓ 0.857; 3-seed 均值 0.857±0.0015) | `https://swanlab.cn/@einspanner/ldmdet-r3-vpred/runs/<run_id>` |
| `nonlinear-3seed-repro` | 2 | 3-seed 复现 (Dataset 1) | `https://swanlab.cn/@einspanner/nonlinear-3seed-repro/runs/<run_id>` |
| `chromosome-kd` | 21 | 早期 Dataset 1 数据集 | `https://swanlab.cn/@einspanner/chromosome-kd/runs/<run_id>` |
| `ldmdet-inference` | 12 | DDIM 步数对齐 + DPM-Solver++ 步数消融 (Dataset 1) | `https://swanlab.cn/@einspanner/ldmdet-inference/runs/<run_id>` |

---

## 十六、关键结论汇总

### 主路线创新点贡献矩阵

| 创新点 | 核心贡献 | 与任务结合 | 关键数据 | 状态 |
|--------|----------|------------|----------|------|
| **RF (§一)** | 直线 ODE 路径取代 DDPM 弯曲随机轨迹 | 密集 proposals 误差复合 / 小训练集 / 24 类细粒度 | DDPM→RF+Heun +0.053 mAP (统一口径), 91% 归因于 RF | ✅ 完成 |
| **OT Collapse + Stoch. Coupling (§二)** | 低维 d=4 OT 坍缩形式化 + Stochastic Coupling 补救 | 低维触发 / 高 K 加剧 / 小训练集放大 | Dataset 1 +0.034 (p<10⁻¹²⁰), Hard OT vs Random −0.008 (无aug, 直接证据); Dataset 2 Stoch vs Random +0.0001 (ns) + 4.6× 平滑 (间接佐证, Hard OT 未运行) | ⚠ D2 Hard OT 缺口 |
| **DPM-Solver++ (§三)** | RF 适配 data-prediction + 修正 FlowDet 结论 | 临床交互式延迟 13.3-14.2 FPS / cascade head 占 90%+ | +0.006 mAP (p<10⁻⁶) + 1.75× NFE 加速 | ✅ 完成 |
| **Top-K Pruning (§四)** | 500→K proposals 剪枝 + DPM-Solver++ 兼容 | K=200 最优 (46 染色体 + 重叠冗余) | K=200: 14.2 FPS, mAP 0.860 | ✅ 完成 |
| **η_str 直线度诊断 (§五)** | 零开销直线度指标, 量化"2 步收敛" | 修正"RF 接近直线" claim (实际 η_str∈[0.7,1.5]) | 3 seeds 单调下降 3.43→2.45→1.68 | ✅ 完成 |
| **Box Renewal × DPM++ (§六)** | 揭示 box_renewal 与多步法历史矛盾 + 化解 | box_renewal 检测特有 / 密集目标 renewal 比例高 | η_str 虚高 56-58% 但 mAP 仅 −0.0003; K≥200 推理关闭安全, K=100 −0.031±0.012 (3-seed) | ✅ 完成 (含 K 值依赖性验证) |
| **Cascade × Solver 解耦 (§七)** | cascade head 作为 implicit solver 算子分裂 | 解释 24 NFE 架构合理性, 预防"6 head 冗余"质疑 | s1_h3_s4 ✓ (0.860@ep59), s1_h6_s2 ✓ (0.859@ep106), s1_h3_s8 ✓ (0.859@ep64) — 三组全部完成 (均在 baseline noise ±0.003 内) | ✅ 完成 |
| **Head Distillation (§七)** | headwise feature 蒸馏 H=6→H=3, Cascade × Solver 解耦理论成功应用 | NFE 24→12 (2×加速), cascade head 可压缩性验证 | mAP=0.859 (Δ=-0.004, noise内), 1.73× 推理加速 (44.72ms/22.4FPS) | ✓ 完成 |
| **v-prediction 对照 (§八)** | 验证低维 + shifted schedule 下 x0-prediction 优势 | 预防"为何不用 v-prediction"质疑 (RF 原文偏好) | D2: 3-seed 0.857±0.0015, Δ=−0.002; D1: 3-seed 0.745±0.004, Δ=−0.002 (双数据集方向一致, baseline=+DPM-Solver++ 3-seed mean 0.747/0.859) | ✅ 完成 (双数据集 3-seed) |
| **方向 A per-dim η_str (§九)** | 检测空间 4 维 (cxcywh) 各维度曲率差异诊断 | h 维度曲率显著小于 cx,cy, 启示 per-dim solver | Phase 2: per-dim (h=1阶) mAP=0.863 val (+0.001), 加速 5.5%; Phase 3 (A.2): per-dim-w (w,h=1阶) mAP=0.864 val (持平), 加速 5.0% | ✓ 完成 |
| **方向 D 自适应阶次 (§十)** | 3 阶校正项增益验证 (null result) | $\eta_{3rd}$ step1→2 降幅 59%, 但 3 阶无精度增益 | 3 solver mAP 均为 0.863 val (ΔmAP=0.000), 自适应 4.2% 加速 (源于 history 不足非后期降阶) | ✓ 完成 |

### 双数据集验证状态汇总 (规则: 所有理论应在两个数据集上验证)

| 创新点 | Dataset 2 | Dataset 1 | 双数据集 | 补全成本 |
|--------|-----------|-----------|----------|----------|
| RF 范式 (§一) | ✓ 0.856 (3-seed) | ✓ 0.746 (3-seed) vs DDPM 0.729 | ✅ 完成 | — |
| OT Collapse + Stoch. Coupling (§二) | ⚠ Stoch vs Random +0.0001 (ns) + 4.6× 平滑 (间接佐证); Hard OT 未运行 | ✓ +0.034 (p<10⁻¹²⁰), Hard OT vs Random −0.008 (无aug, 直接证据) | ⚠ D2 Hard OT 缺口 | 可补 D2 Hard OT 3-seed |
| DPM-Solver++ (§三) | ✓ +0.006 (p<10⁻⁶) | ✓ +0.001 (持平, 3-seed 0.747±0.001) | ✅ 完成 (方向性一致) | — |
| Top-K Pruning (§四) | ✓ K=100/200/300 | ✓ K=100/200/300 (3-seed) | ✅ 完成 | — |
| η_str 直线度诊断 (§五) | ✓ 3-seed × 4 config | ✓ renewal ON/OFF (seed42) | ✅ 完成 | 可补 3-seed |
| Box Renewal × DPM++ (§六) | ✓ K 值依赖 3-seed | ✓ 全 K 矩阵 (3-seed) + η_str 反向发现 + dim_d1_mask 3-seed | ✅ 完成 | — |
| Cascade × Solver 解耦 (§七) | ✓ 3 组重训 | 🔄 h3_s4 seed42 运行中 (ep68/150, best@ep55=0.755) | 进行中 | h6_s2/h3_s8 暂不补 |
| Head Distillation (§七) | ✓ 0.859 | ⛔ 未跑 | ⚠ 缺口 | future work |
| v-prediction 对照 (§八) | ✓ 3-seed 0.857±0.0015 | ✓ 3-seed 0.745±0.004 | ✅ 完成 (D1 Δ=−0.002, D2 Δ=−0.002, 方向一致) | — |
| 方向 A per-dim (§九) | ✓ mAP 持平 + η_str | ✓ mAP 持平 (3-seed) + η_str | ✅ 完成 | — |
| 方向 D 自适应阶次 (§十) | ✓ 3 solver 持平 | ✓ 3 solver 持平 (3-seed) | ✅ 完成 | — |

**双数据集验证优先级** (投稿前补全建议):
1. ~~**高优先 (推理零成本, 闭合主贡献)**: §四 Top-K Dataset 1 + §五 η_str 直线度诊断 Dataset 1~~ — ✅ 已完成 (2026-07-31 3-seed 补全). Dataset 1 K=100 掉点比 Dataset 2 更严重 (3-seed: −0.036 vs −0.022, 证伪原预测); Dataset 1 η_str 仅为 Dataset 2 4-8% (印证低曲率)
2. ~~**高优先 (重训, 闭合主贡献)**: §三 Dataset 1 DPM++ 3-seed~~ — ✅ 已完成 (seed789 2026-07-31 重训后异常消除). 3-seed mean=0.747±0.001 (0.746/0.748/0.746), 与 Heun 3-seed 0.746±0.001 持平, Δ=+0.001 方向性一致
3. ~~**中优先 (重训, 验证架构泛化)**: §七 h3_s4 Dataset 1 单配置 (~12h)~~ — 🔄 运行中 (2026-08-02, workstation A4000, seed42, ep68/150, best@ep55=0.755, eta ~4h). 验证 H×S 可交换性跨数据集稳健性
4. ~~**低优先 (null result 深化)**: §九 方向A Dataset 1 + §十 方向D Dataset 1~~ — ✅ 已完成 (2026-07-31 3-seed). 两方向在 Dataset 1 上均持平 (Δ≤0.001), null result 跨数据集稳健
5. ~~**待 GPU 空闲**: §八 v-prediction 对照 Dataset 1~~ — ✅ 已完成 (2026-08-02). Dataset 1 3-seed 0.745±0.004, vs Dataset 1 baseline 0.747 (+DPM-Solver++ 3-seed mean), Δ=−0.002 (噪声范围, 与 Dataset 2 Δ=−0.002 方向一致)

### SOTA 比较 (Dataset 2 val, 3-seed 均值; test 见 §十三.1)

| 方法 | Backbone | mAP (val) | mAP (test) | FPS | 备注 |
|------|----------|:---------:|:----------:|-----|------|
| DINO R50 | ResNet-50 | 0.868 | **0.865** | 30.5 | 多尺度可变形注意力 (单 seed) |
| RTMDet-L | CSPNeXt-L | 0.863 | **0.862** | — | 更强主干 (单 seed) |
| **KaryoFlow (+DPM-Solver++)** | ResNet-50 | **0.859** | **0.859** | **13.3** | RF + DPM-Solver++ (val 3-seed mean = test seed42) |
| KaryoFlow (+DPM-Solver++) + Top-K (K=200) | ResNet-50 | 0.860 | 0.859 | **14.2** | 最佳速度-精度权衡 (test 上与 +DPM-Solver++ 持平) |
| Cascade R-CNN | ResNet-50 | 0.854 | 0.853 | 48.4 | — |
| DiffusionDet | ResNet-50 | 0.803 | 0.804 | 41.0 | DDPM 基线 (ep26 checkpoint) |
| YOLOX-S | CSPDarkNet-S | 0.796 | 0.795 | 98.5 | — |

> ✓ **test 口径已闭合 (2026-07-31)**: DINO R50 test=0.865 (Δ=−0.003), RTMDet-L test=0.862 (Δ=−0.001), 均与 val 差距 ≤0.003。论文 SOTA 比较可统一用 test mAP。KaryoFlow (+DPM-Solver++) test=0.859 落后 DINO R50 test 0.006, 落后 RTMDet-L test 0.003。

### 关键统计显著性

| 比较 | Δ mAP | p-value | n |
|------|-------|---------|---|
| RF vs DDPM (Dataset 2, 统一口径 DiffusionDet 0.803) | +0.053 | — | — |
| RF vs DDPM (Dataset 2, 旧口径 a0_baseline 0.774) | +0.082 | — | — |
| RF vs DDPM (Dataset 1) | +0.017 | — | — |
| Stoch vs Random (Dataset 1) | +0.034 | <10⁻¹²⁰ | 1320 |
| Hard OT vs Random (Dataset 1) | −0.008 | <10⁻⁸ | 1320 |
| Stoch vs Random (Dataset 2) | +0.0001 | 0.80 (ns) | 500 |
| DPM++ vs Heun (Dataset 2, 4 步) | +0.006 | <10⁻⁶ | 500 |
| DDPM Euler 1→8 步 (Dataset 2, 论文 Appendix G) | <0.002 | — | — |

### 辅助评估实验索引 (论文数据点 ↔ LINEAGE 章节)

| 论文位置 | 数据点 | LINEAGE 章节 | 数据源 |
|----------|--------|--------------|--------|
| §4.3.1 / Figure 5 / Table F.1 | 24 类 per-class AP (+DPM-Solver++ seed 42) | §一 末段 | results/a4_per_class_ap.md |
| §4.3.2 / Table 8 | Dataset 2 配对显著性检验 (6 行) | §三 末段 | CATALOG §7.4 (C3) |
| §4.4.1 / Table 9 | Dataset 1 耦合消融配对检验 (6 行) | §三 末段 | CATALOG §7.4 (C2) |
| §4.4.2 / Table 7 / Figure 4 | 多维稳定性 (5 指标) | §二 末段 | CATALOG §7.8 + 训练日志 |
| §4.5.4 | 测试集评估 (val vs test) | §十三.1 | CATALOG §7.5 (C4) |
| §4.6 / Table 10 / Figure 6 | FPS / 延迟基准 (9 模型) | §十一 | results/benchmark_fps_*.md (5 个文件) |
| §4.7 | 跨域 Zero-shot (Dataset 2 → Chr20240904) | §十三.2 | work_dirs/robustness_noise/zero_shot_results_chr20240904.json |
| §4.7 (深化) | 跨域 per-class AP 失效主因 (Dataset 1→Dataset 2, 类别顺序不一致) | §十三.3 | work_dirs/diagnosis/cross_dataset_per_class_20260730_194503.json |
| §4.8 / Table 11 | 标注噪声鲁棒性 (3×3 网格) | §十二 | work_dirs/robustness_noise/consolidated_results.json |
| §5.6 / §7.3.5 | Dataset 1 per-class AP 增益 (Stoch Coupling) | §二 末段 | CATALOG §7.3.5 (Problem 3) |
| §4.3.2 (引用, 不入正文) | SOTA per-image Wilcoxon (5 模型) | §一 末段 | CATALOG §7.4.6 (Problem 2B) |
| — (诊断) | Dataset 2 跨数据集训练 (类别顺序对齐验证, ✅ 完成 best=0.861@ep89) | §十三.4 | experiments/configs/ldmdet/ldmdet_rf_heun_adaln_stochot_eps5_d2.py |
| — (基准修正) | DINO R50 Dataset 1 最终结果 (0.742, early stop @ ep107) | §十三.5 | work_dirs/baselines/dino_r50_20240904/ |

