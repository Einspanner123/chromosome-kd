# LDMDet 项目结构重构方案

> 日期：2026-06-14 | 状态：方案 | 执行：subagent-driven-development 逐任务推进
>
> **目标**：将 LDMDet 从松散的 mmdet 耦合项目重构为层次清晰的工程结构。**核心层（ldmdet）纯 PyTorch 零 mmdet 依赖**，实验层（experiments）薄封装 mmdet 作为运行框架。

---

## 一、现状诊断

### 当前结构问题

| 问题 | 现状 | 影响 |
|------|------|------|
| 核心代码混入 mmdet 注册 | `model.py` 是 `BaseDetector` 子类，但 `mods/` 内模块已被注册到 `MODELS` | 纯 PyTorch 能力被隐藏，不方便单独测试 |
| 配置平铺且耦合 | configs 全在一层，通过 `_base_` 继承但无分类 | 找配置困难，消融对比不直观 |
| 多 seed 无系统支持 | 手动复制 config 改 seed | 效率低，易出错，不可复现 |
| 测试缺乏规范 | 测试在 `LDMDet/tests/` 中 ad-hoc，无 CI 友好结构 | 每次改完不跑测试 |
| 推理测试无独立入口 | 要跑 test 需要 mmdet 完整环境 | 不方便快速验证 checkpoint |
| 分析工具分散 | `tools/analysis/` 存在但不够规范 | 论文需要的结果不全 |

### 当前结构

```
chromosome-kd/
├── LDMDet/
│   ├── model.py           # mmdet detector wrapper
│   ├── hooks.py           # mmdet hooks (CopyProject, Vis, WeightSummary)
│   ├── mods/              # 核心模块 (纯PyTorch + MODELS注册)
│   │   ├── diffusiondet_head.py
│   │   ├── single_head.py
│   │   ├── ot_coupling.py
│   │   ├── sampling.py
│   │   ├── rectified_flow.py
│   │   ├── criterion.py / loss.py / losses.py / matcher.py / costs.py
│   │   ├── roi_extractor.py
│   │   ├── structures.py / utils.py / embeddings.py
│   │   └── ...
│   ├── configs/           # mmdet config py files
│   │   ├── ldmdet_baseline.py
│   │   ├── ldmdet_rf_heun_shifted_bs2.py
│   │   ├── ablation/
│   │   ├── benchmark/
│   │   ├── benchmark_24obj/
│   │   ├── recipes/
│   │   └── _legacy/
│   ├── tests/             # ad-hoc 测试
│   ├── tools/
│   │   ├── analysis/
│   │   └── ...
│   ├── results/           # JSON 结果
│   └── scripts/
├── projects/LDMDet/docs/from_experiments/
│   └── MASTER_TIMELINE.md
└── work_dirs/
```

---

## 二、目标结构

```
chromosome-kd/
├── ldmdet/                    # ★ 纯 PyTorch 库 — ZERO mmdet 依赖
│   ├── __init__.py            # 公开 API: from ldmdet import DiffusionDetHead, ...
│   ├── version.py             # __version__
│   ├── core/                  # 核心模型组件
│   │   ├── __init__.py
│   │   ├── head.py           # DiffusionDetHead (主入口)
│   │   ├── single_head.py    # SingleDiffusionDetHead
│   │   ├── roi_extractor.py  # SingleRoIExtractor
│   │   └── dynamic_conv.py   # DynamicConv
│   ├── diffusion/             # 扩散范式 (RF, DDPM)
│   │   ├── __init__.py
│   │   ├── rectified_flow.py # RF forward/reverse
│   │   ├── noise_schedule.py # Cosine/linear schedules
│   │   ├── embeddings.py     # SinusoidalPositionEmbeddings
│   │   └── sampling.py       # Euler, Heun, DPM-Solver++, DDIM
│   ├── coupling/              # ★ 耦合策略 (一等公民，独立可测)
│   │   ├── __init__.py       # build_coupling(name, **kwargs)
│   │   ├── base.py           # CouplingStrategy ABC
│   │   ├── random.py         # RandomCoupling
│   │   ├── hard_ot.py        # HardOTCoupling (nearest-neighbor)
│   │   ├── sinkhorn_argmax.py # SinkhornArgmaxCoupling
│   │   ├── sinkhorn_stochastic.py # SinkhornStochasticCoupling
│   │   ├── ghss.py           # GroupHierarchicalStochasticSampling
│   │   └── _sinkhorn_ops.py  # 共享 Sinkhorn 迭代实现
│   ├── criterion/             # 损失与匹配
│   │   ├── __init__.py
│   │   ├── criterion.py      # DiffusionDetCriterion
│   │   ├── matcher.py        # SimOTA-style matcher
│   │   ├── losses.py         # FocalLoss, L1Loss, GIoULoss
│   │   └── costs.py          # Match costs
│   ├── data/                  # 数据结构 (纯 tensor dataclass)
│   │   ├── __init__.py
│   │   ├── structures.py     # DetectionResult, InstanceData, ImageMeta, ModelOutput
│   │   └── transforms.py     # custom_transforms (content injection etc.)
│   └── utils/                 # 工具
│       ├── __init__.py
│       ├── box_ops.py        # bbox_xyxy_to_cxcywh, bbox_cxcywh_to_xyxy, bbox2roi
│       ├── constants.py      # CHROMO_GROUP_OF_CLASS 等常量
│       └── entropy.py        # 速度熵 / 方差分解 (论文分析工具)
│
├── experiments/               # mmdet 依赖层
│   ├── __init__.py
│   ├── mmdet_bridge/          # ★ 薄桥: ldmdet ↔ mmdet
│   │   ├── __init__.py
│   │   ├── detector.py        # LDMDetDetector(BaseDetector): 包装 ldmdet 核心
│   │   ├── hooks.py           # CopyProjectHook, VisHook, WeightSummaryHook
│   │   └── registry.py        # 注册到 MODELS (一次性集中注册)
│   ├── configs/               # mmdet 配置 (清晰分层继承)
│   │   ├── _base_/
│   │   │   ├── default_runtime.py    # 通用运行时 (optimizer, scheduler, hooks)
│   │   │   ├── dataset_24chromo.py   # 24类染色体数据集
│   │   │   └── dataset_single_chromo.py
│   │   ├── baselines/               # 标准检测器基线
│   │   │   ├── cascade_rcnn_r50.py
│   │   │   ├── dino_r50.py
│   │   │   ├── rtmdet_l.py
│   │   │   └── diffusiondet_ddpm.py
│   │   ├── ldmdet/                  # LDMDet 变体
│   │   │   ├── rf_heun_adaln.py     # 基础 LDMDet (random coupling)
│   │   │   ├── hard_ot.py           # + hard OT
│   │   │   ├── sinkhorn_argmax.py   # + Sinkhorn argmax
│   │   │   ├── sinkhorn_stochastic.py # + Sinkhorn stochastic
│   │   │   └── ghss.py              # + GHSS
│   │   └── ablation/                # 消融子目录
│   │       ├── epsilon_sweep/       # ε 扫描 (sample: 0.5/1/2/3/5/10/50, argmax: 1/5/50/100)
│   │       │   ├── stoch_eps05.py
│   │       │   ├── stoch_eps1.py
│   │       │   ├── stoch_eps5.py
│   │       │   ├── argmax_eps1.py
│   │       │   └── ...
│   │       └── component/           # 组件消融 (RF vs DDPM, AdaLN on/off, Shifted on/off 等)
│   ├── runners/                     # 实验执行入口
│   │   ├── train.py                 # 单实验训练
│   │   ├── train_multi_seed.py      # 多 seed 批量训练
│   │   └── test.py                  # 推理测试 (给定 checkpoint)
│   ├── analysis/                    # 后分析脚本
│   │   ├── per_class_ap.py          # 逐类 AP
│   │   ├── velocity_entropy.py      # 速度熵计算
│   │   ├── epsilon_phase.py         # ε 相图
│   │   ├── benchmark_speed.py       # FPS benchmark
│   │   └── report.py                # 聚合多实验生成 LaTeX 表
│   └── data/                        # 数据集制备脚本
│       └── prepare_dataset.py
│
├── tests/                           # ★ 规范化测试
│   ├── __init__.py
│   ├── conftest.py                  # fixtures: 小型虚拟数据、model builder
│   ├── unit/                        # 纯 PyTorch 单元测试 (不依赖 mmdet)
│   │   ├── test_structures.py       # dataclass 序列化/设备迁移
│   │   ├── test_box_ops.py          # bbox 转换
│   │   ├── test_embeddings.py       # 时间嵌入
│   │   ├── test_noise_schedule.py   # 噪声调度
│   │   ├── test_rectified_flow.py   # RF 数学验证
│   │   ├── test_coupling_base.py    # CouplingStrategy ABC
│   │   ├── test_sinkhorn_ops.py     # Sinkhorn 迭代 (代价矩阵→传输矩阵)
│   │   ├── test_coupling_random.py  # RandomCoupling
│   │   ├── test_coupling_hard_ot.py # HardOTCoupling
│   │   ├── test_coupling_sinkhorn.py # SinkhornStochastic + Argmax
│   │   ├── test_coupling_ghss.py    # GHSS
│   │   ├── test_dynamic_conv.py     # DynamicConv
│   │   ├── test_single_head.py      # SingleHead forward/builder
│   │   ├── test_head.py             # DiffusionDetHead forward shape
│   │   ├── test_sampling.py         # Euler/Heun/DPM-Solver++ 数值测试
│   │   ├── test_matcher.py          # SimOTA matcher
│   │   ├── test_losses.py           # FocalLoss/L1Loss/GIoULoss
│   │   └── test_criterion.py        # End-to-end loss compute
│   ├── integration/                 # 集成测试 (需要 backbone, 不需要完整训练)
│   │   ├── test_head_with_backbone.py # ResNet-50 → Head → loss
│   │   ├── test_inference_trace.py  # 推理 trace (随机输入→预测输出)
│   │   └── test_multi_seed_workflow.py # 多 seed 训练→聚合
│   └── regression/                  # 回归测试 (记录已知 good 数值)
│       ├── test_baseline_mAP.py     # 固定 seed 训练 → 断言 mAP >= 阈值
│       └── test_coupling_order.py   # 断言 coupling 相对排序 (Random > OT)
│
├── tools/                           # CLI 入口 (用户直接调用)
│   ├── train.sh                     # bash: 调用 python experiments/runners/train.py
│   ├── train_multi_seed.sh          # bash: 调用 python experiments/runners/train_multi_seed.py
│   ├── test.sh                      # bash: 调用 python experiments/runners/test.py
│   └── analyze.sh                   # bash: 调用 python experiments/analysis/report.py
│
├── configs/                         # ★ 顶层 mmdet 兼容入口 (向后兼容)
│   └── -> experiments/configs/      # (symlink 或 alias)
│
├── work_dirs/                       # 实验输出 (gitignore)
├── pyproject.toml                   # 项目元信息 + 依赖
├── setup.cfg                        # 兼容旧工具
├── README.md
└── .hermes/plans/                   # 本方案文件
```

---

## 三、核心设计决策

### 3.1 ldmdet/ — 纯 PyTorch 库

**约束**：`ldmdet/` 内**任何文件都不 import mmdet / mmengine**。唯一的依赖是 `torch`, `torchvision`, `numpy`（加上 `timm` 作为可选 backbone 后端）。

**API 风格**：显式构建，不依赖注册表。

```python
from ldmdet.core import DiffusionDetHead
from ldmdet.core.single_head import SingleDiffusionDetHead
from ldmdet.diffusion.sampling import build_sampler
from ldmdet.coupling import build_coupling

# 构建耦合策略
coupling = build_coupling('sinkhorn_stochastic', epsilon=5.0, num_iters=20)

# 构建 head
head = DiffusionDetHead(
    num_classes=24,
    single_head=SingleDiffusionDetHead(num_classes=24, time_conditioning='adaln_zero'),
    coupling=coupling,
    ...
)

# 推理
results = head.predict(features, img_metas, rescale=True)
```

### 3.2 coupling/ — 耦合策略为独立可插拔组件

每个耦合策略是一个 `CouplingStrategy` 子类：

```python
class CouplingStrategy(ABC, nn.Module):
    @abstractmethod
    def couple(self, noise: Tensor, gt_diffusion: Tensor, gt_labels: Tensor, device) -> Tuple[Tensor, Tensor]:
        """Return (x_start, matched_idx)"""
```

- `RandomCoupling` — 均匀随机
- `HardOTCoupling` — 最近邻 argmax
- `SinkhornArgmaxCoupling` — Sinkhorn + argmax (ε 可调)
- `SinkhornStochasticCoupling` — Sinkhorn + multinomial (ε 可调)
- `GHSSCoupling` — 群组层次 Sinkhorn stochastic

设计好处：
1. 每个耦合策略可**独立单元测试**（不依赖 head/backbone）
2. 新耦合策略只需添加一个文件 + 注册
3. 论文中对比不同耦合时只需切换 config 中 `coupling: {type: 'sinkhorn_stochastic', epsilon: 5}`

### 3.3 experiments/mmdet_bridge/ — 薄桥层

**唯一依赖 mmdet 的代码集中在此**。`detector.py` 中的 `LDMDetDetector` 继承 `BaseDetector`，将配置解析为 `ldmdet` 纯 PyTorch 对象。

```python
class LDMDetDetector(BaseDetector):
    def __init__(self, backbone, neck, bbox_head, ...):
        super().__init__(...)
        self.backbone = MODELS.build(backbone)  # mmdet 构建 backbone
        self.neck = MODELS.build(neck)
        # bbox_head 内部用 ldmdet 纯 PyTorch 构建
        self.bbox_head = self._build_ldmdet_head(bbox_head)

    def _build_ldmdet_head(self, cfg):
        # 解析 cfg → 调用 from ldmdet import ...
        coupling = build_coupling(cfg.coupling.type, **cfg.coupling.params)
        return DiffusionDetHead(..., coupling=coupling)
```

### 3.4 Multi-Seed 系统化

`experiments/runners/train_multi_seed.py`：

```python
# python -m experiments.runners.train_multi_seed \
#     experiments/configs/ldmdet/sinkhorn_stochastic.py \
#     --seeds 42,123,456,789,1000 \
#     --gpu-ids 0,0,1,1,0

def run_multi_seed(config_path, seeds, gpu_ids):
    results = []
    for seed, gpu in zip(seeds, gpu_ids):
        work_dir = f"work_dirs/multi_seed/{exp_name}/seed_{seed}"
        result = run_single_train(config_path, seed=seed, gpu_id=gpu, work_dir=work_dir)
        results.append(result)
    aggregate_and_report(results)  # → mean ± std, best/worst, per-seed table
```

### 3.5 Test Harness (推理测试)

`experiments/runners/test.py`：

```python
# python -m experiments.runners.test \
#     experiments/configs/ldmdet/ghss.py \
#     --checkpoint work_dirs/ghss/best.pth \
#     --dataset test  # 或 val

def test_model(config, checkpoint, dataset='val'):
    model = build_from_config(config)
    load_checkpoint(model, checkpoint)
    metrics = evaluate(model, get_dataloader(dataset))
    print(metrics)  # mAP, AP50, AP75, per-class AP
```

---

## 四、实施路线 (Phase 1-5)

### Phase 1: 核心库迁移 (不破坏现有功能)

**目标**：创建 `ldmdet/` 目录，将所有 `mods/` 中的纯 PyTorch 模块迁移进去，**保持 `LDMDet/mods/` 原地不动作为 fallback**。`ldmdet/` 内使用显式 import，不依赖 mmdet 注册表。

**文件清单**：

| 源文件 (LDMDet/mods/) | 目标文件 (ldmdet/) | 变更 |
|---|---|---|
| `structures.py` | `ldmdet/data/structures.py` | 移除 `InstanceData` 的 `gt_instances` 引用 |
| `utils.py` | `ldmdet/utils/box_ops.py` | 拆分 bbox 工具 |
| `embeddings.py` | `ldmdet/diffusion/embeddings.py` | 不变 |
| `noise_schedule.py` | `ldmdet/diffusion/noise_schedule.py` | 不变 |
| `rectified_flow.py` | `ldmdet/diffusion/rectified_flow.py` | 不变 |
| `sampling.py` | `ldmdet/diffusion/sampling.py` | 改 `ImageMeta` import |
| `ot_coupling.py` | `ldmdet/coupling/` | 拆分为 6 个文件 |
| `dynamic_conv.py` | `ldmdet/core/dynamic_conv.py` | 不变 |
| `single_head.py` | `ldmdet/core/single_head.py` | 移除 MODELS 注册 |
| `roi_extractor.py` | `ldmdet/core/roi_extractor.py` | 移除 MODELS 注册 |
| `diffusiondet_head.py` | `ldmdet/core/head.py` | 改用 `ldmdet.coupling` 构建 |
| `losses.py` / `costs.py` | `ldmdet/criterion/losses.py` / `costs.py` | 不变 |
| `matcher.py` | `ldmdet/criterion/matcher.py` | 不变 |
| `criterion.py` | `ldmdet/criterion/criterion.py` | 不变 |
| `chromo_constants.py` | `ldmdet/utils/constants.py` | 不变 |

**Phase 1 完成后验证**：
```bash
# 纯 PyTorch 导入测试 — 不应触发 mmdet import
python -c "from ldmdet.core import DiffusionDetHead; print('OK')"
python -c "from ldmdet.coupling import build_coupling; print(build_coupling('random'))"
```

### Phase 2: mmdet 桥接层 + 配置重组

**目标**：创建 `experiments/mmdet_bridge/`，将 `model.py` 重构为 `detector.py`，将 `hooks.py` 移入桥接层。重组 `experiments/configs/` 为清晰分层。

### Phase 3: 多 seed + 测试 harness

**目标**：实现 `train_multi_seed.py`, `test.py`, 相关分析脚本。

### Phase 4: 单元测试全覆盖

**目标**：为 `ldmdet/` 的每个模块编写单元测试，CI 可运行。回归测试记录 baseline mAP 阈值。

### Phase 5: 文档 + 旧代码清理

**目标**：编写 `ldmdet/README.md` API 文档。确认新结构可用后，删除 `LDMDet/mods/` 中的冗余文件。

---

## 五、已知风险与缓解

| 风险 | 缓解 |
|------|------|
| 迁移过程中破坏现有训练的 config | Phase 1 保持 `LDMDet/mods/` 不变，双轨运行，验证后再删 |
| coupling 拆分引入 import 循环 | `ldmdet/coupling/base.py` 不 import 其他模块，`build_coupling` 用延迟 import |
| mmdet config 系统与新的 config 层次不兼容 | 使用 `_base_` 继承链保持 mmdet 兼容，config 只是组织方式变化 |
| 多 seed 训练 diff 大 (TF32 问题) | `train_multi_seed.py` 显式记录 `torch.backends.cudnn` 状态并存档 |
| ConvNeXt backbone 退化问题 | 暂不迁移 convnext 相关代码，仅保留 ResNet-50 路径 |

---

## 六、验收标准

1. `pip install -e .` 后可以 `from ldmdet import DiffusionDetHead` 且**不触发 mmdet import**
2. 现有所有实验 config 均可通过 `tools/train.sh` 运行，且产生与旧结构相同的结果
3. `tools/train_multi_seed.sh configs/ldmdet/sinkhorn_stochastic.py --seeds 42,123,456` 输出 mean±std 报告
4. `tools/test.sh --checkpoint work_dirs/xxx/best.pth --dataset val` 输出 mAP 值
5. `pytest tests/ -v` 全部通过 (≥ 30 个单元测试)
6. `ldmdet/coupling/` 下每个策略有 ≥ 2 个单元测试验证其数学行为
