# 将根目录 `configs/` 引用内联到 `experiments/` — 实施计划

> **目标：** 消除 `experiments/configs/` 对根目录 `configs/_base_/` 的所有外部引用，使 `experiments/` 完全自包含。
>
> **架构：** 将 2 个被引用的根配置文件直接复制到 `experiments/configs/_base_/` 下，更新继承路径，验证 mmengine 配置加载无误。
>
> **影响范围：** 零 — 根目录 `configs/` 文件本身不作改动，仅 `experiments/` 内部文件变更。
>
> **验证方式：** `python -c "from mmengine.config import Config; Config.fromfile('experiments/configs/ldmdet/rf_heun_adaln.py')"` 确认加载成功，回归测试确保 24obj/ablation/baseline 等衍生配置也可加载。

---

## 背景 — 依赖关系图

```
experiments/configs/_base_/default_runtime.py
  → _base_ = ['../../../configs/_base_/default_runtime.py']          ← root REF

experiments/configs/ldmdet/rf_heun_adaln.py  ← 21 个配置文件依赖此文件
  → _base_ = [
      '../_base_/default_runtime.py',                                ← experiments 内
      '../../../configs/_base_/datasets/chromo_coco_detection.py',   ← root REF
    ]

experiments/configs/ldmdet/sinkhorn_stochastic.py  → rf_heun_adaln.py
experiments/configs/ldmdet/ghss.py                → rf_heun_adaln.py
... 共 15 个 ldmdet/* → rf_heun_adaln
... 共 8 个 ablation/* → ldmdet/sinkhorn_stochastic 或 sinkhorn_argmax
... 共 7 个 multiset/* → ldmdet/* 或 baselines/diffusiondet_ddpm
```

**只有 2 个根配置文件被引用：**
1. `configs/_base_/default_runtime.py` (26 行) — mmdet 通用运行时配置
2. `configs/_base_/datasets/chromo_coco_detection.py` (219 行) — 染色体 24 类数据集配置

---

## 任务

### 任务 1：将 `chromo_coco_detection.py` 复制到 `experiments/configs/_base_/datasets/`

**目标：** 把根目录的数据集配置复制到 `experiments/` 下，消除第一个外部引用。

**文件：**
- 创建: `experiments/configs/_base_/datasets/chromo_coco_detection.py`
- 修改: `experiments/configs/ldmdet/rf_heun_adaln.py`

**步骤：**
1. 将根目录 `configs/_base_/datasets/chromo_coco_detection.py` **完整复制**到 `experiments/configs/_base_/datasets/chromo_coco_detection.py`
2. 修改 `experiments/configs/ldmdet/rf_heun_adaln.py` 第 9 行：
   - 旧: `'../../../configs/_base_/datasets/chromo_coco_detection.py',`
   - 新: `'../_base_/datasets/chromo_coco_detection.py',`

**验证：**
```bash
cd /media/ross/8TB/linkst/chromo/chromosome-kd
python -c "from mmengine.config import Config; c=Config.fromfile('experiments/configs/ldmdet/rf_heun_adaln.py'); print('OK:', len(c.train_dataloader.dataset.classes), 'classes')"
```
预期输出: `OK: 24 classes`

---

### 任务 2：将 `default_runtime.py` 内联到 `experiments/configs/_base_/default_runtime.py`

**目标：** 将根目录的 `default_runtime.py` 内容合并到 experiments 版本中，移除 `_base_` 继承链。

**文件：**
- 修改: `experiments/configs/_base_/default_runtime.py`

**策略：** 当前 experiments 版本通过 `_base_` 继承根目录版本，然后覆盖 `custom_imports`、`default_hooks.checkpoint`、`vis_backends`。我们将 root 版内容作为基值直接写入，让 experiments 版的覆盖保持原位，移除 `_base_` 行。

**步骤：**
1. 读取当前 `experiments/configs/_base_/default_runtime.py`
2. 读取根 `configs/_base_/default_runtime.py`
3. 合并：用 root 版的 `default_hooks`、`env_cfg`、`vis_backends`、`visualizer`、`log_processor`、`log_level`、`load_from`、`resume` 作为基线，然后用 experiments 版的 `custom_imports` 和覆盖后的 `default_hooks`/`vis_backends` 写入，**移除 `_base_` 行**

**合并后的文件应包含的最终内容：**
```python
"""基础运行时配置 — experiments/configs/_base_/default_runtime.py

自包含版本（不再继承根目录 configs/_base_/default_runtime.py），
加上 LDMDet bridge 自定义导入。
覆盖 vis_backends 使用独立 SwanLab 项目。
"""

default_scope = 'mmdet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=2, save_last=True),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False

# 自动注册 ldmdet bridge 模块
custom_imports = dict(
    imports=[
        'experiments.mmdet_bridge.registry',
        'experiments.mmdet_bridge.detector',
        'experiments.mmdet_bridge.hooks',
        'swanlab.integration.mmengine',
    ],
    allow_failed_imports=False,
)

# SwanLab + Tensorboard + Local
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(type='SwanlabVisBackend', init_kwargs=dict(project='ldmdet-ablation', api_key='Huzvq1fnDeqOwgQo2AMAI', resume='allow')),
]
visualizer = dict(type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
```

关键差异点（与 root 版对比）：
- `default_hooks.checkpoint`: root 是 `max_keep_ckpts=1, save_best='coco/bbox_mAP'`，experiments 是 `max_keep_ckpts=2, save_last=True`（无 `save_best`—已在根版本中保留，但实验覆盖去掉了它。让我们保留实验版本的意图）
- `vis_backends`: root 只有 `LocalVisBackend`，experiments 追加了 `TensorboardVisBackend` 和 `SwanlabVisBackend`
- `custom_imports`: experiments 专属

**验证：**
```bash
cd /media/ross/8TB/linkst/chromo/chromosome-kd
python -c "from mmengine.config import Config; c=Config.fromfile('experiments/configs/_base_/default_runtime.py'); print('OK: scope=', c.default_scope)"
```
预期输出: `OK: scope= mmdet`

---

### 任务 3：全面回归验证

**目标：** 确认所有派生配置仍能加载。

**步骤：**
```bash
cd /media/ross/8TB/linkst/chromo/chromosome-kd
# 直接节点
python -c "from mmengine.config import Config; c=Config.fromfile('experiments/configs/ldmdet/rf_heun_adaln.py'); print('rf_heun_adaln: OK')"
python -c "from mmengine.config import Config; c=Config.fromfile('experiments/configs/ldmdet/sinkhorn_stochastic.py'); print('sinkhorn_stochastic: OK')"
python -c "from mmengine.config import Config; c=Config.fromfile('experiments/configs/ldmdet/ghss.py'); print('ghss: OK')"

# ldmdet 系列
for f in experiments/configs/ldmdet/*.py; do python -c "from mmengine.config import Config; Config.fromfile('$f')" && echo "$f: OK" || echo "$f: FAIL"; done

# ablation 系列
for f in experiments/configs/ablation/*.py; do python -c "from mmengine.config import Config; Config.fromfile('$f')" && echo "$f: OK" || echo "$f: FAIL"; done

# multiset 系列
for f in experiments/configs/multiset/*.py; do python -c "from mmengine.config import Config; Config.fromfile('$f')" && echo "$f: OK" || echo "$f: FAIL"; done

# baselines
for f in experiments/configs/baselines/*.py; do python -c "from mmengine.config import Config; Config.fromfile('$f')" && echo "$f: OK" || echo "$f: FAIL"; done
```

**预期结果：** 所有配置加载成功，无一 FAIL。

---

### 任务 4：提交

```bash
cd /media/ross/8TB/linkst/chromo/chromosome-kd
git add experiments/configs/_base_/datasets/chromo_coco_detection.py
git add experiments/configs/_base_/default_runtime.py
git add experiments/configs/ldmdet/rf_heun_adaln.py
git diff --cached --stat
```

确认 diff 合理后手动提交。
