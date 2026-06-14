# Benchmark: Chromosome20240904 数据集

跨检测器基准对比，数据集为 Chromosome20240904_NoAug_NoResize_coco（24类，1980张，平均46目标/图）。

## 模型配置对比

| 维度 | LDMDet (SOTA) | YOLOX-S | RTMDet-L | DINO-R50 | Cascade R-CNN-R50 |
|------|--------------|---------|----------|----------|-------------------|
| 配置文件 | `sota_seed42.py` 等 | `yolox_s.py` | `rtmdet_l.py` | `dino_r50.py` | `cascade_rcnn_r50.py` |
| 检测范式 | 扩散式 (RF+Heun) | Anchor-free | Anchor-free | DETR-style | Two-stage |
| 骨干网络 | ResNet-50 + FPN | CSPDarknet-S | CSPNeXt-L | ResNet-50 | ResNet-50 + FPN |
| 推理步数 | 4步 (Heun) | 1步 | 1步 | 1步 | 1步 |
| 提案/查询数 | 500 | — | — | 900 | RPN→1000 |
| 级联头数 | 6 | 1 (PANet) | 1 | 6 (decoder) | 3 (IoU递进) |
| 耦合策略 | Sinkhorn采样 ε=5 | SimOTA | DynamicSoftLabel | Hungarian | MaxIoU |
| 时间条件 | AdaLN-Zero | — | — | — | — |

## 训练配置对比

| 维度 | LDMDet (SOTA) | YOLOX-S | RTMDet-L | DINO-R50 | Cascade R-CNN-R50 |
|------|--------------|---------|----------|----------|-------------------|
| Batch Size | 8 | 8 | 2 | 2 | 4 |
| Max Epochs | 150 | 300 | 150 | 150 | 150 |
| 优化器 | AdamW | AdamW | AdamW | AdamW | AdamW |
| 学习率 | 1e-4 | 1e-4 | 1e-4 | 2.5e-5 | 1e-4 |
| Weight Decay | 1e-4 | 1e-4 | 1e-4 | 1e-4 | 1e-4 |
| LR Scheduler | CosineAnnealing | CosineAnnealing | CosineAnnealing | CosineAnnealing | CosineAnnealing |
| Warmup | 5ep Linear | 10ep Linear | 10ep Linear | 10ep Linear | 10ep Linear |
| Backbone LR Mult | 1.0 | 1.0 | 1.0 | 0.1 | 1.0 |
| Grad Clip | max_norm=1.0 | max_norm=1.0 | max_norm=1.0 | max_norm=1.0 | max_norm=1.0 |
| EMA | — | ExpMomentumEMA | — | — | — |
| Early Stopping | — | patience=30 | patience=30 | patience=30 | patience=30 |
| Seed | 1769925607 | 1769925607 | 1769925607 | 1769925607 | 1769925607 |

## 数据增强对比

| 维度 | LDMDet (SOTA) | YOLOX-S | RTMDet-L | DINO-R50 | Cascade R-CNN-R50 |
|------|--------------|---------|----------|----------|-------------------|
| Mosaic | — | Stage1 | — | — | — |
| RandomFlip | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| MultiScale | RandomChoiceResize | RandomChoiceResize | RandomChoiceResize | RandomChoiceResize | RandomChoiceResize |
| Scale Range | (480-800, 1333) | (480-800, 1333) | (480-800, 1333) | (480-800, 1333) | (480-800, 1333) |
| RandomCrop | 可选 | 可选 | 可选 | 可选 | 可选 |
| HSV Aug | — | YOLOXHSVRandomAug | — | — | — |

## 实验结果

### LDMDet 多种子稳定性

| Seed | mAP | AP50 | AP75 | 配置 |
|------|-----|------|------|------|
| 42 | — | — | — | `sota_seed42.py` |
| 123 | — | — | — | `sota_seed123.py` |
| 456 | — | — | — | `sota_seed456.py` |
| 789 | — | — | — | `sota_seed789.py` |
| 1000 | — | — | — | `sota_seed1000.py` |

> 已知单次运行 mAP 跨种子标准差约 0.003-0.004，主实验 0.751（seed=1769925607）

### 跨检测器对比

| 模型 | mAP | AP50 | AP75 | 备注 |
|------|-----|------|------|------|
| **LDMDet (Ours)** | **0.751** | **0.944** | **0.842** | RF+AdaLN+Sinkhorn采样 ε=5, 4步推理 |
| YOLOX-S | — | — | — | 待补充 |
| RTMDet-L | — | — | — | 待补充 |
| DINO-R50 | — | — | — | 待补充 |
| Cascade R-CNN-R50 | — | — | — | 待补充 |

## SwanLab 实验日志

| 项目 | 实验名 | 配置文件 |
|------|--------|---------|
| `chromosome-kd-multiseed` | `sota_seed42` | `sota_seed42.py` |
| `chromosome-kd-multiseed` | `sota_seed123` | `sota_seed123.py` |
| `chromosome-kd-multiseed` | `sota_seed456` | `sota_seed456.py` |
| `chromosome-kd-multiseed` | `sota_seed789` | `sota_seed789.py` |
| `chromosome-kd-multiseed` | `sota_seed1000` | `sota_seed1000.py` |
| `chromosome-kd-benchmark` | `yolox-s` | `yolox_s.py` |
| `chromosome-kd-benchmark` | `rtmdet-l` | `rtmdet_l.py` |
| `chromosome-kd-benchmark` | `dino-r50-4scale` | `dino_r50.py` |
| `chromosome-kd-benchmark` | `cascade-rcnn-r50` | `cascade_rcnn_r50.py` |

## 本地 work_dirs 日志路径

| 实验 | 预期路径 |
|------|---------|
| LDMDet SOTA | `work_dirs/ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5/` |
| LDMDet SOTA 复现 | `work_dirs/reproduce_0751_stochot_eps5_v2/` |
| YOLOX-S | `work_dirs/yolox_s/` |
| RTMDet-L | `work_dirs/rtmdet_l/` |
| DINO-R50 | `work_dirs/dino_r50/` |
| Cascade R-CNN-R50 | `work_dirs/cascade_rcnn_r50/` |

> 注：本地 work_dirs 当前为空，实验日志存储于 SwanLab 云端
