# Benchmark: 24 Chromosomes Object 数据集

跨检测器基准对比，数据集为 24_chromosomes_object/coco（24类，标准C组排序 C6-C12）。

## 与 Chromosome20240904 数据集的差异

| 维度 | Chromosome20240904 | 24 Chromosomes Object |
|------|-------------------|----------------------|
| C组类别顺序 | C10,C11,C12,C6,C7,C8,C9 | C6,C7,C8,C9,C10,C11,C12 |
| data_root | `data/Chromosome20240904_NoAug_NoResize_coco/` | `data/24_chromosomes_object/coco/` |
| 数据集配置 | `chromo_coco_detection.py` | `chromo_24obj_coco_detection.py` |

## 模型配置对比

| 维度 | LDMDet (Heun 4步) | LDMDet (DPM-Solver++ 8步) | YOLOX-S | RTMDet-L | DINO-R50 | Cascade R-CNN-R50 |
|------|-------------------|---------------------------|---------|----------|----------|-------------------|
| 配置文件 | `ldmdet_rf_heun_adaln_stochot_eps5.py` | `ldmdet_rf_dpmsolver_adaln_stochot_eps5_s8.py` | `yolox_s.py` | `rtmdet_l.py` | `dino_r50.py` | `cascade_rcnn_r50.py` |
| 基础配置 | `recipes/rf_heun_adaln_stochot_eps5.py` | 继承自左列 | 独立 | 独立 | 独立 | 独立 |
| 检测范式 | 扩散式 (RF) | 扩散式 (RF) | Anchor-free | Anchor-free | DETR-style | Two-stage |
| 骨干网络 | ResNet-50 + FPN | ResNet-50 + FPN | CSPDarknet-S | CSPNeXt-L | ResNet-50 | ResNet-50 + FPN |
| 求解器 | Heun (二阶) | DPM-Solver++ (二阶) | — | — | — | — |
| 推理步数 | 4 | 8 | 1 | 1 | 1 | 1 |
| 提案/查询数 | 500 | 500 | — | — | 900 | RPN→1000 |
| 级联头数 | 6 | 6 | 1 | 1 | 6 | 3 |
| 耦合策略 | Sinkhorn采样 ε=5 | Sinkhorn采样 ε=5 | SimOTA | DynamicSoftLabel | Hungarian | MaxIoU |
| 时间条件 | AdaLN-Zero | AdaLN-Zero | — | — | — | — |

## 训练配置对比

| 维度 | LDMDet (Heun) | LDMDet (DPM++) | YOLOX-S | RTMDet-L | DINO-R50 | Cascade R-CNN-R50 |
|------|--------------|----------------|---------|----------|----------|-------------------|
| Batch Size | 8 | 8 | 8 | 2 | 2 | 4 |
| Max Epochs | 150 | 150 | 200 | 150 | 150 | 150 |
| 优化器 | AdamW | AdamW | AdamW | AdamW | AdamW | AdamW |
| 学习率 | 1e-4 | 1e-4 | 1e-4 | 1e-4 | 2.5e-5 | 1e-4 |
| Weight Decay | 1e-4 | 1e-4 | 1e-4 | 1e-4 | 1e-4 | 1e-4 |
| LR Scheduler | CosineAnnealing | CosineAnnealing | CosineAnnealing | CosineAnnealing | CosineAnnealing | CosineAnnealing |
| Warmup | 5ep Linear | 5ep Linear | 10ep Linear | 10ep Linear | 10ep Linear | 10ep Linear |
| Backbone LR Mult | 1.0 | 1.0 | 1.0 | 1.0 | 0.1 | 1.0 |
| Grad Clip | max_norm=1.0 | max_norm=1.0 | max_norm=1.0 | max_norm=1.0 | max_norm=1.0 | max_norm=1.0 |
| EMA | — | — | ExpMomentumEMA | — | — | — |
| Early Stopping | — | — | patience=30 | patience=30 | patience=30 | patience=30 |
| Seed | 1769925607 | 1769925607 | 1769925607 | 1769925607 | 1769925607 | 1769925607 |

## 数据增强对比

| 维度 | LDMDet | YOLOX-S | RTMDet-L | DINO-R50 | Cascade R-CNN-R50 |
|------|--------|---------|----------|----------|-------------------|
| Mosaic | — | Stage1 | — | — | — |
| RandomFlip | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| MultiScale | RandomChoiceResize | RandomChoiceResize | RandomChoiceResize | RandomChoiceResize | RandomChoiceResize |
| Scale Range | (480-800, 1333) | (480-800, 1333) | (480-800, 1333) | (480-800, 1333) | (480-800, 1333) |
| RandomCrop | 可选 | 可选 | 可选 | 可选 | 可选 |
| HSV Aug | — | YOLOXHSVRandomAug | — | — | — |

## 实验结果

| 模型 | 求解器 | 推理步数 | mAP | AP50 | AP75 | 备注 |
|------|--------|---------|-----|------|------|------|
| **LDMDet (Ours)** | Heun | 4 | — | — | — | RF+AdaLN+Sinkhorn采样 ε=5 |
| **LDMDet (Ours)** | DPM-Solver++ | 8 | — | — | — | DPM-2 二阶，8步训练 |
| YOLOX-S | — | 1 | — | — | — | 待补充 |
| RTMDet-L | — | 1 | — | — | — | 待补充 |
| DINO-R50 | — | 1 | — | — | — | 待补充 |
| Cascade R-CNN-R50 | — | 1 | — | — | — | 待补充 |

> 参考 Chromosome20240904 数据集结果：LDMDet Heun 4步 mAP=0.751，DPM-Solver++ 8步 mAP=0.755

## SwanLab 实验日志

| 项目 | 实验名 | 配置文件 |
|------|--------|---------|
| `chromosome-kd-benchmark-24obj` | `ldmdet-rf-adaln-stochot-eps5` | `ldmdet_rf_heun_adaln_stochot_eps5.py` |
| `chromosome-kd-benchmark-24obj` | `ldmdet-rf-adaln-stochot-eps5-dpmsolver-s8` | `ldmdet_rf_dpmsolver_adaln_stochot_eps5_s8.py` |
| `chromosome-kd-benchmark-24obj` | `yolox-s` | `yolox_s.py` |
| `chromosome-kd-benchmark-24obj` | `rtmdet-l` | `rtmdet_l.py` |
| `chromosome-kd-benchmark-24obj` | `dino-r50-4scale` | `dino_r50.py` |
| `chromosome-kd-benchmark-24obj` | `cascade-rcnn-r50` | `cascade_rcnn_r50.py` |

## 本地 work_dirs 日志路径

| 实验 | 预期路径 |
|------|---------|
| LDMDet Heun | `work_dirs/ldmdet_rf_heun_adaln_stochot_eps5_24obj/` |
| LDMDet DPM-Solver++ | `work_dirs/ldmdet_rf_dpmsolver_adaln_stochot_eps5_s8_24obj/` |
| YOLOX-S | `work_dirs/yolox_s_24obj/` |
| RTMDet-L | `work_dirs/rtmdet_l_24obj/` |
| DINO-R50 | `work_dirs/dino_r50_24obj/` |
| Cascade R-CNN-R50 | `work_dirs/cascade_rcnn_r50_24obj/` |

> 注：本地 work_dirs 当前为空，实验日志存储于 SwanLab 云端
