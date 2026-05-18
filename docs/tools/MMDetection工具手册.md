# MMDetection 工具手册

本文档提供了 MMDetection 工具包中可用工具的全面指南，按类别组织。

## 主要训练和测试工具

### train.py

训练模型的主脚本。

**功能：**

- 使用指定的配置文件训练检测器
- 支持自动混合精度训练
- 启用自动学习率缩放
- 支持从检查点恢复
- 适用于各种启动器（none, pytorch, slurm, mpi）

**参数：**

- config：训练配置文件的路径
- `--work-dir`：保存日志和模型的目录
- `--amp`：启用自动混合精度训练
- `--auto-scale-lr`：启用自动学习率缩放
- `--resume`：从检查点恢复（自动或指定路径）
- `--cfg-options`：覆盖配置文件中的设置
- `--launcher`：作业启动器选择

### test.py

测试（和评估）模型的主脚本。

**功能：**

- 使用指定的配置和检查点测试训练好的模型
- 支持结果可视化
- 可以将预测结果转储到离线评估的 pickle 文件中
- 支持测试时增强

**参数：**

- config：测试配置文件的路径
- checkpoint：检查点文件的路径
- `--work-dir`：保存评估指标的目录
- `--out`：将预测结果转储到 pickle 文件
- `--show`：显示预测结果
- `--show-dir`：保存可视化图像的目录
- `--wait-time`：显示结果之间的间隔
- `--cfg-options`：覆盖配置文件中的设置
- `--launcher`：作业启动器选择
- `--tta`：启用测试时增强

## 分析工具

### analyze_logs.py

分析日志文件以可视化训练指标并分析训练时间。

**功能：**

- 绘制指定指标的曲线
- 分析训练时间统计
- 支持多个日志文件进行比较

**参数：**

- `--json_logs`：日志文件的路径
- `--keys`：要分析的指标
- `--title`：图表的标题
- `--legend`：图表的图例
- `--backend`：matplotlib 的后端
- `--style`：seaborn 的样式
- `--out`：图表的输出路径
- `--mode`：模式（train_time/eval）
- `--interval`：迭代间隔

### analyze_results.py

分析检测结果并显示表现最差和最好的图像。

**功能：**

- 根据 mAP 识别表现最好和最差的图像
- 可视化检测结果
- 计算混淆矩阵

**参数：**

- config：配置文件路径
- `prediction_path`：预测结果的路径
- `--show`：显示图像
- `--wait-time`：显示图像之间的间隔
- `--topk`：要显示的图像数量
- `--show-score-thr`：显示边界框的分数阈值
- `--cfg-options`：覆盖配置文件中的设置

### benchmark.py

使用指定指标对模型进行基准测试。

**功能：**

- 对推理速度进行基准测试
- 对数据加载性能进行基准测试
- 测量数据集加载时间

**参数：**

- config：测试配置文件路径
- `--checkpoint`：检查点文件
- `--task`：基准测试任务（inference/dataloader/dataset）
- `--repeat-num`：测量的重复次数
- `--max-iter`：最大迭代次数
- `--log-interval`：日志记录间隔
- `--num-warmup`：预热迭代次数
- `--fuse-conv-bn`：融合卷积和批归一化
- `--cfg-options`：覆盖配置文件中的设置

### browse_dataset.py

浏览数据集并可视化注释。

**功能：**

- 使用注释可视化数据集样本
- 将可视化图像保存到目录
- 支持不同类型的数据集

**参数：**

- config：训练配置文件路径
- `--output-dir`：保存图像的目录
- `--not-show`：不显示图像
- `--show-interval`：显示图像之间的间隔
- `--cfg-options`：覆盖配置文件中的设置

### coco_error_analysis.py

对 COCO 格式结果进行错误分析。

**功能：**

- 按照 COCO 评估协议分析检测错误
- 为不同错误类型生成图表
- 计算每个类别的各种指标

**参数：**

- `result`：结果文件路径
- `--config`：配置文件路径
- `--out_dir`：输出目录
- `--type`：评估类型（bbox/segm）
- `--extraplots`：生成额外的图表

### confusion_matrix.py

从检测结果生成混淆矩阵。

**功能：**

- 基于检测结果生成混淆矩阵
- 支持混淆矩阵的可视化
- 根据置信度阈值过滤检测

**参数：**

- config：测试配置文件路径
- `prediction_path`：预测结果的路径
- `save_dir`：保存混淆矩阵的目录
- `--show`：显示混淆矩阵
- `--color-theme`：矩阵颜色映射的主题
- `--score-thr`：过滤检测的分数阈值
- `--tp-iou-thr`：真阳性 IoU 阈值
- `--nms-iou-thr`：NMS IoU 阈值
- `--cfg-options`：覆盖配置文件中的设置

### get_flops.py

计算模型的 FLOPs。

**功能：**

- 计算模型的浮点运算次数（FLOPs）
- 计算模型参数
- 分析模型复杂度

**参数：**

- config：训练配置文件路径
- `--num-images`：计算 FLOPs 的图像数量
- `--cfg-options`：覆盖配置文件中的设置

### eval_metric.py

评估模型预测的指标。

**功能：**

- 在预测结果上评估特定指标
- 支持各种评估指标

**参数：**

- config：配置文件路径
- `prediction_path`：预测结果的路径
- `--format-only`：仅格式化结果而不进行评估
- `--eval`：评估指标
- `--cfg-options`：覆盖配置文件中的设置

### fuse_results.py

使用不同策略融合多个结果。

**功能：**

- 组合来自多个模型的结果
- 支持不同的融合策略

**参数：**

- `--result`：结果文件路径
- `--fusion`：融合策略
- `--weights`：每个结果的权重
- `--output`：融合结果的输出路径

## 数据集转换器

### pascal_voc.py

将 PASCAL VOC 数据集转换为 COCO 格式。

**功能：**

- 将 PASCAL VOC 注释转换为 COCO 格式
- 处理检测和分割注释

**参数：**

- `--devkit-path`：VOCdevkit 的路径
- `--split`：要转换的拆分（train/val/test）
- `--out-dir`：输出目录

### cityscapes.py

将 Cityscapes 数据集转换为 COCO 格式。

**功能：**

- 将 Cityscapes 数据集转换为 COCO 格式
- 处理实例分割注释

**参数：**

- `--cityscapes-path`：Cityscapes 数据集的路径
- `--img-dir`：相对于 Cityscapes 路径的图像目录
- `--gt-dir`：相对于 Cityscapes 路径的注释目录
- `--nproc`：进程数

### crowdhuman2coco.py

将 CrowdHuman 数据集转换为 COCO 格式。

**功能：**

- 将 CrowdHuman 数据集注释转换为 COCO 格式

### images2coco.py

从图像文件创建 COCO 格式的注释。

**功能：**

- 从图像文件生成 COCO 格式的注释
- 创建基本注释结构

**参数：**

- `--image-dir`：图像目录
- `--output`：输出注释文件
- `--categories`：类别名称

## 部署工具

### mmdet2torchserve.py

将 MMDetection 模型转换为 TorchServe 格式。

**功能：**

- 将 MMDetection 模型转换为 TorchServe .mar 格式
- 为使用 TorchServe 部署准备模型

**参数：**

- config：配置文件路径
- checkpoint：检查点文件路径
- `--output-folder`：.mar 文件的输出文件夹
- `--model-name`：模型名称
- `--model-version`：模型版本
- `--force`：强制覆盖现有文件

### test_torchserver.py

测试由 TorchServe 提供服务的 MMDetection 模型。

**功能：**

- 测试使用 TorchServe 部署的模型
- 向 TorchServe 端点发送推理请求

**参数：**

- config：配置文件路径
- `--inference-addr`：推理地址
- `--image`：图像文件路径
- `--device`：推理设备

## 杂项工具

### print_config.py

打印整个配置。

**功能：**

- 显示完整配置
- 可以将配置保存到文件

**参数：**

- config：配置文件路径
- `--save-path`：保存配置的路径
- `--cfg-options`：覆盖配置文件中的设置

### download_dataset.py

下载用于训练的数据集。

**功能：**

- 下载和提取数据集
- 支持多个数据集（COCO, VOC 等）

**参数：**

- `--dataset-name`：要下载的数据集名称
- `--save-dir`：保存数据集的目录
- `--unzip`：解压下载的文件
- `--delete`：提取后删除压缩文件
- `--threads`：下载线程数

### split_coco.py

按类别拆分 COCO 数据集。

**功能：**

- 根据类别将 COCO 数据集拆分为子集
- 为每个子集创建单独的注释文件

**参数：**

- `--data-root`：数据集根目录
- `--out-dir`：输出目录
- `--num-categories`：每个拆分的类别数

## 模型转换器

### publish_model.py

处理要发布的检查点。

**功能：**

- 处理用于发布的检查点
- 从检查点中删除不必要的信息
- 在检查点文件名中添加校验和

**参数：**

- `in_file`：输入检查点文件名
- `out_file`：输出检查点文件名
- `--save-keys`：在已发布检查点中保存的键

### upgrade_model_version.py

将模型版本从旧版本升级。

**功能：**

- 将旧版本的模型检查点升级
- 更新模型键和结构以匹配当前版本

## MOT（多对象跟踪）分析工具

### mot_param_search.py

为 MOT 模型搜索最佳参数。

**功能：**

- 对 MOT 模型执行参数搜索
- 测试不同的参数组合
- 评估跟踪性能

### mot_error_visualize.py

可视化 MOT 结果中的错误。

**功能：**

- 可视化不同类型的跟踪错误
- 分析 ID 切换和误报/漏报

这个全面的工具集使用户能够在 MMDetection 框架内高效地训练、测试、分析和部署目标检测模型。
