# V2实验配置迁移说明

## 最终结构决策

新实验使用“dataset base + dataset-independent method + experiment matrix”。
矩阵只负责编排；每个矩阵单元都被展开为扁平的标准MMEngine Python配置，随后仍由
`Config.fromfile()`和`Runner.from_cfg()`执行。

V2不继承旧实验配置，也不为D2历史OT权重保留兼容分支。旧运行的resolved config、
checkpoint和数据库记录保持冻结；以后由独立兼容检查脚本逐项导入。

## 当前科学定义

- 两个数据集引用同一份方法定义和150-epoch检测器训练策略。
- KaryoFlow：RF、shift=3、AdaLN-Zero、DPM++四步、random coupling、K=500。
- renewal显式为`True`；任何变化必须注册为单变量消融。
- LQCR：父模型冻结、最终stage连续IoU分支、final-only、`p*q^2`。
- DDPM/Euler和RF/Heun是平行方法，不构造伪累计A0→A4继承链。

## 解析与证据

`launch_v2.py`读取矩阵后完成以下操作：

1. 合并dataset、method和runtime；
2. 校验method是否属于矩阵以及训练seed是否已注册；
3. 生成扁平`resolved_config.py`；
4. 计算科学配置和resolved config的SHA256；
5. 写出`resolution_manifest.json`，记录数据标注哈希、源文件哈希和Git状态；
6. 可选调用原生训练runner。

LQCR的矩阵关系为`same_training_seed`，正式启动时必须传入同训练seed的父checkpoint。
SwanLab项目由矩阵提供，认证来自本机登录或环境变量。

## 验证门

- 方法文件不得出现数据集路径或数据集身份。
- matrix中所有方法必须存在，config ID必须唯一。
- parent方法必须同时出现在matrix，且LQCR必须保持final-only。
- resolved config必须经过dump/reload后科学哈希不变。
- 所有组合必须成功构建MMDetection模型。
- 数据库登记的是`matrix + method`组合；seed属于运行记录，不复制为科学配置。

## 尚未迁移

- DINO、RTMDet、Cascade R-CNN、YOLOX基线。
- H3蒸馏训练方法及部署矩阵。
- solver/step、Top-K/renewal和beta消融矩阵。
- D2历史checkpoint兼容检查和导入。
