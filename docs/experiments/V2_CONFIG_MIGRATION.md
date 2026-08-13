# V2实验配置迁移说明

## 决策

新实验只使用`experiments/configs/v2/`。V2不继承旧实验配置，也不为D2历史
OT权重保留兼容分支。旧运行的resolved config、checkpoint和数据库记录保持冻结；
后续使用独立兼容检查脚本逐项导入，而不是让历史条件进入新主线。

## 当前科学定义

- 两个数据集共享同一个KaryoFlow模型和150-epoch训练策略。
- KaryoFlow：RF、shift=3、AdaLN-Zero、DPM++四步、random coupling、K=500。
- renewal显式为`True`，用于保持当前D1运行语义；任何改动必须创建新版本配置。
- LQCR：父模型冻结、最终stage连续IoU分支、final-only、`p*q^2`。
- DDPM/Euler和RF/Heun是独立兄弟配置，不通过伪累计A0→A4链继承。

## 运行边界

seed、服务器、GPU、work目录和父checkpoint不是科学配置。LQCR运行必须显式提供：

```bash
python experiments/runners/train.py \
  experiments/configs/v2/recipes/d1/karyoflow_lqcr.py \
  --seed 42 --parent-checkpoint /path/to/parent.pth
```

SwanLab项目由recipe元数据交给runner注入；认证来自本机登录或环境变量，配置中
不允许出现密钥。

## 当前验证

- V2结构审计：18个Python配置，8个recipe，最大继承深度3，无外部父配置。
- 8个recipe全部解析并成功构建模型。
- D1 KaryoFlow的模型、优化器、scheduler和数据pipeline与当前运行配置等价。
- D1 epoch-53 checkpoint严格加载：590个tensor，0 missing，0 unexpected。
- D2新KaryoFlow构建为random coupling；LQCR仅5个quality-head tensor可训练。

## 尚未迁移

- DINO、RTMDet、Cascade R-CNN、YOLOX基线。
- H3蒸馏训练配置及实现。
- solver/step、Top-K/renewal和beta矩阵manifest。
- D2历史OT checkpoint兼容导入。

这些内容必须以V2新父配置补齐，不能重新连接旧继承树。
