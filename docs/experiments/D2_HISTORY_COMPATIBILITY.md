# D2历史实验兼容迁移

## 迁移原则

历史checkpoint能够加载并不等于科学配置相同。每个历史结果进入V2前必须通过：

1. train/val/test标注SHA、样本数和有效类别映射；
2. 模型、优化器、scheduler、pipeline和测试协议的字段级diff；
3. 训练日志、checkpoint metadata和结果summary中的seed交叉核对；
4. checkpoint对目标V2模型的严格加载；
5. 保存预测使用统一pycocotools重新计算六项COCO指标；
6. LQCR父子权重逐张量比较。

报告状态为`EXACT`、`STRUCTURAL_ONLY`、`INCOMPATIBLE`或
`MISSING_EVIDENCE`。只有`EXACT`允许写入canonical V2 config ID；数据库表
`config_compatibility`同时以CHECK约束和导入器前置检查执行该规则。

## 当前D2主线结论

已检查三条KaryoFlow独立检测器及其三条配对LQCR子模型。相对于canonical V2
random-coupling主方法，六条历史结果都是`STRUCTURAL_ONLY`，因为历史父模型使用
`ot_flow`、epsilon=5、20次Sinkhorn迭代及multinomial coupling。

因此V2新增了隔离的历史身份：

- `legacy_karyoflow_ot_r50`；
- `legacy_karyoflow_ot_lqcr_r50`；
- matrix `v2.d2_taichung.history_ot`。

六条结果对这一历史matrix全部为`EXACT`，并已映射到相应legacy config ID：

- checkpoint可对V2目标结构严格加载；
- 三个父检测器的真实框架seed为335778785、790448076和1342286018；
- 所有test预测独立复算与历史精确指标逐项相同；
- LQCR每对均保持590个共享张量不变，只新增最后stage的5个quality-head张量；
这些映射没有、也不能写入canonical random-coupling KaryoFlow的config ID。要获得
该主方法的D2结果，仍需要按新主matrix重新训练。

旧主配置的类别显示顺序曾写为`C10,C11,C12,C6,...`，但COCO标注和pycocotools
实际有效类别映射保持标准ID顺序；该项作为metadata warning保留。部分历史test配置
还保留了陈旧路径，兼容检查只接受已有test summary中经SHA核验的标注与预测协议。

## 复现入口

批量清单：

```text
experiments/configs/v2/manifests/d2_history_compatibility.yaml
```

重新执行全部检查：

```bash
python tools/configs/check_legacy_suite.py \
  --suite experiments/configs/v2/manifests/d2_history_compatibility.yaml
```

仅重建汇总索引：

```bash
python tools/configs/check_legacy_suite.py \
  --suite experiments/configs/v2/manifests/d2_history_compatibility.yaml \
  --aggregate-only
```
