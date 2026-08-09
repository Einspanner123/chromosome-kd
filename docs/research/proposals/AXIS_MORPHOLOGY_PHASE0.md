# 主轴形态 RoI 残差分支 Phase-0 记录

## 目的

检验从最终级联头 7×7 RoI 特征显式提取的空间二阶矩、主方向、轴向与横向轮廓，在给定现有 256 维 `fc_feature` 后，是否仍携带可泛化的终点框残差信息。该检验对应“末级形态感知 RoI 残差分支”，不是直接训练新模块。

## 方法与预注册门槛

- A4 Dataset1 seed42 checkpoint，固定 seed=42。
- 使用动态 matcher 正样本中 aligned IoU≥0.50 且宽高有效的终点候选。
- 按图像分组的 5 折交叉验证，固定 ridge `alpha=10`。
- 基线输入为最终 `fc_feature`；候选输入为 `fc_feature +` 主轴形态描述。
- 通过门槛：相对基线残差 MSE 至少下降 5%，并使 corrected aligned-IoU 至少增加 0.002。

## 结果

100 张初测得到 705 个候选；完整 440 张得到 3,648 个候选。完整集结果：

- `fc_only` MSE：0.0207481；校正后 mean IoU：0.648838。
- `fc_plus_morphology` MSE：0.0212954；校正后 mean IoU：0.645503。
- 增量 MSE 改善率：**−2.64%**（恶化）。
- 增量 mean-IoU：**−0.00334**。
- 预注册门控：**失败**。

此外，零校正本身的 MSE 为 0.0171238、mean IoU 为 0.678274，优于两个线性探针，说明最终剩余误差在按图像外推时难以由这些静态描述稳定预测；直接增加显式主轴统计具有较高过拟合风险。

## 决策

停止该形态残差分支，不实现、不重训。该结论只证伪当前“显式二阶矩 + 轴向轮廓”的模块化假设，不等价于否定所有图像形态信息。

来源：

- 脚本：`experiments/analysis/morphology_residual_phase0.py`
- 100 张：`work_dirs/diagnosis/morphology_residual_phase0_chr2024_seed42.json`
- 完整 440 张：`work_dirs/diagnosis/morphology_residual_phase0_chr2024_seed42_full.json`
