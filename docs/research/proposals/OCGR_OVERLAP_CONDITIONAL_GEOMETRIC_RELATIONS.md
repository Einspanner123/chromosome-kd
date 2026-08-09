# OCGR：重叠条件几何关系注意力

## 1. 研究结论与定位

OCGR（Overlap-Conditional Geometric Relations）是一个待完整训练验证的独立结构创新，目标是补足 proposal self-attention 对密集、重叠候选框之间显式相对几何关系建模不足的问题。它不修改 Rectified Flow 或 DPM-Solver++，而是在级联检测头的后半段，把相对框几何作为逐头 additive attention bias 注入。

Phase-0 的全验证集诊断表明：全局替换 attention 关系的预注册门槛没有通过，但在真正关心的重叠子集上，几何关系有稳定、显著的增量信息。因此最终实现采用“后段级联 + 连续邻近包络”的条件干预，而不是全局几何注意力。

## 2. 数学动机

令 proposal `i,j` 的外观/语义特征为 `f_i,f_j`，相对框几何为 `g_ij`，二元变量 `z_ij` 表示二者是否应由同一真实目标解释。标准 self-attention 的 logit 为

```text
a_ij = q(f_i)^T k(f_j) / sqrt(d).
```

它只能间接从 proposal 特征恢复几何关系。对关系后验写出对数优势：

```text
logit P(z_ij=1 | f_i,f_j,g_ij)
= logit P(z_ij=1 | f_i,f_j)
  + log [p(g_ij | z_ij=1,f_i,f_j) / p(g_ij | z_ij=0,f_i,f_j)].
```

因此，在条件独立近似或把剩余项视作可学习校正时，最小结构扩展是

```text
a'_ij,h = a_ij,h + e_ij b_h(g_ij),
```

其中 `b_h` 是每个 attention head 的几何对数似然比近似，`e_ij` 是邻近包络。OCGR 使用

```text
e_ij = exp[-0.5 (d_ij / r)^2],
```

使干预集中在 Phase-0 已证明存在信息缺口的近邻/重叠 proposal，而远距离关系连续衰减到零。

几何描述符共 7 维：

```text
g_ij = [
  (c_i-c_j)/sqrt(w_i h_i elementwise w_j h_j),
  log(w_i/w_j), log(h_i/h_j),
  IoU(i,j),
  ||normalized center delta||_2,
  log(area_i/area_j)
].
```

实现中的中心差分按对应宽、高的几何均值归一化，因此对共同平移和各向同性尺度变换不变。最后一层投影初始化为零，故初始时 `b_h(g_ij)=0`，模型在数学上精确退化为原 A4 基线。这一性质把训练差异限制为“学习到的关系校正”，避免随机初始化立即扰动已知有效的注意力结构。

## 3. Phase-0 证据

### 3.1 数据与定义

- 模型来源：Dataset1 A4（RF + DPM-Solver++）既有最佳权重。
- 数据来源：`data/Chromosome20240904_NoAug_NoResize_coco/valid/`。
- 提取位置：最终 solver 调用、级联 head 5 输出。
- 正样本：经动态 matcher 对齐到同一 GT 的 proposal 对。
- 候选过滤：输出框与匹配 GT 的 IoU 不低于 0.30。
- 划分：严格按 image 分组交叉验证，避免同图 pair 泄漏。
- 诊断代码：`experiments/analysis/geometric_relation_phase0.py`。

### 3.2 完整验证集结果

完整 440 张图、12,496 对 proposal：

| 子集 | Attention AUC | Geometry AUC | Augmented AUC | 相对 Attention 增益 |
|---|---:|---:|---:|---:|
| 全部 pair | 0.91082 | 0.99903 | 0.99896 | +0.08813 |
| overlap >= 0.2 | 0.89306 | 0.99522 | 0.99522 | +0.10216 |

200 图先导结果为：全局 attention AUC 0.9172、geometry 0.9993、augmented 0.99936，增益 0.0822；overlap 子集增益 0.1132。两次取样方向一致。

### 3.3 解释边界

上述结果证明的是“相对几何包含当前 attention 未充分编码的关系信息”，不直接证明加入模块必然提升 COCO mAP。全局增益未达到预注册的 0.10 门槛，因此不支持全局替换；重叠子集两次均超过 0.10，支持更窄的 overlap-conditional 实现。Geometry 单独接近满 AUC 也可能部分来自 matcher 的几何规则，因此完整训练必须用检测指标证伪，不能把 Phase-0 AUC 当作最终贡献。

## 4. 具体实现

- 核心模块：`ldmdet/core/geometric_relation_attention.py`
- 接线：`ldmdet/core/single_head.py`
- 级联范围控制：`ldmdet/core/head.py`
- 单元测试：`ldmdet/tests/test_geometric_relation_attention.py`
- 完整训练配置：`experiments/configs/ldmdet/directions/georel/ocgr_chr2024_seed42.py`
- 训练输出：`work_dirs/ocgr_chr2024_seed42/`

默认参数为 7 -> 32 -> 8 的 MLP，邻近半径 `r=3.0`，只在 6 个级联 head 的后 3 个（index 3--5）启用。前 3 个 head 仍承担从高噪声 proposal 建立粗定位；后 3 个 head 才利用逐渐可信的框几何处理重复、密集和重叠关系。

代码验证：几何偏置零初始化恒等性、共同平移/缩放不变性、有限梯度和配置构建测试均通过；相关测试合计 85 passed。真实训练路径两步 smoke test 中，AdaLN-Zero 令首步 attention 分支梯度为零；首步门更新后，第二步三个 OCGR head 的输出投影 weight/bias 共 6 组参数全部获得有限非零梯度。这是两层零初始化导致的一步延迟，不是断图。

## 5. Dataset1 完整训练预注册判据

这是 seed 42、从头训练的结构实验，不使用 checkpoint 续训。与同 seed A4 基线比较：

- 主基线：mAP 0.746，AP50 0.940，AP75 0.833，AP-S 0.511，AP-M 0.738，AP-L 0.646。
- 通过：最佳 mAP 必须高于 0.746；优先要求达到或超过 0.749，以越过当前约 0.003 的重复评估波动。
- 保护条件：AP75 不下降；同时检查重叠目标的重复框/漏检分解是否改善。
- 若最佳 mAP 不高于 0.746：停止 OCGR，不做超参数搜索。
- 若通过：先在 workstation A5000 固定 seed 复评，再到 ross A6000 做统一速度/显存测试；随后补 seed 123/789，最后才考虑 mini-COCO。

## 6. 当前状态

Phase-0 已完成，模块、配置和测试已实现。Dataset1 seed 42 完整训练已于 2026-08-09 23:35（Asia/Shanghai）在 workstation 的 A5000 GPU0 从头启动：

- PID：`2339090`
- 代码提交：workstation `b96e83bfd`，代码树与 ross `5e081b34` 完全相同（tree `62610a5acfef54a51b22e2d6b684e6718d0bdcb5`）
- 主日志：`/home/linkst/workplace/chromo/chromosome-kd/work_dirs/ocgr_chr2024_seed42/launch.log`
- 源码备份：`/home/linkst/workplace/chromo/chromosome-kd/work_dirs/ocgr_chr2024_seed42/20260809_233554/`
- SwanLab run：`https://swanlab.cn/@einspanner/ldmdet-ablation/runs/8ertj8vz`
- 首 50 iterations：loss 42.0760，有限梯度，约 1.11 s/iter，峰值显存 19,766 MiB
- 首 100 iterations：loss 38.2890，约 0.77 s/iter，显存 11,905 MiB
- 预计单 epoch 约 12--16 分钟；150 epoch 上限约 30--40 小时，EarlyStoppingHook 可能提前结束

首次验证 mAP 出现后，应在此追加 epoch 级轨迹；训练结束后记录 best checkpoint 和 COCO 完整指标，严格执行第 5 节的停止/推进规则。
