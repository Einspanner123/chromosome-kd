# Table Templates

This file stores the publication-oriented table skeletons separately from the main paper draft so results can be filled in quickly once experiments finish.

> ⚠️ **数据集标注说明**：以下表格模板填充时应使用 **24obj 数据集**（24_chromosomes_object，mAP 量级 0.77-0.87）的实验结果。
>
> - **Table A (Main Dataset Comparison)** 应填入 24obj 数据集结果（主路线 A0-A4、耦合策略消融等）。
> - 旧数据集 Chromosome20240904（mAP 量级 0.72-0.75）的实验结论已**暂时废弃**，不得作为主结果填入；如需引用旧数据集数值作历史对照，须在该行 Notes 列标注"旧数据集 chromo，已废弃"。
> - **Table D (Epsilon Sweep)** 中 D_eff 等机制统计量的具体数值若来自旧数据集 checkpoint（mAP≈0.753），须在 Notes/Comment 列标注数据集来源。

## Table A. Main Dataset Comparison

| Method              | Coupling         | Decoder       | epsilon | Seed Count | mAP   | AP50  | AP75  | Best Epoch | Notes |
| ------------------- | ---------------- | ------------- | ------- | ---------- | ----- | ----- | ----- | ---------- | ----- |
| Random baseline     | Random           | Sample        | N/A     | `TBD`      | `TBD` | `TBD` | `TBD` | `TBD`      | `TBD` |
| Hard OT             | Deterministic OT | Deterministic | 0       | `TBD`      | `TBD` | `TBD` | `TBD` | `TBD`      | `TBD` |
| Sinkhorn OT         | Entropic OT      | Argmax        | 1       | `TBD`      | `TBD` | `TBD` | `TBD` | `TBD`      | `TBD` |
| Sinkhorn OT         | Entropic OT      | Argmax        | 5       | `TBD`      | `TBD` | `TBD` | `TBD` | `TBD`      | `TBD` |
| Stochastic coupling | Entropic OT      | Sample        | 1       | `TBD`      | `TBD` | `TBD` | `TBD` | `TBD`      | `TBD` |
| Stochastic coupling | Entropic OT      | Sample        | 5       | `TBD`      | `TBD` | `TBD` | `TBD` | `TBD`      | `TBD` |

## Table B. External Dataset Comparison

| Method              | Coupling         | Decoder       | epsilon | Seed Count | mAP   | AP50  | AP75  | Best Epoch | Notes |
| ------------------- | ---------------- | ------------- | ------- | ---------- | ----- | ----- | ----- | ---------- | ----- |
| Random baseline     | Random           | Sample        | N/A     | `TBD`      | `TBD` | `TBD` | `TBD` | `TBD`      | `TBD` |
| Hard OT             | Deterministic OT | Deterministic | 0       | `TBD`      | `TBD` | `TBD` | `TBD` | `TBD`      | `TBD` |
| Sinkhorn OT         | Entropic OT      | Argmax        | 1       | `TBD`      | `TBD` | `TBD` | `TBD` | `TBD`      | `TBD` |
| Stochastic coupling | Entropic OT      | Sample        | 1       | `TBD`      | `TBD` | `TBD` | `TBD` | `TBD`      | `TBD` |
| Stochastic coupling | Entropic OT      | Sample        | 5       | `TBD`      | `TBD` | `TBD` | `TBD` | `TBD`      | `TBD` |

## Table C. Mechanism Statistics

| Coupling | Decoder | epsilon | H(V|Z) | H(V|X_t) | Total Variance | Between Variance | Within Variance | CAM Degree |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Random | Sample | N/A | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | N/A |
| Hard OT | Deterministic | 0 | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | N/A |
| Sinkhorn OT | Argmax | 1 | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| Sinkhorn OT | Argmax | 10 | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| Sinkhorn OT | Sample | 1 | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| Sinkhorn OT | Sample | 5 | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |

## Table D. Epsilon Sweep

| epsilon | Decoder | Diversity Recovery rho | Transport Efficiency eta | mAP   | AP50  | AP75  | Regime         | Comment |
| ------- | ------- | ---------------------- | ------------------------ | ----- | ----- | ----- | -------------- | ------- |
| 0.01    | Sample  | `TBD`                  | `TBD`                    | `TBD` | `TBD` | `TBD` | OT-dominated   | `TBD`   |
| 0.1     | Sample  | `TBD`                  | `TBD`                    | `TBD` | `TBD` | `TBD` | OT-dominated   | `TBD`   |
| 0.5     | Sample  | `TBD`                  | `TBD`                    | `TBD` | `TBD` | `TBD` | Transition     | `TBD`   |
| 1       | Sample  | `TBD`                  | `TBD`                    | `TBD` | `TBD` | `TBD` | Sweet spot     | `TBD`   |
| 5       | Sample  | `TBD`                  | `TBD`                    | `TBD` | `TBD` | `TBD` | Sweet spot     | `TBD`   |
| 10      | Sample  | `TBD`                  | `TBD`                    | `TBD` | `TBD` | `TBD` | Transition     | `TBD`   |
| 50      | Sample  | `TBD`                  | `TBD`                    | `TBD` | `TBD` | `TBD` | Bias-dominated | `TBD`   |
| 100     | Sample  | `TBD`                  | `TBD`                    | `TBD` | `TBD` | `TBD` | Bias-dominated | `TBD`   |

## Table E. Reviewer-Facing Summary

| Claim                                                  | Evidence Type            | Figure/Table               | Status | Risk  |
| ------------------------------------------------------ | ------------------------ | -------------------------- | ------ | ----- |
| Deterministic OT hurts chromosome detection            | Main results             | Table A                    | `TBD`  | `TBD` |
| Stochastic coupling recovers diversity and performance | Main results + mechanism | Table A, Table C, Figure 3 | `TBD`  | `TBD` |
| The trend transfers to the external chromosome dataset | Cross-dataset validation | Table B                    | `TBD`  | `TBD` |
| Epsilon has a sweet spot                               | Epsilon scan             | Table D, Figure 3          | `TBD`  | `TBD` |
| Theory matches the empirical trend                     | Theory + mechanism stats | Section 6, Table C         | `TBD`  | `TBD` |
