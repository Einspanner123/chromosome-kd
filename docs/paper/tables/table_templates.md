# Table Templates

This file stores the publication-oriented table skeletons separately from the main paper draft so results can be filled in quickly once experiments finish.

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
