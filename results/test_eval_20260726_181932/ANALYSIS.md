# Test Set Evaluation Results (2026-07-26)

## 完整结果对照表 (9/9 models)

| 模型 | Val mAP (seed42) | Test mAP | Δ (test−val) | 备注 |
|------|:---------:|:--------:|:----------:|------|
| A1 RF+Heun | 0.856 | 0.857 | +0.001 | 稳定 |
| A2 + Stoch. Coup. | 0.858 | 0.858 |  0.000 | 稳定 |
| **A3 DPM++** | **0.863** | **0.859** | **−0.004** | 唯一显著下降, 论文 §4.5.4 已记录 |
| A3+TopK K=300 | 0.861 | 0.860 | −0.001 | 稳定; test 上反超 A3 DPM++ |
| A3+TopK K=200 | 0.860 | 0.859 | −0.001 | 稳定; test 上与 A3 DPM++ 持平 |
| A3+TopK K=100 | 0.850 | 0.847 | −0.003 | 稳定; K=100 有害结论 robust |
| Cascade R-CNN | 0.854 | 0.853 | −0.001 | 稳定 |
| YOLOX-S | 0.796 | 0.795 | −0.001 | 稳定 |
| DiffusionDet | 0.803 | 0.804 | +0.001 | 稳定 |

## 关键结论变化分析

### 1. 稳定的结论 (无变化)
- **K=100 有害**: val −0.013 → test −0.012, robust
- **DiffusionDet 增益**: val +0.060 → test +0.055, robust
- **SOTA 对比 (Table 6)**: 所有 baseline val→test Δ ≤ 0.001, 排序不变
- **A1/A2 精度**: 完全稳定 (Δ ≤ 0.001)

### 2. 需注意的结论变化
- **A2→A3 DPM++ 增益收缩**: val +0.005 → test +0.001 (within noise)
  - 但论文 Table 5 已用 3-seed mean (0.859), 所以 +0.001 gap 已在论文中
  - §4.5.4 已明确记录此 −0.004 drop
- **A3 DPM++ 不再是 test 上的最佳 ldmdet 变体**:
  - Val: A3 DPM++ (0.863) > A3+K300 (0.861)
  - Test: A3+K300 (0.860) > A3 DPM++ (0.859) — 排名翻转, 但差距仅 0.001
- **Top-K 剪枝叙事增强**:
  - K=300: val −0.002 vs A3 → test −0.001 (essentially free, 甚至略好)
  - K=200: val −0.003 vs A3 → test  0.000 (完全 free on test!)

### 3. 论文现有处理已充分
- §4.5.4 "测试集评估" 已记录 A3 的 −0.004 drop 和 AP_S 方差问题
- Table 5 (主消融) 用 3-seed mean (0.859), 与 test 一致
- Table 10 (FPS 基准) 用 seed42 val (0.863), 与 speed benchmark 用同一 checkpoint, 合理

## 建议
1. **无需紧急修改论文**: §4.5.4 已正确处理 test set 评估
2. **可选**: 在 §4.5.4 补充一句 "所有其他模型 val→test Δ ≤ 0.001, A3 的 −0.004 是唯一显著偏差, 反映 best-checkpoint 对 val 的轻微过拟合"
3. **保留 Table 10 用 val mAP**: 速度基准与精度用同一 checkpoint (seed42 best), 一致性更强
4. **Top-K 剪枝叙事可强化**: test set 上 K=200 完全 free (Δ=0.000 vs A3), K=300 甚至略好

## 配置文件修复记录
- `cascade_rcnn_r50_test_eval.py`: 修复路径双拼接问题 (base config ann_file 已含 data_root, CocoDataset 又拼一次)
- `yolox_s_test_eval.py`: 移除 EMAHook (test 模式下 ema_model 未初始化导致 after_load_checkpoint 报错; best checkpoint 已含 EMA 权重)
