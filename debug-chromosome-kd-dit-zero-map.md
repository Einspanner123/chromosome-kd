# Debug Session: chromosome-kd-dit-zero-map
- **Status**: [OPEN]
- **Issue**: LDMDet-DiT (24-class chromosome detection) 的两个变体（`shifted3_direct` 与 `shifted3_adaln9`）训练均无法收敛，val mAP 在 50+ epoch 始终 ≤ 0.0010。
- **Debug Server**: http://127.0.0.1:7777/event（用户在本地 shell 启动）
- **Log File**: .dbg/trae-debug-log-chromosome-kd-dit-zero-map.ndjson
- **一键脚本**: `./debug_run_adaln9.sh`

## Reproduction Steps
1. 启动 Debug Server + 训练（用户在本地 shell 执行）：
   ```bash
   cd /home/linkst/workplace/chromo/chromosome-kd
   bash debug_run_adaln9.sh
   ```
   或手工两步：
   ```bash
   python3 /home/linkst/.trae-cn/builtin_skills/TRAE-debugger/tools/debug-server/python/debug-server.py \
     --session chromosome-kd-dit-zero-map --outdir .dbg --clean --idle 1800 &
   curl -s http://127.0.0.1:7777/health
   python tools/train.py projects/LDMDet/configs/ldmdet_dit.py \
     --work-dir work_dirs/ldmdet_dit_r50_shifted3_adaln9_dbg
   ```
2. 跑满 1 个 epoch
3. 实际：`coco/bbox_mAP` 在 epoch 0 仍为 0.0000 (val 第一次跑会跌到 0.0010)

## 插桩清单（已落地，只读探针，无业务逻辑改动）
| ID | 文件 | 函数 | 触发频率 | data 字段 |
|----|------|------|----------|-----------|
| A | `deformable_attn.py` | `MultiScaleDeformableAttention.forward` 末尾 | iter%50 | ref_min/max/mean, sl_min/max/mean, sl_out_of_range_ratio, sl_has_nan |
| B | `box_tokenizer.py` | `BoxTokenizer.forward` 末尾 | iter%50 | bt_std_across_instances/features, bt_pairwise_l2_consec, level_hist, init_mode |
| C | `loss.py` | `DiffusionDetMatcher.forward` 末尾 | iter%20 | num_pred_per_img, gt_per_img, matched_per_img, center_radius, candidate_topk |
| D | `dit_head.py` | `DiTDiffusionDetHead.loss` 入口 | iter%20 | bs, gt_counts, gt_total, t_min/max/mean, t_hist[4桶] |
| E | `dit_head.py` | `DiTDiffusionDetHead.forward` (head_series 循环后) | iter%50 | n_heads, pairwise_iou_mean/min, pred_bbox_range_min/max |

## 运行后用户需要贴回的产物
1. `.dbg/trae-debug-log-chromosome-kd-dit-zero-map.ndjson`（每行一个 JSON 事件）
2. `.dbg/train_chromosome-kd-dit-zero-map.log`（训练 stdout，截取前 200 行）
3. （可选）`work_dirs/ldmdet_dit_r50_shifted3_adaln9_dbg/` 第一个 epoch 末的 `coco/bbox_mAP`

## 假设-证据矩阵（Step 7 填）
| ID | 假设 | 状态 | 证据（已收集） |
|----|------|------|----------------|
| **A** | DeformAttn 坐标空间错乱 | ✅ **CONFIRMED** | `sl_min` mean=-3.05 (range -4.40 ~ -2.34), `sl_max` mean=4.00 (range 3.21 ~ 4.90), `sl_out_of_range` mean=**0.57** (53%~62%) 跨所有 540 个采样点稳定存在。ref_points 本身 [0, 1] 正常，问题在 offsets 无界导致 55%+ 采样点出 FPN 范围。 |
| B | BoxTokenizer 多样性崩塌 | ❌ **REJECTED** | `bt_std_across_instances` 稳定 0.95, `bt_pairwise_l2_consec` 稳定 21.0, 与 GT 几何位置关联。Box tokens 正常。 |
| C | Matcher 几乎不匹配 | ⚠️ **PARTIAL** | match_ratio mean=0.564 (median 0.548, min 0.021)。SimOTA 在 `center_radius=0.5` 下只匹配 56% 的 GT；不是 0% 但偏低，会放大 loss。 |
| D | t→0 尖峰 | ⚠️ **PARTIAL** | 156 个 t 样本的分布：4.5% t<0.1, 27.6% t∈[0.1,0.5), 50% t∈[0.5,0.9), 17.9% t>0.9。t_mean≈0.63。rf_shift=3.0 把分布推向 1，但不是病态。 |
| E | 6 head 输出雷同 | ❌ **REJECTED** | `pairwise_iou_mean` mean=0.067 (range 0.003-0.182), `pairwise_iou_min` mean=0.002。6 head 输出 IoU≈0.07，多样性正常。 |

## 根因结论
**Hypothesis A 是主因**：`MultiScaleDeformableAttention.sampling_offsets` 没有任何边界约束，offsets 自由成长到 ±4 量级，导致 57% 的采样点落在 FPN 特征图之外，被 `grid_sample(padding_mode='zeros')` 填 0。
- 后果 1：cross-attn 拿到的 value 有 ~57% 是 0，等价于注意力看到的图像信息严重衰减
- 后果 2：6 head 都在学"如何正确采样"，但因为大部分采样点都越界，监督信号是混乱的 → 收敛不动
- 后果 3：6 个辅助 head 的 `loss_bbox` 都 ~60+（不收敛），但 cls loss ~2.3（接近 focal loss baseline 1.39），说明分类头在学到东西但回归学不到

## 最小修复方案
**Hypothesis A 的 fix**：在 `MultiScaleDeformableAttention.forward` 把 offsets 用 `tanh` 夹到 [-0.5, 0.5] 范围：
```python
# 修改 dit_head.py 中 sampling_locations 的计算（实际在 deformable_attn.py）
offsets = offsets.view(...).tanh() * 0.5
sampling_locations = ref_points_expanded + offsets
```

具体改 [deformable_attn.py:212-225](file:///home/linkst/workplace/chromo/chromosome-kd/projects/LDMDet/mods/deformable_attn.py#L212-L225)，保持其它代码不变。

## Step 8 — Minimal Fix（已落地）
在 [deformable_attn.py:213-216](file:///home/linkst/workplace/chromo/chromosome-kd/projects/LDMDet/mods/deformable_attn.py#L213-L216) 插入：
```python
# Fix (chromosome-kd-dit-zero-map / Hypothesis A):
# offsets 自由成长到 ±4 量级导致 57% 采样点出 FPN 范围被 padding zero。
# 用 tanh 夹到 [-0.5, 0.5] 限制相对参考点的最大采样半径 = 半个 FPN 单元。
offsets = offsets.tanh() * 0.5
sampling_locations = ref_points_expanded + offsets
```
- **保留所有探针**（A/B/C/D/E），runId 通过 `DEBUG_RUN_ID` 环境变量切换。
- 单元测试通过：output shape=(2,100,256), all finite。

## Step 9 — Post-Fix Verification（用户操作）
在本地 shell 执行：
```bash
cd /home/linkst/workplace/chromo/chromosome-kd
bash debug_run_adaln9_postfix.sh
```
脚本会：
1. 清空 `.dbg/trae-debug-log-*.ndjson`
2. 设置 `DEBUG_RUN_ID=post-fix`
3. 启动 Debug Server + adaln9 训练 1 epoch 到 `work_dirs/ldmdet_dit_r50_shifted3_adaln9_postfix/`

## 修复 / 验证
（待 Step 9-10 填充：pre vs post 的 `sl_out_of_range` / val mAP 对比）

## Step 9 — 训练日志自查（Post-Fix 实际已运行）
用户答复"现在去跑 post-fix"后，postfix 脚本在 base conda env 启动时崩了（缺 mmcv）；但用户在**正确 conda env (mm/chromo)** 中手动重启了同一个训练 (`work_dirs/ldmdet_dit_r50_shifted3_adaln9_dbg`)，因此 post-fix 数据实际从 12:27 起一直在写。
- `.ndjson` 第一条事件 `12:27:15 iter=1 sl=[-0.48, 1.49]` ⇒ A 修复从 iter=1 生效
- 但 env var `DEBUG_RUN_ID=post-fix` 未设，所以事件都被打成了 `runId="pre-fix"`（label 错，代码已生效）

**A 修复效果 (834 events)**:
| 指标 | pre-fix | post-fix (实际) |
|------|---------|---------------|
| `sl_min` mean | -3.05 | **-0.332** ✅ |
| `sl_max` mean | +4.00 | **+1.334** ✅ |
| `sl_out_of_range` | 0.57 | **0.15** ✅ |
| match_ratio | 0.564 | **0.648** ✅ |

**训练轨迹 (3 epochs)**:
| ep | val mAP | loss  | bbox  | giou | grad_norm |
|----|---------|-------|-------|------|-----------|
| 1  | 0.0000  | 387→377 | 77→75 | 1.95 | 1100~1170 稳 |
| 2  | 0.0000  | 234→65  | 60→7  | 1.62 | 877→2745 ⚠️ 爆 |
| 3  | 进行中   | 60→57   | 7→6.5 | 1.55 | 2453→3149 ⚠️ 继续涨 |

**bbox loss 卡在 ~7** — 模型在前 epoch 大跳 (77→7) 之后停止收敛；grad_norm 单调上涨是核心问题。

## Step 10 — 第二轮修复 (grad clip + matcher radius)
发现 config 里 `clip_grad=dict(max_norm=1.0, ...)` 极不合理（Deformable DETR 标准是 35），1.0 把 update 几乎切零。

[configs/ldmdet_dit.py:233](file:///home/linkst/workplace/chromo/chromosome-kd/projects/LDMDet/configs/ldmdet_dit.py#L233) 改为 `max_norm=35.0`；[configs/ldmdet_dit.py:99](file:///home/linkst/workplace/chromo/chromosome-kd/projects/LDMDet/configs/ldmdet_dit.py#L99) 把 `center_radius` 0.5 → 1.5（matcher 默认 2.5，0.5 太严）。

**当前训练用的是旧 config (max_norm=1.0, radius=0.5)**，所以需要重启训练才能验证这两个修复。

## 清理
（待用户确认后；将删除 `.dbg/`、`debug-chromosome-kd-dit-zero-map.md`、`debug_run_adaln9.sh` 以及 4 个源码文件中的 `#region debug-point` 块）
