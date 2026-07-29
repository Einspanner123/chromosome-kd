"""DiffuDETR 单元测试 — 对照原仓库 (MBadran2000/DiffuDETR) 验证修复正确性.

测试覆盖:
  - criterion: num_boxes 归一化 (R1 修复), use_vlb 开关 (R5 修复), GIoU 无断言 (Bug1)
  - scheduler: loss_weight 公式, q_sample
  - head: inverse_sigmoid/sigmoid 互逆, 空间转换互逆, sigmoid 输出有界 (R4 修复)
  - 端到端: forward_train 产生有限 loss
"""
