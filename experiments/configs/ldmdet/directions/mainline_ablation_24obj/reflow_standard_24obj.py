"""ReFlow (Standard MSE): 基于 +DPM-Solver++ 预测耦合的 2-Rectification (重试配置 v2)

核心机制 (REFLOW_HEAD_DISTILL_IMPL_PLAN.md §1):
  用已训练 +DPM-Solver++ (mAP=0.863) 对训练集推理, 生成新 coupling (x_0^pred, x_1^noise)
  替代 (x_0^GT, x_1^noise), 再用标准检测损失 (cls=GT, box=x_0^pred) 训练 2-RF 拉直轨迹。
  - 触发条件: η_str 实测 3.39~8.35 (远超阈值 0.1), 轨迹显著非直线
  - 与已证伪 velocity-loss ReFlow 区别: 不加 velocity loss → 消除梯度冲突 (旧版 cos=−0.104)

混合 target 设计 (§1.2):
  - cls_target = GT              (SimOTA 用 GT 分配正负样本, x_0^pred cls 不可靠)
  - box_target = x_0^pred        (正样本 box 回归到 +DPM-Solver++ 预测, RF 拉直目标, per-proposal)
  - 正样本筛选仍用 GT (matcher 基于 GT)

前置步骤 (必须先运行, coupling 文件已生成):
  python experiments/runners/generate_reflow_couplings.py \\
      experiments/configs/ldmdet/directions/mainline_ablation_24obj/a4_dpm_pp_24obj.py \\
      --checkpoint work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth \\
      --output work_dirs/reflow_couplings/train_couplings.pt

配置要点 (2026-07-27 重试, 修复 v1 配置Bug):
  - use_reflow_coupling=True + reflow_coupling_path 指向生成的 coupling 文件
  - criterion.box_target_mode='x0_pred' (box target 改为 x_0^pred, cls 始终 GT)
  - load_from=+DPM-Solver++ best (v1 缺失致从零训练, 欠训练)
  - lr=5e-5 (v1 用 1e-5 过小, 5x 提升)
  - max_epoch=150 (v1 用 50ep 过短, 3x 延长)
  - warmup 5ep + cosine 150ep
  - reflow_dims='all' (全维度拉直)

风险与早停 (§1.7):
  - Circular Dependency (高): ep5/10/20 检查 mAP, 下降立即停;
    监控 self_pred_amplification (EMA teacher, Phase 2 实现)
  - 关键判据: mAP_75 是否仍崩塌 (v1: 0.733→0.543); 若仍崩塌则方法本质问题, → FALSIFIED

v1 失败归因 (→ FALSIFIED §十四):
  - 配置Bug: 缺失 load_from + lr=1e-5 过小 + max_epoch=50 过短 → best 0.646@ep42
  - 方法风险: mAP_75 崩塌 0.733→0.543, cls/box 不一致
  - 本重试隔离 "配置Bug" vs "方法本质问题"

SwanLab: 项目 'ldmdet-reflow', 实验 'reflow_standard'
work_dir: work_dirs/reflow_standard_24obj/
"""
_base_ = ['./a4_dpm_pp_24obj.py']

# === 从 +DPM-Solver++ checkpoint 加载 (v1 缺失致从零训练, 欠训练) ===
load_from = 'work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth'

# === ReFlow: 预存 coupling + 混合 target ===
model = dict(
    bbox_head=dict(
        use_reflow_coupling=True,
        reflow_coupling_path='work_dirs/reflow_couplings/train_couplings.pt',
        criterion=dict(
            # ReFlow 混合 target: box target = x_0^pred (per-proposal), cls 始终 GT
            box_target_mode='x0_pred',
        ),
    ),
)

# === 微调训练计划 (150 epoch, lr=5e-5; v1 用 50ep+1e-5 过小欠训练) ===
max_epoch = 150
train_cfg = dict(max_epochs=max_epoch)

optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=0.00005, weight_decay=0.0001, _delete_=True
    ),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=True, begin=0, end=5),
    dict(
        type='CosineAnnealingLR',
        T_max=max_epoch,
        eta_min=0,
        begin=5,
        end=max_epoch,
        by_epoch=True,
    ),
]

# === batch_size=2 (workstation A5000 24GB) ===
train_dataloader = dict(batch_size=2)

# === SwanLab ===
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SwanlabVisBackend',
        init_kwargs=dict(
            project='ldmdet-reflow',
            experiment_name='reflow_standard',
            description='ReFlow (Standard MSE) 重试: +DPM-Solver++ 预测耦合 2-RF 拉直 | cls=GT, box=x0_pred | load_from=+DPM-Solver++, lr=5e-5, 150ep, bs=2',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
