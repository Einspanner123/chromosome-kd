"""ReFlow (Standard MSE): 基于 A4 预测耦合的 2-Rectification

核心机制 (REFLOW_HEAD_DISTILL_IMPL_PLAN.md §1):
  用已训练 A4 (mAP=0.863) 对训练集推理, 生成新 coupling (x_0^pred, x_1^noise)
  替代 (x_0^GT, x_1^noise), 再用标准检测损失 (cls=GT, box=x_0^pred) 训练 2-RF 拉直轨迹。
  - 触发条件: η_str 实测 3.39~8.35 (远超阈值 0.1), 轨迹显著非直线
  - 与已证伪 velocity-loss ReFlow 区别: 不加 velocity loss → 消除梯度冲突 (旧版 cos=−0.104)

混合 target 设计 (§1.2):
  - cls_target = GT              (SimOTA 用 GT 分配正负样本, x_0^pred cls 不可靠)
  - box_target = x_0^pred        (正样本 box 回归到 A4 预测, RF 拉直目标, per-proposal)
  - 正样本筛选仍用 GT (matcher 基于 GT)

前置步骤 (必须先运行):
  python experiments/runners/generate_reflow_couplings.py \\
      experiments/configs/ldmdet/directions/mainline_ablation_24obj/a4_dpm_pp_24obj.py \\
      --checkpoint work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth \\
      --output work_dirs/reflow_couplings/train_couplings.pt

配置要点:
  - use_reflow_coupling=True + reflow_coupling_path 指向生成的 coupling 文件
  - criterion.box_target_mode='x0_pred' (box target 改为 x_0^pred, cls 始终 GT)
  - lr=1e-5 (微调场景), max_epoch=50 (冒烟+正式, §1.7 限制 ≤50ep 防 circular dependency)
  - warmup 5ep + cosine 50ep
  - reflow_dims='all' (全维度拉直; per-dim 'cxcy' 作为可选消融, 暂未在 criterion 实现)

风险与早停 (§1.7):
  - Circular Dependency (高): 限制 ≤50ep; ep5/10/20 检查 mAP, 下降立即停;
    监控 self_pred_amplification (EMA teacher, Phase 2 实现)
  - 冒烟测试: 1-seed 30ep, ep5 内崩塌则停止 (PD-RF 前车之鉴)

SwanLab: 项目 'ldmdet-reflow', 实验 'reflow_standard'
work_dir: work_dirs/reflow_standard_24obj/
"""
_base_ = ['./a4_dpm_pp_24obj.py']

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

# === 微调训练计划 (50 epoch, lr=1e-5) ===
max_epoch = 50
train_cfg = dict(max_epochs=max_epoch)

optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=0.00001, weight_decay=0.0001, _delete_=True
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

# === batch_size=2 (ross A6000 48GB) ===
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
            description='ReFlow (Standard MSE): A4 预测耦合 2-RF 拉直 | cls=GT, box=x0_pred | lr=1e-5, 50ep, bs=2',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
