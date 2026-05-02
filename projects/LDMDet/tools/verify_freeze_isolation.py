import sys
import os

import torch
import torch.nn.functional as F


def main():
    sys.path.insert(0, os.getcwd())
    from mmdet.utils import setup_cache_size_limit_of_dynamo
    from mmengine.config import Config
    from mmengine.runner import Runner

    setup_cache_size_limit_of_dynamo()

    config_path = "projects/LDMDet/configs/ldmdet_flowdet_adaln_reflow.py"
    checkpoint_path = "work_dirs/ldmdet_flowdet_adaln_reflow_v5/best_coco_bbox_mAP_epoch_1.pth"

    cfg = Config.fromfile(config_path)
    cfg.work_dir = "work_dirs/gradient_analysis"
    cfg.load_from = checkpoint_path
    cfg.train_cfg.max_epochs = 1

    runner = Runner.from_cfg(cfg)
    runner.load_or_resume()

    model = runner.model
    if hasattr(model, 'module'):
        model = model.module

    dataloader = runner.train_dataloader
    data_batch = next(iter(dataloader))

    data = model.data_preprocessor(data_batch, True)
    batch_inputs = data['inputs']
    batch_data_samples = data['data_samples']

    from projects.LDMDet.mods.structures import ImageMeta
    img_metas = []
    gt_bboxes = []
    gt_labels = []
    for ds in batch_data_samples:
        meta = ImageMeta(
            img_shape=ds.metainfo["img_shape"],
            ori_shape=ds.metainfo.get("ori_shape"),
            scale_factor=ds.metainfo.get("scale_factor"),
        )
        img_metas.append(meta)
        gt_bboxes.append(ds.gt_instances.bboxes)
        gt_labels.append(ds.gt_instances.labels)

    bbox_head = model.bbox_head

    print(f"\n{'='*60}")
    print(f"Freezing all layers except velocity_head...")
    print(f"{'='*60}")

    frozen_count = 0
    trainable_count = 0
    for name, param in bbox_head.named_parameters():
        if 'velocity_head' in name:
            param.requires_grad = True
            trainable_count += 1
        else:
            param.requires_grad = False
            frozen_count += 1

    print(f"  Frozen: {frozen_count} params")
    print(f"  Trainable: {trainable_count} params (velocity_head only)")

    model.train()
    features = model.extract_feat(batch_inputs)
    bbox_head.zero_grad()

    losses = bbox_head.loss(features, img_metas, gt_bboxes, gt_labels)

    det_keys = {"loss_cls", "loss_bbox", "loss_giou"}
    vel_keys = {"loss_velocity", "loss_velocity_aux"}

    det_loss = sum(losses[k] for k in det_keys if k in losses)
    vel_loss = sum(losses[k] for k in vel_keys if k in losses)

    print(f"\n{'='*60}")
    print(f"Loss values (with frozen shared layers):")
    for k, v in sorted(losses.items()):
        print(f"  {k}: {v.item():.6f}")
    print(f"  det_loss (sum): {det_loss.item():.6f}")
    print(f"  vel_loss (sum): {vel_loss.item():.6f}")
    print(f"{'='*60}")

    named_params = {name: p for name, p in bbox_head.named_parameters() if p.requires_grad}

    bbox_head.zero_grad()
    vel_loss.backward(retain_graph=True)
    grad_vel_dict = {}
    for name, p in named_params.items():
        if p.grad is not None:
            grad_vel_dict[name] = p.grad.clone().flatten()

    print(f"\n{'='*60}")
    print(f"Velocity-head gradient norms (frozen shared layers):")
    print(f"{'='*60}")
    total_norm = 0
    for name, g in sorted(grad_vel_dict.items()):
        gn = g.norm().item()
        total_norm += gn ** 2
        print(f"  {name:60s} ||vel||={gn:.6f}")
    print(f"  Total ||vel||={total_norm**0.5:.6f}")

    bbox_head.zero_grad()
    det_loss.backward(retain_graph=True)
    det_grad_count = 0
    for name, p in bbox_head.named_parameters():
        if p.grad is not None and p.grad.norm() > 0:
            det_grad_count += 1

    print(f"\n  Detection loss gradients on frozen layers: {det_grad_count} (should be 0)")
    if det_grad_count == 0:
        print(f"  *** CONFIRMED: Detection loss is fully isolated from velocity_head ***")
    else:
        print(f"  *** WARNING: Some frozen layers still have gradients! ***")

    for name, param in bbox_head.named_parameters():
        param.requires_grad = True

    print(f"\n{'='*60}")
    print(f"CONCLUSION:")
    print(f"  When shared layers are frozen, velocity_head trains independently.")
    print(f"  This should eliminate the degradation trap (Epoch 1 best → decline).")
    print(f"  Next step: Run full training with frozen shared layers for 10 epochs.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
