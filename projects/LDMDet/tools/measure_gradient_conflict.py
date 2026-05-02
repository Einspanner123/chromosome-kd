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

    model.train()
    features = model.extract_feat(batch_inputs)

    bbox_head = model.bbox_head
    bbox_head.zero_grad()

    losses = bbox_head.loss(features, img_metas, gt_bboxes, gt_labels)

    det_keys = {"loss_cls", "loss_bbox", "loss_giou"}
    vel_keys = {"loss_velocity", "loss_velocity_aux"}

    det_loss = sum(losses[k] for k in det_keys if k in losses)
    vel_loss = sum(losses[k] for k in vel_keys if k in losses)

    print(f"\n{'='*60}")
    print(f"Loss values:")
    for k, v in sorted(losses.items()):
        print(f"  {k}: {v.item():.6f}")
    print(f"  det_loss (sum): {det_loss.item():.6f}")
    print(f"  vel_loss (sum): {vel_loss.item():.6f}")
    print(f"{'='*60}")

    named_params = {name: p for name, p in bbox_head.named_parameters() if p.requires_grad}

    bbox_head.zero_grad()
    det_loss.backward(retain_graph=True)
    grad_det_dict = {}
    for name, p in named_params.items():
        if p.grad is not None:
            grad_det_dict[name] = p.grad.clone().flatten()

    bbox_head.zero_grad()
    vel_loss.backward(retain_graph=True)
    grad_vel_dict = {}
    for name, p in named_params.items():
        if p.grad is not None:
            grad_vel_dict[name] = p.grad.clone().flatten()

    common_names = sorted(set(grad_det_dict.keys()) & set(grad_vel_dict.keys()))
    print(f"\n  Params with det grad: {len(grad_det_dict)}")
    print(f"  Params with vel grad: {len(grad_vel_dict)}")
    print(f"  Params with both:     {len(common_names)}")

    grad_det_vec = torch.cat([grad_det_dict[n] for n in common_names])
    grad_vel_vec = torch.cat([grad_vel_dict[n] for n in common_names])

    cos_sim = F.cosine_similarity(grad_det_vec.unsqueeze(0), grad_vel_vec.unsqueeze(0)).item()
    angle = torch.acos(torch.clamp(torch.tensor(cos_sim), -1, 1)).item() * 180 / 3.14159265

    print(f"\n{'='*60}")
    print(f"Gradient Analysis Results:")
    print(f"  ||grad_det|| = {grad_det_vec.norm().item():.4f}")
    print(f"  ||grad_vel|| = {grad_vel_vec.norm().item():.4f}")
    print(f"  cos(grad_det, grad_vel) = {cos_sim:.6f}")
    print(f"  angle = {angle:.2f} degrees")
    print(f"{'='*60}")

    if cos_sim < 0:
        print(f"\n  *** GRADIENT CONFLICT DETECTED! ***")
        print(f"  cos < 0 means velocity loss gradient DIRECTLY HURTS detection loss")
    elif cos_sim < 0.3:
        print(f"\n  *** WEAK ALIGNMENT ***")
        print(f"  cos < 0.3 means velocity loss gradient provides little help to detection")
    else:
        print(f"\n  *** GOOD ALIGNMENT ***")
        print(f"  Velocity and detection gradients are well aligned")

    per_layer_results = []
    for name in common_names:
        g_det = grad_det_dict[name]
        g_vel = grad_vel_dict[name]
        if g_det.norm() > 1e-8 and g_vel.norm() > 1e-8:
            cos_layer = F.cosine_similarity(g_det.unsqueeze(0), g_vel.unsqueeze(0)).item()
            per_layer_results.append((name, cos_layer, g_det.norm().item(), g_vel.norm().item()))

    print(f"\n{'='*60}")
    print(f"Per-layer gradient cosine similarity (sorted by conflict):")
    print(f"{'='*60}")
    per_layer_results.sort(key=lambda x: x[1])
    for name, cos_l, gn_det, gn_vel in per_layer_results[:20]:
        conflict = "CONFLICT" if cos_l < 0 else ("weak" if cos_l < 0.3 else "ok")
        print(f"  {name:60s} cos={cos_l:+.4f}  ||det||={gn_det:.4f}  ||vel||={gn_vel:.4f}  [{conflict}]")

    print(f"\n  ... (showing top 20 most conflicting layers out of {len(per_layer_results)})")

    num_conflict = sum(1 for _, c, _, _ in per_layer_results if c < 0)
    num_weak = sum(1 for _, c, _, _ in per_layer_results if 0 <= c < 0.3)
    num_ok = sum(1 for _, c, _, _ in per_layer_results if c >= 0.3)
    print(f"\n  Summary: {num_conflict} CONFLICT, {num_weak} weak, {num_ok} ok (out of {len(per_layer_results)} layers)")

    print(f"\n{'='*60}")
    print(f"Velocity-head only layers:")
    print(f"{'='*60}")
    vel_only = sorted(set(grad_vel_dict.keys()) - set(grad_det_dict.keys()))
    for name in vel_only:
        print(f"  {name}: ||vel||={grad_vel_dict[name].norm().item():.6f}")

    print(f"\nDetection-only layers:")
    det_only = sorted(set(grad_det_dict.keys()) - set(grad_vel_dict.keys()))
    for name in det_only:
        print(f"  {name}: ||det||={grad_det_dict[name].norm().item():.6f}")


if __name__ == "__main__":
    main()
