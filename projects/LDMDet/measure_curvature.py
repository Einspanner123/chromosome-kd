import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from mmdet.apis import init_detector
from mmdet.utils import ConfigType


def measure_curvature_single(model, config, num_samples=50, num_t_steps=20, device="cuda"):
    bs = 1
    num_proposals = model.bbox_head.num_proposals
    snr_scale = model.bbox_head.snr_scale
    diffusion_type = model.bbox_head.diffusion_type

    t_values = torch.linspace(0.05, 0.95, num_t_steps, device=device)

    all_curvatures = []
    all_velocities = []
    all_velocity_norms = []

    for _ in range(num_samples):
        noise = torch.randn(bs, num_proposals, 4, device=device)
        x_start = torch.randn(bs, num_proposals, 4, device=device) * snr_scale

        velocities = []
        with torch.no_grad():
            for t_val in t_values:
                t = torch.full((bs,), t_val, device=device)

                if diffusion_type == "rectified_flow":
                    t_view = t.view(-1, 1, 1)
                    x_t = (1.0 - t_view) * x_start + t_view * noise
                else:
                    t_int = (t_val * model.bbox_head.timesteps).long()
                    t_input = torch.full((bs,), t_int, device=device, dtype=torch.long)
                    x_t = model.bbox_head.q_sample(x_start.squeeze(0), t_input)

                curr_bboxes = model.bbox_head._raw_to_xyxy(
                    x_t,
                    [type("M", (), {"img_shape": (1024, 1024), "ori_shape": None, "scale_factor": None})],
                )

                t_input = t * model.bbox_head.timesteps if diffusion_type == "rectified_flow" else t_int
                cls_logits_seq, pred_bboxes_seq, _, _ = model.bbox_head(
                    model.extract_feat(torch.randn(1, 3, 1024, 1024, device=device)),
                    curr_bboxes,
                    t_input if isinstance(t_input, torch.Tensor) else t_input,
                )

                x0_raw = model.bbox_head._xyxy_to_raw(
                    pred_bboxes_seq[-1],
                    [type("M", (), {"img_shape": (1024, 1024), "ori_shape": None, "scale_factor": None})],
                )

                if diffusion_type == "rectified_flow":
                    t_clamp = max(t_val, 1e-5)
                    v = (x_t - x0_raw) / t_clamp
                else:
                    v = model.bbox_head.predict_noise_from_start(x_t.squeeze(0), t_input, x0_raw.squeeze(0))
                    v = v.unsqueeze(0)

                velocities.append(v)

        velocities = torch.stack(velocities)
        v_norms = velocities.norm(dim=-1).mean(dim=-1).mean(dim=-1)
        all_velocity_norms.append(v_norms.cpu().numpy())

        if len(velocities) > 1:
            dv = velocities[1:] - velocities[:-1]
            dt = (t_values[1:] - t_values[:-1]).view(-1, 1, 1)
            curvature = (dv / dt).norm(dim=-1).mean(dim=-1).mean(dim=-1)
            total_curvature = curvature.sum().item()
            all_curvatures.append(total_curvature)

        all_velocities.append(velocities.cpu().numpy())

    results = {
        "diffusion_type": diffusion_type,
        "num_samples": num_samples,
        "num_t_steps": num_t_steps,
        "total_curvature_mean": float(np.mean(all_curvatures)) if all_curvatures else None,
        "total_curvature_std": float(np.std(all_curvatures)) if all_curvatures else None,
        "velocity_norm_per_t": [float(x) for x in np.mean(all_velocity_norms, axis=0)],
        "t_values": t_values.cpu().numpy().tolist(),
    }

    return results


def measure_curvature_with_gt(model, config, dataloader, num_batches=10, num_t_steps=20, device="cuda"):
    from projects.LDMDet.mods.structures import ImageMeta

    bs_limit = 2
    num_proposals = model.bbox_head.num_proposals
    snr_scale = model.bbox_head.snr_scale
    diffusion_type = model.bbox_head.diffusion_type

    t_values = torch.linspace(0.05, 0.95, num_t_steps, device=device)

    all_curvatures_ot = []
    all_curvatures_random = []
    all_transport_cost_ot = []
    all_transport_cost_random = []

    batch_count = 0
    for data_batch in dataloader:
        if batch_count >= num_batches:
            break
        batch_count += 1

        batch_inputs = data_batch["inputs"].to(device)
        features = model.extract_feat(batch_inputs)

        for b in range(min(batch_inputs.shape[0], bs_limit)):
            gt_bboxes = data_batch["data_samples"][b].gt_instances.bboxes.to(device)
            gt_labels = data_batch["data_samples"][b].gt_instances.labels.to(device)
            h, w = data_batch["data_samples"][b].metainfo["img_shape"][:2]

            if gt_bboxes.shape[0] == 0:
                continue

            scale = gt_bboxes.new_tensor([w, h, w, h])
            norm_bboxes = gt_bboxes / scale
            from projects.LDMDet.mods.utils import bbox_xyxy_to_cxcywh
            norm_gt_cxcywh = bbox_xyxy_to_cxcywh(norm_bboxes)
            gt_diffusion = (norm_gt_cxcywh * 2 - 1) * snr_scale

            K = gt_diffusion.shape[0]
            N = num_proposals

            noise = torch.randn(N, 4, device=device)

            cost = torch.cdist(noise, gt_diffusion, p=2)
            ot_idx = cost.argmin(dim=1)
            x_start_ot = gt_diffusion[ot_idx]

            random_idx = torch.randint(0, K, (N,), device=device)
            x_start_random = gt_diffusion[random_idx]

            v_ot = noise - x_start_ot
            v_random = noise - x_start_random

            cost_ot = v_ot.norm(dim=-1).mean().item()
            cost_random = v_random.norm(dim=-1).mean().item()
            all_transport_cost_ot.append(cost_ot)
            all_transport_cost_random.append(cost_random)

            img_meta = ImageMeta(img_shape=(h, w), ori_shape=None, scale_factor=None)
            img_metas = [img_meta]

            curvatures_ot = []
            curvatures_random = []

            for t_val in t_values:
                t = torch.full((1,), t_val, device=device)
                t_view = t.view(-1, 1, 1)

                x_t_ot = (1.0 - t_view) * x_start_ot.unsqueeze(0) + t_view * noise.unsqueeze(0)
                x_t_random = (1.0 - t_view) * x_start_random.unsqueeze(0) + t_view * noise.unsqueeze(0)

                curr_bboxes_ot = model.bbox_head._raw_to_xyxy(x_t_ot, img_metas)
                curr_bboxes_random = model.bbox_head._raw_to_xyxy(x_t_random, img_metas)

                t_input = t * model.bbox_head.timesteps

                with torch.no_grad():
                    _, pred_ot, _, _ = model.bbox_head(features, curr_bboxes_ot, t_input)
                    _, pred_random, _, _ = model.bbox_head(features, curr_bboxes_random, t_input)

                x0_ot = model.bbox_head._xyxy_to_raw(pred_ot[-1], img_metas)
                x0_random = model.bbox_head._xyxy_to_raw(pred_random[-1], img_metas)

                t_clamp = max(t_val, 1e-5)
                v_pred_ot = (x_t_ot - x0_ot) / t_clamp
                v_pred_random = (x_t_random - x0_random) / t_clamp

                curvatures_ot.append(v_pred_ot)
                curvatures_random.append(v_pred_random)

            curvatures_ot = torch.stack(curvatures_ot)
            curvatures_random = torch.stack(curvatures_random)

            if len(curvatures_ot) > 1:
                dv_ot = curvatures_ot[1:] - curvatures_ot[:-1]
                dv_random = curvatures_random[1:] - curvatures_random[:-1]
                dt = (t_values[1:] - t_values[:-1]).view(-1, 1, 1)

                total_curv_ot = (dv_ot / dt).norm(dim=-1).mean(dim=-1).mean(dim=-1).sum().item()
                total_curv_random = (dv_random / dt).norm(dim=-1).mean(dim=-1).mean(dim=-1).sum().item()

                all_curvatures_ot.append(total_curv_ot)
                all_curvatures_random.append(total_curv_random)

    results = {
        "curvature_ot_mean": float(np.mean(all_curvatures_ot)) if all_curvatures_ot else None,
        "curvature_ot_std": float(np.std(all_curvatures_ot)) if all_curvatures_ot else None,
        "curvature_random_mean": float(np.mean(all_curvatures_random)) if all_curvatures_random else None,
        "curvature_random_std": float(np.std(all_curvatures_random)) if all_curvatures_random else None,
        "transport_cost_ot_mean": float(np.mean(all_transport_cost_ot)) if all_transport_cost_ot else None,
        "transport_cost_random_mean": float(np.mean(all_transport_cost_random)) if all_transport_cost_random else None,
        "num_samples": len(all_curvatures_ot),
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Measure ODE path curvature for diffusion detectors")
    parser.add_argument("--config", required=True, help="Config file path")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint file path")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--num-t-steps", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="curvature_results.json")
    parser.add_argument("--with-data", action="store_true", help="Use real data for measurement")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = init_detector(args.config, args.checkpoint or "", device=device)
    model.eval()

    print(f"Model: {args.config}")
    print(f"Diffusion type: {model.bbox_head.diffusion_type}")
    print(f"RF schedule: {getattr(model.bbox_head, 'rf_schedule', 'N/A')}")
    print(f"OT coupling: {getattr(model.bbox_head, 'ot_coupling', False)}")
    print()

    results = measure_curvature_single(
        model, args.config,
        num_samples=args.num_samples,
        num_t_steps=args.num_t_steps,
        device=device,
    )

    print("=" * 60)
    print("CURVATURE MEASUREMENT RESULTS")
    print("=" * 60)
    print(f"Diffusion type: {results['diffusion_type']}")
    if results["total_curvature_mean"] is not None:
        print(f"Total curvature: {results['total_curvature_mean']:.4f} ± {results['total_curvature_std']:.4f}")
    print(f"Velocity norm per t:")
    for i, (t, v) in enumerate(zip(results["t_values"], results["velocity_norm_per_t"])):
        print(f"  t={t:.3f}: ||v||={v:.4f}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
