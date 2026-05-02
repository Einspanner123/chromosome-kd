"""
训练多样性实时测量工具

在验证集上运行模型前向传播，统计不同耦合策略下的条件速度多样性指标：
1. 条件速度熵 H(V|X_t): 衡量给定中间状态时速度场的不确定性
2. 分配多样性 D_assign: 每个 GT 被分配的噪声框数量的熵
3. 传输代价 C_transport: 噪声→GT 的平均传输距离
4. CAM 度: Argmax vs Stochastic 的多样性差异

用法:
    python tools/measure_diversity.py CONFIG CHECKPOINT [--out OUT_FILE]

示例:
    # Random baseline
    python tools/measure_diversity.py \
        projects/LDMDet/configs/ldmdet_flowdet_adaln.py \
        work_dirs/ldmdet_flowdet_adaln/best_coco_bbox_mAP_epoch_56.pth \
        --out diversity_random.json

    # Stochastic eps5
    python tools/measure_diversity.py \
        projects/LDMDet/configs/ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5.py \
        work_dirs/ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5/best_coco_bbox_mAP_epoch_86.pth \
        --out diversity_stochastic_eps5.json
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from mmdet.utils import register_all_modules
from mmengine.config import Config
from mmengine.runner import Runner


def compute_sinkhorn_transport(noise, gt_diffusion, epsilon=1.0, num_iters=20):
    cost = torch.cdist(noise, gt_diffusion, p=2)
    N, K = cost.shape
    a = torch.ones(N, device=noise.device) / N
    proposals_per_gt = max(N // max(K, 1), 1)
    gt_mass = torch.full((K,), proposals_per_gt / N, device=noise.device)
    b = gt_mass / gt_mass.sum()
    log_K_mat = -cost / epsilon
    log_u = torch.zeros(N, device=noise.device)
    log_v = torch.zeros(K, device=noise.device)
    for _ in range(num_iters):
        log_u = torch.log(a + 1e-10) - torch.logsumexp(
            log_K_mat + log_v.unsqueeze(0), dim=1
        )
        log_v = torch.log(b + 1e-10) - torch.logsumexp(
            log_K_mat + log_u.unsqueeze(1), dim=0
        )
    transport = torch.exp(log_u.unsqueeze(1) + log_K_mat + log_v.unsqueeze(0))
    return transport, cost


def compute_diversity_metrics(transport, cost, noise, gt_diffusion):
    metrics = {}
    N, K = transport.shape

    row_probs = transport / transport.sum(dim=1, keepdim=True)

    argmax_idx = transport.argmax(dim=1)
    stochastic_idx = torch.multinomial(row_probs, 1).squeeze(-1)

    assign_counts_argmax = torch.bincount(argmax_idx, minlength=K).float()
    assign_counts_argmax = assign_counts_argmax / assign_counts_argmax.sum()
    assign_counts_argmax = assign_counts_argmax[assign_counts_argmax > 0]
    metrics["assign_entropy_argmax"] = -(
        assign_counts_argmax * assign_counts_argmax.log()
    ).sum().item()

    assign_counts_stoch = torch.bincount(stochastic_idx, minlength=K).float()
    assign_counts_stoch = assign_counts_stoch / assign_counts_stoch.sum()
    assign_counts_stoch = assign_counts_stoch[assign_counts_stoch > 0]
    metrics["assign_entropy_stochastic"] = -(
        assign_counts_stoch * assign_counts_stoch.log()
    ).sum().item()

    metrics["assign_entropy_random"] = np.log(K)

    row_entropies = -(
        row_probs * torch.clamp(row_probs, min=1e-10).log()
    ).sum(dim=1)
    metrics["row_entropy_mean"] = row_entropies.mean().item()
    metrics["row_entropy_max"] = np.log(K)

    argmax_cost = cost[torch.arange(N), argmax_idx].mean().item()
    stochastic_cost = cost[torch.arange(N), stochastic_idx].mean().item()
    random_cost = cost.mean(dim=1).mean().item()
    metrics["transport_cost_argmax"] = argmax_cost
    metrics["transport_cost_stochastic"] = stochastic_cost
    metrics["transport_cost_random"] = random_cost

    argmax_diversity = row_entropies[torch.arange(N), ].scatter_(
        0, argmax_idx.unsqueeze(0), 0
    )
    metrics["cam_degree"] = (
        (metrics["assign_entropy_stochastic"] - metrics["assign_entropy_argmax"])
        / max(metrics["assign_entropy_random"] - metrics["assign_entropy_argmax"], 1e-10)
    )

    return metrics


def measure_on_dataloader(model, dataloader, device, max_batches=100):
    model.eval()
    all_metrics = {
        "assign_entropy_argmax": [],
        "assign_entropy_stochastic": [],
        "assign_entropy_random": [],
        "row_entropy_mean": [],
        "transport_cost_argmax": [],
        "transport_cost_stochastic": [],
        "transport_cost_random": [],
    }

    bbox_head = model.bbox_head
    ot_coupling = bbox_head.ot_coupling
    ot_matcher = getattr(bbox_head, "ot_matcher", "nearest")
    ot_epsilon = getattr(bbox_head, "ot_epsilon", 1.0)
    ot_num_iters = getattr(bbox_head, "ot_num_iters", 20)
    ot_sample = getattr(bbox_head, "ot_sample", False)
    snr_scale = bbox_head.snr_scale

    with torch.no_grad():
        for batch_idx, data_batch in enumerate(tqdm(dataloader, desc="Measuring")):
            if batch_idx >= max_batches:
                break

            data = model.data_preprocessor(data_batch, True)
            gt_bboxes = data_batch["data_samples"]
            device_curr = data["inputs"][0].device

            for i, sample in enumerate(gt_bboxes):
                gt_instances = sample.gt_instances
                num_gt = len(gt_instances.bboxes)
                if num_gt < 2:
                    continue

                h, w = sample.img_shape[:2]
                scale = gt_instances.bboxes.new_tensor([w, h, w, h])
                norm_bboxes = gt_instances.bboxes / scale

                norm_gt_cxcywh = bbox_xyxy_to_cxcywh(norm_bboxes)
                gt_diffusion = (norm_gt_cxcywh * 2 - 1) * snr_scale

                noise = torch.randn(bbox_head.num_proposals, 4, device=device_curr)

                if ot_coupling and ot_matcher == "sinkhorn":
                    transport, cost = compute_sinkhorn_transport(
                        noise, gt_diffusion, ot_epsilon, ot_num_iters
                    )
                    metrics = compute_diversity_metrics(
                        transport, cost, noise, gt_diffusion
                    )
                elif ot_coupling:
                    cost = torch.cdist(noise, gt_diffusion, p=2)
                    matched_idx = cost.argmin(dim=1)
                    N = noise.shape[0]
                    K = gt_diffusion.shape[0]
                    assign_counts = torch.bincount(matched_idx, minlength=K).float()
                    assign_counts = assign_counts / assign_counts.sum()
                    assign_counts = assign_counts[assign_counts > 0]
                    metrics = {
                        "assign_entropy_argmax": -(
                            assign_counts * assign_counts.log()
                        )
                        .sum()
                        .item(),
                        "assign_entropy_stochastic": -(
                            assign_counts * assign_counts.log()
                        )
                        .sum()
                        .item(),
                        "assign_entropy_random": np.log(K),
                        "row_entropy_mean": 0.0,
                        "transport_cost_argmax": cost[torch.arange(N), matched_idx]
                        .mean()
                        .item(),
                        "transport_cost_stochastic": cost[torch.arange(N), matched_idx]
                        .mean()
                        .item(),
                        "transport_cost_random": cost.mean(dim=1).mean().item(),
                    }
                else:
                    K = gt_diffusion.shape[0]
                    N = noise.shape[0]
                    cost = torch.cdist(noise, gt_diffusion, p=2)
                    metrics = {
                        "assign_entropy_argmax": np.log(K),
                        "assign_entropy_stochastic": np.log(K),
                        "assign_entropy_random": np.log(K),
                        "row_entropy_mean": np.log(K),
                        "transport_cost_argmax": cost.mean(dim=1).mean().item(),
                        "transport_cost_stochastic": cost.mean(dim=1).mean().item(),
                        "transport_cost_random": cost.mean(dim=1).mean().item(),
                    }

                for k, v in metrics.items():
                    if k in all_metrics:
                        all_metrics[k].append(v)

    result = {}
    for k, v in all_metrics.items():
        if v:
            result[k] = {"mean": float(np.mean(v)), "std": float(np.std(v)), "count": len(v)}

    result["config_info"] = {
        "ot_coupling": ot_coupling,
        "ot_matcher": ot_matcher,
        "ot_epsilon": ot_epsilon,
        "ot_sample": ot_sample,
    }

    return result


def bbox_xyxy_to_cxcywh(bboxes):
    cx = (bboxes[:, 0] + bboxes[:, 2]) / 2
    cy = (bboxes[:, 1] + bboxes[:, 3]) / 2
    w = bboxes[:, 2] - bboxes[:, 0]
    h = bboxes[:, 3] - bboxes[:, 1]
    return torch.stack([cx, cy, w, h], dim=-1)


def main():
    parser = argparse.ArgumentParser(description="Measure training diversity metrics")
    parser.add_argument("config", help="Config file path")
    parser.add_argument("checkpoint", help="Checkpoint file path")
    parser.add_argument("--out", default="diversity_metrics.json", help="Output JSON file")
    parser.add_argument("--max-batches", type=int, default=100, help="Max batches to process")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device")
    args = parser.parse_args()

    register_all_modules()

    cfg = Config.fromfile(args.config)
    cfg.load_from = args.checkpoint
    cfg.work_dir = "/tmp/diversity_measure"

    runner = Runner.from_cfg(cfg)
    runner.load_or_resume()

    model = runner.model
    model.eval()

    dataloader = runner.val_dataloader

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    result = measure_on_dataloader(model, dataloader, device, args.max_batches)

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nDiversity metrics saved to {args.out}")
    print("\nSummary:")
    for k, v in result.items():
        if isinstance(v, dict) and "mean" in v:
            print(f"  {k}: {v['mean']:.4f} ± {v['std']:.4f} (n={v['count']})")
        elif isinstance(v, dict):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
