"""
Matching-Based Karyotype Arrangement Baseline

直接预测每个 crop 的目标槽位索引 (0~45)，用 Hungarian 保证排列合法性。
避免 slot embedding 的相似度匹配问题 — 改为每个 crop 独立分类 + 匈牙利后处理。

策略:
  1. Per-crop classifier: crop_feat → MLP → 46-way logits → slot prediction
  2. 训练: CE loss (无视排列约束), Hungarian 仅用于推理/评估
  3. 评估: exact match, position acc, Kendall tau, per-group acc
"""

import argparse
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from karyoflow.constants import NUM_SLOTS, SLOT_ORDER, SLOT_TO_GROUP, CLASS_TO_SLOTS
from karyoflow.dataset import KaryoFlowDataset
from karyoflow.encoder import ChromosomeEncoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str,
                        default="/data/linkst/datasets/Chromosome20240904_NoAug_NoResize_coco")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--encoder-lr", type=float, default=1e-4,
                        help="lower LR for pretrained encoder")
    parser.add_argument("--freeze-encoder", type=int, default=15,
                        help="freeze encoder for first N epochs")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class PerCropSlotClassifier(nn.Module):
    """对每个 crop 独立预测其目标 slot index (0~45)"""

    def __init__(self, d_model=256, num_slots=NUM_SLOTS):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, num_slots),
        )

    def forward(self, crop_features):
        """crop_features: (N, d_model) → logits: (N, num_slots)"""
        return self.classifier(crop_features)


@torch.no_grad()
def evaluate(encoder, classifier, dataloader, device) -> Dict:
    encoder.eval()
    classifier.eval()

    total = 0
    exact_match = 0
    correct_positions = 0
    kendall_tau_sum = 0.0
    per_group_correct = {}
    per_group_total = {}

    for batch in dataloader:
        crops = batch["crops"].to(device)
        B, N = crops.shape[:2]
        bboxes = batch["bboxes"].to(device)
        gt_perm = batch["permutation"].to(device)  # (B, 46): slot → crop_idx

        crops_flat = crops.view(B * N, *crops.shape[2:])
        bboxes_flat = bboxes.view(B * N, 4)

        feats = encoder(crops_flat, bboxes_flat)   # (B*46, d)
        logits = classifier(feats)                   # (B*46, 46)

        for b in range(B):
            b_logits = logits[b * N : (b + 1) * N]   # (46, 46)
            gt_b = gt_perm[b]                         # (46,) slot → crop

            # Hungarian matching on logits as cost
            cost = -b_logits.cpu().numpy()            # (46, 46)
            row_ind, col_ind = linear_sum_assignment(cost)
            pred = torch.tensor(col_ind, device=device)  # slot → crop

            # 简易 greedy 也测一下
            total += 1
            if (pred == gt_b).all():
                exact_match += 1
            correct_positions += (pred == gt_b).sum().item()

            # Per-group accuracy
            for slot_i in range(N):
                gt_group = SLOT_TO_GROUP.get(slot_i, "?")
                per_group_correct[gt_group] = per_group_correct.get(gt_group, 0)
                per_group_total[gt_group] = per_group_total.get(gt_group, 0) + 1
                if pred[slot_i].item() == gt_b[slot_i].item():
                    per_group_correct[gt_group] += 1

            kendall_tau_sum += _kendall_tau(pred, gt_b)

    n = total
    return {
        "num_samples": n,
        "exact_match_acc": exact_match / n * 100,
        "position_acc": correct_positions / (n * NUM_SLOTS) * 100,
        "kendall_tau": kendall_tau_sum / n,
        "per_group_acc": {
            g: per_group_correct[g] / max(per_group_total[g], 1) * 100
            for g in sorted(per_group_correct.keys())
        },
    }


def _kendall_tau(pred, gt):
    N = pred.shape[0]
    if N < 2:
        return 1.0
    pred, gt = pred.long(), gt.long()
    conc, disc = 0, 0
    for i in range(N):
        for j in range(i + 1, N):
            if (pred[i] < pred[j]).item() == (gt[i] < gt[j]).item():
                conc += 1
            else:
                disc += 1
    total_pairs = conc + disc
    return (conc - disc) / max(total_pairs, 1)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  crop_size: {args.crop_size}  freeze_encoder: {args.freeze_encoder}ep")

    data_root = args.data_root
    train_coco = f"{data_root}/train/_annotations.coco.json"
    valid_coco = f"{data_root}/valid/_annotations.coco.json"
    train_kf = f"{data_root}/train_karyoflow.json"
    valid_kf = f"{data_root}/valid_karyoflow.json"

    def _image_dir(split):
        d = f"{data_root}/{split}"
        return f"{d}/images" if Path(f"{d}/images").is_dir() else d

    print("Loading datasets...")
    train_ds = KaryoFlowDataset(
        coco_json=train_coco, karyoflow_json=train_kf,
        image_dir=_image_dir("train"), crop_size=args.crop_size, augment=True,
    )
    valid_ds = KaryoFlowDataset(
        coco_json=valid_coco, karyoflow_json=valid_kf,
        image_dir=_image_dir("valid"), crop_size=args.crop_size, augment=False,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
    print(f"  Train: {len(train_ds)}  Valid: {len(valid_ds)}")

    encoder = ChromosomeEncoder(
        d_model=args.d_model, crop_size=args.crop_size,
        use_geometry=True, pretrained=True,
    ).to(device)
    classifier = PerCropSlotClassifier(d_model=args.d_model).to(device)

    # 分离参数组
    encoder_params = list(encoder.parameters())
    classifier_params = list(classifier.parameters())
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": args.encoder_lr},
        {"params": classifier_params, "lr": args.lr},
    ], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    criterion = nn.CrossEntropyLoss()

    print(f"\nTraining {args.epochs} epochs...")
    print(f"  Encoder frozen for first {args.freeze_encoder} epochs")
    best_pos_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # Unfreeze encoder after freeze period
        if epoch == args.freeze_encoder + 1:
            print(f"  Epoch {epoch}: unfreezing encoder...")
            for p in encoder.parameters():
                p.requires_grad = True
        elif epoch == 1:
            for p in encoder.parameters():
                p.requires_grad = False

        encoder.train()
        classifier.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for batch in pbar:
            crops = batch["crops"].to(device)
            B, N = crops.shape[:2]
            bboxes = batch["bboxes"].to(device)
            gt_perm = batch["permutation"].to(device)

            # Flatten
            crops_flat = crops.view(B * N, *crops.shape[2:])
            bboxes_flat = bboxes.view(B * N, 4)

            # 目标: 对每个 crop 预测其目标 slot
            # 反推: gt_perm[slot] = crop_idx → crop_target[crop_idx] = slot
            crop_targets = torch.zeros(B * N, dtype=torch.long, device=device)
            for bi in range(B):
                for si in range(N):
                    ci = gt_perm[bi, si].item()
                    crop_targets[bi * N + ci] = si

            feats = encoder(crops_flat, bboxes_flat)
            logits = classifier(feats)  # (B*46, 46)
            loss = criterion(logits, crop_targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(-1)
            total_correct += (preds == crop_targets).sum().item()
            total_samples += crop_targets.size(0)
            pbar.set_postfix(loss=f"{total_loss/(pbar.n+1):.3f}",
                             raw_acc=f"{total_correct/max(total_samples,1)*100:.1f}%")

        scheduler.step()

        # Evaluate (with Hungarian)
        val_results = evaluate(encoder, classifier, valid_loader, device)
        lr_info = f"e={optimizer.param_groups[0]['lr']:.2e} c={optimizer.param_groups[1]['lr']:.2e}"
        print(f"  Epoch {epoch:3d}: raw_train_acc={total_correct/max(total_samples,1)*100:.1f}%  "
              f"val_pos={val_results['position_acc']:.1f}%  "
              f"val_exact={val_results['exact_match_acc']:.1f}%  "
              f"val_tau={val_results['kendall_tau']:.3f}  "
              f"lr=[{lr_info}]")

        if val_results["position_acc"] > best_pos_acc:
            best_pos_acc = val_results["position_acc"]

    # Final evaluation
    print("\n=== Final Results ===")
    val_results = evaluate(encoder, classifier, valid_loader, device)
    print(f"  Best position accuracy:    {best_pos_acc:.1f}%")
    print(f"  Exact match:               {val_results['exact_match_acc']:.1f}%")
    print(f"  Position accuracy:         {val_results['position_acc']:.1f}%")
    print(f"  Kendall tau:               {val_results['kendall_tau']:.3f}")
    print(f"  Per-group:")
    for g, acc in val_results["per_group_acc"].items():
        print(f"    Group {g:>4s}: {acc:.1f}%")

    # Comparison to area-based heuristic
    print("\n=== Area-Based Heuristic Baseline ===")
    area_results = evaluate_area_baseline(valid_loader, device)
    print(f"  Position accuracy:         {area_results['position_acc']:.1f}%")
    print(f"  Kendall tau:               {area_results['kendall_tau']:.3f}")
    print(f"  Per-group:")
    for g, acc in area_results["per_group_acc"].items():
        print(f"    Group {g:>4s}: {acc:.1f}%")


@torch.no_grad()
def evaluate_area_baseline(dataloader, device) -> Dict:
    """纯面积启发式: 按 bbox 面积降序排列 (模拟 Denver 组排序)

    在标准核型中，染色体按 Denver 组 A→G→Sex 排列，面积递减。
    这个基线可用于评估"仅靠简单规则能排多准"。
    """
    total = 0
    exact_match = 0
    correct_positions = 0
    kendall_tau_sum = 0.0
    per_group_correct = {}
    per_group_total = {}

    for batch in dataloader:
        bboxes = batch["bboxes"]  # (B, 46, 4)
        gt_perm = batch["permutation"]
        B, N = bboxes.shape[:2]

        for b in range(B):
            areas = (bboxes[b, :, 2] - bboxes[b, :, 0]) * (bboxes[b, :, 3] - bboxes[b, :, 1])
            # 面积降序 → 排列 (最大的 → slot 0, 次大 → slot 1, ...)
            _, area_pred = torch.sort(areas, descending=True)
            # area_pred[i] = crop index that goes to slot i

            gt_b = gt_perm[b]
            total += 1
            if (area_pred == gt_b).all():
                exact_match += 1
            correct_positions += (area_pred == gt_b).sum().item()

            for slot_i in range(N):
                gt_group = SLOT_TO_GROUP.get(slot_i, "?")
                per_group_correct[gt_group] = per_group_correct.get(gt_group, 0)
                per_group_total[gt_group] = per_group_total.get(gt_group, 0) + 1
                if area_pred[slot_i].item() == gt_b[slot_i].item():
                    per_group_correct[gt_group] += 1

            kendall_tau_sum += _kendall_tau(area_pred, gt_b)

    n = total
    return {
        "num_samples": n,
        "exact_match_acc": exact_match / n * 100,
        "position_acc": correct_positions / (n * N) * 100,
        "kendall_tau": kendall_tau_sum / n,
        "per_group_acc": {
            g: per_group_correct[g] / max(per_group_total[g], 1) * 100
            for g in sorted(per_group_correct.keys())
        },
    }


if __name__ == "__main__":
    main()
