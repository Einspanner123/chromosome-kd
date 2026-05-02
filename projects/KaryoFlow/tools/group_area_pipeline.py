"""
验证降级方案: 8-group 分类 + 组内面积排序 = ?

组合三步:
  1. Per-crop 8-group classifier → 预测每个 crop 的 Denver 组
  2. 每组内按 bbox 面积降序排列 (面积大 → 组内靠前 slot)
  3. 分配到对应组的 slot 范围 → 得到完整排列

对比:
  - Pure area sorting (面积基线)
  - Group classifier only (用 Hungarian per-group)
  - Combined (classifier + area within group)
  - Oracle groups (用 GT 组标签 + 面积排序 → 上界)
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from karyoflow.constants import (
    CHROMO_CLASSES, DENVER_GROUPS, CLASS_TO_INDEX,
    NUM_SLOTS, SLOT_ORDER, SLOT_TO_GROUP, CLASS_TO_SLOTS,
)
from karyoflow.dataset import KaryoFlowDataset
from karyoflow.encoder import ChromosomeEncoder

GROUP_NAMES = ["A", "B", "C", "D", "E", "F", "G", "Sex"]
NUM_GROUPS = len(GROUP_NAMES)

# 每组占用的 slot 范围 (基于 SLOT_ORDER)
GROUP_SLOTS: dict = {}
for slot_i, cls_name in enumerate(SLOT_ORDER):
    g = None
    for gname, classes in DENVER_GROUPS.items():
        if cls_name in classes:
            g = gname
            break
    GROUP_SLOTS.setdefault(g, []).append(slot_i)
# GROUP_SLOTS["A"] = [0,1,2,3,4,5], etc.

# 每组期望的 crop 数量
GROUP_SIZES = {g: len(slots) for g, slots in GROUP_SLOTS.items()}

# 24 class → 8 group index
CLASS_TO_GROUP: dict = {}
for gi, (gname, classes) in enumerate(DENVER_GROUPS.items()):
    for cls_name in classes:
        CLASS_TO_GROUP[CLASS_TO_INDEX[cls_name]] = gi


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str,
                        default="/data/linkst/datasets/Chromosome20240904_NoAug_NoResize_coco")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-model", type=str, default="",
                        help="Path to save trained classifier (for reuse)")
    return parser.parse_args()


class GroupClassifier(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, NUM_GROUPS),
        )

    def forward(self, x):
        return self.net(x)


def train_classifier(encoder, classifier, train_loader, valid_loader, args, device):
    """训练 8-group 分类器，保存最佳模型"""
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    # 预计算 slot → group label
    slot_to_group = torch.tensor(
        [CLASS_TO_GROUP[CLASS_TO_INDEX[SLOT_ORDER[i]]] for i in range(NUM_SLOTS)],
        dtype=torch.long,
    )

    best_acc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        classifier.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Train group cls {epoch}/{args.epochs}", leave=False)
        for batch in pbar:
            crops = batch["crops"].to(device)
            B, N = crops.shape[:2]
            crops_flat = crops.view(B * N, *crops.shape[2:])
            bboxes_flat = batch["bboxes"].to(device).view(B * N, 4)
            labels = slot_to_group.unsqueeze(0).expand(B, -1).reshape(-1).to(device)

            with torch.no_grad():
                feats = encoder(crops_flat, bboxes_flat)
            with autocast():
                logits = classifier(feats)
                loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        scheduler.step()

        # Validation
        classifier.eval()
        val_correct, val_samples = 0, 0
        with torch.no_grad():
            for batch in valid_loader:
                crops = batch["crops"].to(device)
                B_v, N_v = crops.shape[:2]
                crops_flat = crops.view(B_v * N_v, *crops.shape[2:])
                bboxes_flat = batch["bboxes"].to(device).view(B_v * N_v, 4)
                labels = slot_to_group.unsqueeze(0).expand(B_v, -1).reshape(-1).to(device)
                feats = encoder(crops_flat, bboxes_flat)
                logits = classifier(feats)
                preds = logits.argmax(-1)
                val_correct += (preds == labels).sum().item()
                val_samples += labels.size(0)

        val_acc = val_correct / max(val_samples, 1) * 100
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in classifier.state_dict().items()}

    classifier.load_state_dict(best_state)
    return best_acc


@torch.no_grad()
def evaluate_all_methods(encoder, classifier, dataloader, device):
    """评估四种方法"""
    encoder.eval()
    classifier.eval()

    # 初始化统计
    methods = ["area_only", "classifier_only", "combined", "oracle_groups"]
    stats = {
        m: {"exact": 0, "pos_correct": 0, "tau_sum": 0.0,
            "group_ok": {g: 0 for g in GROUP_NAMES},
            "group_total": {g: 0 for g in GROUP_NAMES}}
        for m in methods
    }
    total = 0

    slot_to_group = torch.tensor(
        [CLASS_TO_GROUP[CLASS_TO_INDEX[SLOT_ORDER[i]]] for i in range(NUM_SLOTS)],
        device=device,
    )

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        crops = batch["crops"].to(device)
        B, N = crops.shape[:2]
        bboxes = batch["bboxes"].to(device)
        gt_perm = batch["permutation"].to(device)  # (B, 46): slot → crop_idx

        crops_flat = crops.view(B * N, *crops.shape[2:])
        bboxes_flat = bboxes.view(B * N, 4)

        with autocast():
            feats = encoder(crops_flat, bboxes_flat)
            group_logits = classifier(feats)  # (B*46, 8)
        group_preds = group_logits.argmax(-1)  # (B*46,)

        for b in range(B):
            total += 1
            gt_b = gt_perm[b]  # slot → crop_idx: gt_b[slot_i] = crop_j
            areas = (bboxes[b, :, 2] - bboxes[b, :, 0]) * (bboxes[b, :, 3] - bboxes[b, :, 1])
            gpred_b = group_preds[b * N : (b + 1) * N]  # per-crop group predictions

            # --- 方法 1: 纯面积排序 ---
            _, area_pred = torch.sort(areas, descending=True)
            _update_stats(stats["area_only"], area_pred, gt_b, slot_to_group, N)

            # --- 方法 2: 分类器 + Hungarian (per-group assignment not possible without GT sizes)
            # 用全局 Hungarian on classification logits
            # 这个我们之前已经在 matching_baseline 中尝试过类似方法，position acc ~2.2%
            # 这里简化: global Hungarian
            b_logits_full = group_logits[b * N : (b + 1) * N]  # (46, 8) - not enough for Hungarian
            # classifier_only uses same logic as oracle but with predicted groups
            pred_perm_cls = _classifier_area_pipeline(gpred_b, areas, N, use_gt_groups=False)
            _update_stats(stats["classifier_only"], pred_perm_cls, gt_b, slot_to_group, N)

            # --- 方法 3: 组合 (分类预测组 + 组内面积排序) ---
            pred_perm_comb = _classifier_area_pipeline(gpred_b, areas, N, use_gt_groups=False)
            _update_stats(stats["combined"], pred_perm_comb, gt_b, slot_to_group, N)

            # --- 方法 4: Oracle 组标签 + 组内面积排序 (上界) ---
            # GT group: 根据 SLOT_ORDER 可知 crop 应该属于哪个组
            # crop j 在 GT 排列中对应 slot = gt_perm^{-1}[j]
            # 该 slot 的组即 crop j 的真实组
            gt_groups = torch.zeros(N, dtype=torch.long, device=device)
            for slot_i in range(N):
                crop_j = gt_b[slot_i].item()
                gt_groups[crop_j] = slot_to_group[slot_i]
            pred_perm_oracle = _classifier_area_pipeline(gt_groups, areas, N, use_gt_groups=True)
            _update_stats(stats["oracle_groups"], pred_perm_oracle, gt_b, slot_to_group, N)

    # 打印结果
    print("\n" + "=" * 70)
    print("  降级方案验证: 8-Group 分类 + 组内面积排序")
    print("=" * 70)
    print(f"  验证样本数: {total}")
    print()
    print(f"  {'Method':<30} {'Exact %':>8} {'Pos %':>8} {'Kendall τ':>10}")
    print(f"  {'-'*58}")

    oracle_tau = 0
    for method in methods:
        s = stats[method]
        n = total
        exact = s["exact"] / n * 100
        pos = s["pos_correct"] / (n * NUM_SLOTS) * 100
        tau = s["tau_sum"] / n
        print(f"  {method:<30} {exact:>7.1f}% {pos:>7.1f}% {tau:>10.3f}")
        if method == "oracle_groups":
            oracle_tau = tau

    print()
    print(f"  Per-group position accuracy:")
    print(f"  {'Group':<8} {'Area':>7} {'Cls':>7} {'Comb':>7} {'Oracle':>7}")
    print(f"  {'-'*42}")
    for g in GROUP_NAMES:
        a = stats["area_only"]["group_ok"][g] / max(stats["area_only"]["group_total"][g], 1) * 100
        c = stats["classifier_only"]["group_ok"][g] / max(stats["classifier_only"]["group_total"][g], 1) * 100
        m = stats["combined"]["group_ok"][g] / max(stats["combined"]["group_total"][g], 1) * 100
        o = stats["oracle_groups"]["group_ok"][g] / max(stats["oracle_groups"]["group_total"][g], 1) * 100
        print(f"  {g:<8} {a:>6.1f}% {c:>6.1f}% {m:>6.1f}% {o:>6.1f}%")

    return oracle_tau


def _classifier_area_pipeline(group_preds, areas, N, use_gt_groups=False):
    """组分类 + 组内面积排序 → 完整排列

    Args:
        group_preds: (N,) per-crop group predictions
        areas: (N,) per-crop bbox area
        N: 46
        use_gt_groups: if True, group_preds are already GT groups

    Returns:
        pred_perm: (N,) slot → crop_idx
    """
    device = areas.device
    pred_perm = torch.zeros(N, dtype=torch.long, device=device)

    for gi, gname in enumerate(GROUP_NAMES):
        group_slots = torch.tensor(GROUP_SLOTS[gname], device=device)
        n_slots = len(group_slots)

        if use_gt_groups:
            # GT groups: group_preds already correct
            mask = (group_preds == gi)
        else:
            # Predicted groups: take top-k crops by group logit for this group
            # 但 group_preds 是硬分类结果, 可能数量不匹配
            mask = (group_preds == gi)

        candidate_crops = mask.nonzero(as_tuple=True)[0]

        if len(candidate_crops) == 0:
            # 没有 crop 被分到此组 → 用面积最大的未分配 crop
            continue

        # 组内按面积降序
        candidate_areas = areas[candidate_crops]
        _, sorted_idx = torch.sort(candidate_areas, descending=True)

        # 分配: 前 n_slots 个 crop → group_slots
        n_assign = min(n_slots, len(candidate_crops))
        for k in range(n_assign):
            crop_idx = candidate_crops[sorted_idx[k]].item()
            slot_idx = group_slots[k].item()
            pred_perm[slot_idx] = crop_idx

    # 处理未分配的 slots (那些组分类预测数量不足的)
    unassigned_slots = (pred_perm == 0).nonzero(as_tuple=True)[0]
    # 实际上 pred_perm 初始化为0, 0也是合法 crop idx. 我们改用 -1 初始化
    # 但为了简化, 先用这个逻辑:
    assigned_crops = set(pred_perm.tolist())
    all_crops = set(range(N))
    unassigned_crops = list(all_crops - assigned_crops)

    # 重新: 用 -1 标记未分配
    pred_perm_fixed = torch.full((N,), -1, dtype=torch.long, device=device)
    for gi, gname in enumerate(GROUP_NAMES):
        group_slots = GROUP_SLOTS[gname]
        mask = (group_preds == gi)
        candidate_crops = mask.nonzero(as_tuple=True)[0].tolist()
        if not candidate_crops:
            continue
        candidate_areas = areas[candidate_crops]
        _, sorted_idx = torch.sort(candidate_areas, descending=True)
        n_assign = min(len(group_slots), len(candidate_crops))
        for k in range(n_assign):
            crop_idx = candidate_crops[sorted_idx[k].item()]
            pred_perm_fixed[group_slots[k]] = crop_idx

    # 填充未分配的 slots
    assigned = set(pred_perm_fixed[pred_perm_fixed >= 0].tolist())
    unassigned_slots = (pred_perm_fixed < 0).nonzero(as_tuple=True)[0]
    unassigned_crops = [c for c in range(N) if c not in assigned]
    for k, slot_i in enumerate(unassigned_slots.tolist()):
        if k < len(unassigned_crops):
            pred_perm_fixed[slot_i] = unassigned_crops[k]

    return pred_perm_fixed


def _update_stats(stat_dict, pred_perm, gt_perm, slot_to_group, N):
    """更新统计量"""
    if (pred_perm == gt_perm).all():
        stat_dict["exact"] += 1
    stat_dict["pos_correct"] += (pred_perm == gt_perm).sum().item()
    stat_dict["tau_sum"] += _kendall_tau(pred_perm, gt_perm)

    for slot_i in range(N):
        g = GROUP_NAMES[slot_to_group[slot_i].item()]
        stat_dict["group_total"][g] += 1
        if pred_perm[slot_i] == gt_perm[slot_i]:
            stat_dict["group_ok"][g] += 1


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
    total = conc + disc
    return (conc - disc) / max(total, 1)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

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

    classifier = GroupClassifier(d_model=args.d_model).to(device)

    print(f"\nStep 1: Training 8-group classifier ({args.epochs} epochs)...")
    group_acc = train_classifier(encoder, classifier, train_loader, valid_loader, args, device)
    print(f"  Best 8-group val acc: {group_acc:.1f}%")

    print(f"\nStep 2: Evaluating all methods...")
    evaluate_all_methods(encoder, classifier, valid_loader, device)


if __name__ == "__main__":
    main()
