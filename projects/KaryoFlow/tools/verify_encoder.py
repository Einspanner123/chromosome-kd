"""
ChromosomeEncoder 分类能力基准验证

验证 KaryoFlow 的 encoder 是否能从染色体裁剪图中提取有效的类别特征。
如果 24-way 分类 acc > 80% → encoder 没问题，问题在排列学习
如果 acc < 50% → encoder 本身是瓶颈，需换 backbone 或增大输入分辨率
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from karyoflow.constants import CHROMO_CLASSES, CLASS_TO_INDEX, NUM_CLASSES, NUM_SLOTS, SLOT_ORDER
from karyoflow.dataset import KaryoFlowDataset
from karyoflow.encoder import ChromosomeEncoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str,
                        default="/data/linkst/datasets/Chromosome20240904_NoAug_NoResize_coco")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 数据集: 使用 KaryoFlow 的排列标注来获取每个裁剪图的类别
    data_root = args.data_root
    train_coco = f"{data_root}/train/_annotations.coco.json"
    valid_coco = f"{data_root}/valid/_annotations.coco.json"
    train_kf   = f"{data_root}/train_karyoflow.json"
    valid_kf   = f"{data_root}/valid_karyoflow.json"

    def _image_dir(split):
        d = f"{data_root}/{split}"
        return f"{d}/images" if Path(f"{d}/images").is_dir() else d

    # 使用 KaryoFlowDataset 并提取每个裁剪图的类别
    print("Loading datasets...")
    train_ds = KaryoFlowDataset(
        coco_json=train_coco, karyoflow_json=train_kf,
        image_dir=_image_dir("train"), crop_size=args.crop_size, augment=True,
    )
    valid_ds = KaryoFlowDataset(
        coco_json=valid_coco, karyoflow_json=valid_kf,
        image_dir=_image_dir("valid"), crop_size=args.crop_size, augment=False,
    )
    print(f"  Train: {len(train_ds)} images × 46 = {len(train_ds)*46} crops")
    print(f"  Valid: {len(valid_ds)} images × 46 = {len(valid_ds)*46} crops")

    # 构建编码器 + 分类头
    encoder = ChromosomeEncoder(
        d_model=args.d_model, crop_size=args.crop_size,
        use_geometry=True, pretrained=True,
    ).to(device)

    classifier = nn.Sequential(
        nn.LayerNorm(args.d_model),
        nn.Linear(args.d_model, args.d_model),
        nn.GELU(),
        nn.Linear(args.d_model, NUM_CLASSES),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    all_params = list(encoder.parameters()) + list(classifier.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()

    # 数据加载器: 直接加载每张图的 46 个裁剪图, 展开成 (B*46, ...) 的 batch
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    print(f"\nTraining {args.epochs} epochs...")
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        classifier.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for batch in pbar:
            crops = batch["crops"].to(device)    # (B, 46, C, H, W)
            B, N = crops.shape[:2]
            # 展平: (B*46, ...)
            crops_flat = crops.view(B * N, *crops.shape[2:])
            bboxes_flat = batch["bboxes"].to(device).view(B * N, 4)

            # 从排列标注推断每个裁剪图的类别
            # perm: (B, N) → slot_i → detection_j
            # 我们需要 detection_j 的类别。在标注中 det_classes[perm[b, i]] 给出类别。
            # 但由于标注结构限制，我们用 slot 顺序映射到类别:
            # SLOT_ORDER 定义了 slots[0..45] 分别对应哪些 class_name
            # 在 KaryoFlowDataset 中，crops 按 slot 顺序排列 (slot 0 到 slot 45)
            # 所以 crops[b, i] 的类别 = SLOT_ORDER[i] 对应的 class_id
            slot_classes = torch.tensor(
                [CLASS_TO_INDEX[SLOT_ORDER[i]] for i in range(N)],
                device=device,
            ).unsqueeze(0).expand(B, -1)  # (B, N)
            labels = slot_classes.reshape(-1)  # (B*46,)

            with autocast():
                feats = encoder(crops_flat, bboxes_flat)  # (B*46, d)
                logits = classifier(feats)                  # (B*46, 24)
                loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            preds = logits.argmax(-1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            pbar.set_postfix(loss=f"{total_loss/(pbar.n+1):.3f}",
                             acc=f"{total_correct/max(total_samples,1)*100:.1f}%")

        scheduler.step()
        train_acc = total_correct / max(total_samples, 1) * 100

        # 验证
        encoder.eval()
        classifier.eval()
        val_correct, val_samples = 0, 0
        with torch.no_grad():
            for batch in valid_loader:
                crops = batch["crops"].to(device)
                B_v, N_v = crops.shape[:2]
                crops_flat = crops.view(B_v * N_v, *crops.shape[2:])
                bboxes_flat = batch["bboxes"].to(device).view(B_v * N_v, 4)
                # recompute for this batch size
                labels_v = torch.tensor(
                    [CLASS_TO_INDEX[SLOT_ORDER[i]] for i in range(N_v)],
                    device=device,
                ).unsqueeze(0).expand(B_v, -1).reshape(-1)

                feats = encoder(crops_flat, bboxes_flat)
                logits = classifier(feats)
                val_correct += (logits.argmax(-1) == labels_v).sum().item()
                val_samples += labels_v.size(0)
        val_acc = val_correct / max(val_samples, 1) * 100

        print(f"  Epoch {epoch:3d}: train_acc={train_acc:.1f}%  val_acc={val_acc:.1f}%  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        if val_acc > best_acc:
            best_acc = val_acc

    print(f"\n=== Result ===")
    print(f"Best val accuracy: {best_acc:.1f}%")
    if best_acc > 80:
        print("✓ PASS — Encoder produces good features, problem is in permutation learning")
    elif best_acc > 50:
        print("△ MARGINAL — Encoder features are usable but weak, consider larger backbone or crop_size")
    else:
        print("✗ FAIL — Encoder is the bottleneck, need stronger backbone or higher resolution")


if __name__ == "__main__":
    main()
