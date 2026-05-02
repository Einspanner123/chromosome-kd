"""
KaryoFlow 独立训练脚本 (纯 PyTorch, 不依赖 MMDet)

性能优化:
  - AMP (混合精度)            : bs=128 时内存仅 6.5GB，吞吐 ~200 img/s
  - non_blocking 数据传输     : CPU→GPU 与计算并行
  - num_workers 自动设置      : 充分利用 CPU 核心
  - prefetch_factor=2         : 预取 2 个 batch

用法:
    cd projects/KaryoFlow
    python train.py --data-root /data/linkst/datasets/Chromosome20240904_NoAug_NoResize_coco
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from karyoflow.constants import NUM_SLOTS
from karyoflow.dataset import KaryoFlowDataset
from karyoflow.encoder import ChromosomeEncoder
from karyoflow.flow_module import KaryoFlowModule
from karyoflow.inference import karyoflow_inference
from karyoflow.loss import KaryoFlowLoss


def parse_args():
    parser = argparse.ArgumentParser(description="KaryoFlow Training")
    parser.add_argument("--data-root", type=str,
                        default="/data/linkst/datasets/Chromosome20240904_NoAug_NoResize_coco")
    parser.add_argument("--epochs",       type=int,   default=300)
    parser.add_argument("--batch-size",   type=int,   default=128,   help="Per-GPU batch size (default 128, ~6.5GB on A5000)")
    parser.add_argument("--lr",           type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--d-model",      type=int,   default=256)
    parser.add_argument("--num-layers",   type=int,   default=6)
    parser.add_argument("--nhead",        type=int,   default=8)
    parser.add_argument("--crop-size",    type=int,   default=64)
    parser.add_argument("--num-workers",  type=int,   default=-1,    help="-1 = auto (min(os.cpu_count()//2, 8))")
    parser.add_argument("--eval-steps",   type=int,   default=10,    help="Evaluate every N epochs")
    parser.add_argument("--eval-inf-steps", type=int, default=10,    help="Unmask steps during evaluation inference")
    parser.add_argument("--save-dir",     type=str,   default="/data/linkst/chromosome-kd/work_dirs/karyoflow_v1")
    parser.add_argument("--device",       type=str,   default="cuda")
    parser.add_argument("--amp",          type=int,   default=1,     help="1=use AMP (default), 0=disable")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--overfit-test", action="store_true", help="Overfit on 10 samples to verify pipeline")
    return parser.parse_args()


def build_model(args):
    encoder = ChromosomeEncoder(
        d_model=args.d_model, crop_size=args.crop_size,
        use_geometry=True, pretrained=True,
    )
    flow_module = KaryoFlowModule(
        d_model=args.d_model, num_layers=args.num_layers,
        nhead=args.nhead, dim_feedforward=args.d_model * 4,
        num_slots=NUM_SLOTS,
    )
    return encoder, flow_module


def train_one_epoch(
    encoder, flow_module, criterion, dataloader,
    optimizer, scaler, scheduler, device, epoch, total_epochs, use_amp,
):
    encoder.train()
    flow_module.train()

    total_loss, total_correct, total_masked, num_batches = 0., 0, 0, 0
    all_params = list(encoder.parameters()) + list(flow_module.parameters())

    pbar = tqdm(dataloader, desc=f"Train {epoch}/{total_epochs}",
                leave=False, dynamic_ncols=True)

    for batch in pbar:
        # non_blocking: CPU→GPU 与下一次 CPU 数据准备并行
        crops   = batch["crops"].to(device, non_blocking=True)
        bboxes  = batch["bboxes"].to(device, non_blocking=True)
        perm_gt = batch["permutation"].to(device, non_blocking=True)
        B, N    = perm_gt.shape

        with autocast(enabled=use_amp):
            # 编码
            chrom_features = encoder(
                crops.view(B * N, *crops.shape[2:]),
                bboxes.view(B * N, 4),
            ).view(B, N, -1)

            # 随机 mask
            t = torch.clamp(torch.rand(B, device=device), min=1.0 / N)
            mask = torch.rand(B, N, device=device) < t.unsqueeze(1)
            perm_noisy = perm_gt.clone()
            perm_noisy[mask] = flow_module.mask_token_id

            logits = flow_module(chrom_features, perm_noisy, t)
            loss   = criterion(logits, perm_gt, mask)

        optimizer.zero_grad(set_to_none=True)   # set_to_none 比置 0 更快
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # 统计 (detach 避免保留计算图)
        total_loss += loss.detach().item()
        with torch.no_grad():
            correct = (logits.argmax(-1)[mask] == perm_gt[mask]).sum().item()
            total_correct += correct
            total_masked  += mask.sum().item()
        num_batches += 1

        acc = total_correct / max(total_masked, 1) * 100
        pbar.set_postfix(
            loss=f"{total_loss / num_batches:.4f}",
            acc=f"{acc:.1f}%",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )

    scheduler.step()
    return total_loss / max(num_batches, 1), total_correct / max(total_masked, 1) * 100


@torch.no_grad()
def evaluate(encoder, flow_module, dataloader, device, num_steps=10):
    encoder.eval()
    flow_module.eval()

    total_pos_correct, total_positions = 0, 0
    total_perm_correct, total_samples  = 0, 0

    pbar = tqdm(dataloader, desc="  Eval ", leave=False, dynamic_ncols=True)

    for batch in pbar:
        crops   = batch["crops"].to(device, non_blocking=True)
        bboxes  = batch["bboxes"].to(device, non_blocking=True)
        perm_gt = batch["permutation"].to(device, non_blocking=True)
        B, N    = perm_gt.shape

        chrom_features = encoder(
            crops.view(B * N, *crops.shape[2:]),
            bboxes.view(B * N, 4),
        ).view(B, N, -1)

        perm_pred = karyoflow_inference(flow_module, chrom_features,
                                        num_steps=num_steps)["permutation"]

        total_pos_correct  += (perm_pred == perm_gt).sum().item()
        total_positions    += B * N
        total_perm_correct += (perm_pred == perm_gt).all(dim=1).sum().item()
        total_samples      += B

        pbar.set_postfix(
            pos_acc=f"{total_pos_correct / max(total_positions, 1) * 100:.1f}%"
        )

    return {
        "pos_acc":  total_pos_correct  / max(total_positions, 1) * 100,
        "perm_acc": total_perm_correct / max(total_samples,   1) * 100,
    }


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True   # 固定输入尺寸时加速 conv

    device  = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp) and device.type == "cuda"

    # num_workers 自动设置
    if args.num_workers < 0:
        args.num_workers = min(os.cpu_count() // 2, 8)

    print(f"Device: {device}  |  AMP: {use_amp}  |  workers: {args.num_workers}")

    # ── 数据集 ──────────────────────────────────────────────────
    data_root  = args.data_root
    train_coco = os.path.join(data_root, "train", "_annotations.coco.json")
    valid_coco = os.path.join(data_root, "valid", "_annotations.coco.json")
    train_kf   = os.path.join(data_root, "train_karyoflow.json")
    valid_kf   = os.path.join(data_root, "valid_karyoflow.json")

    if not os.path.exists(train_kf):
        print(f"[ERROR] KaryoFlow annotations not found: {train_kf}")
        print("Run first: python tools/generate_annotations.py --data-root " + data_root)
        return

    def _image_dir(split):
        d = os.path.join(data_root, split)
        return os.path.join(d, "images") if os.path.isdir(os.path.join(d, "images")) else d

    print("Loading datasets...")
    train_dataset = KaryoFlowDataset(
        coco_json=train_coco, karyoflow_json=train_kf,
        image_dir=_image_dir("train"), crop_size=args.crop_size, augment=True,
    )
    valid_dataset = KaryoFlowDataset(
        coco_json=valid_coco, karyoflow_json=valid_kf,
        image_dir=_image_dir("valid"), crop_size=args.crop_size, augment=False,
    )
    print(f"  Train: {len(train_dataset)}  |  Valid: {len(valid_dataset)}")

    if args.overfit_test:
        train_dataset.valid_image_ids = train_dataset.valid_image_ids[:10]
        valid_dataset = train_dataset
        print(f"  Overfit mode: {len(train_dataset)} samples")

    loader_kw = dict(num_workers=args.num_workers, pin_memory=True,
                     prefetch_factor=2 if args.num_workers > 0 else None,
                     persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, **loader_kw)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                              shuffle=False, **loader_kw)

    # ── 模型 ────────────────────────────────────────────────────
    encoder, flow_module = build_model(args)
    encoder     = encoder.to(device)
    flow_module = flow_module.to(device)
    criterion   = KaryoFlowLoss(label_smoothing=0.0)

    enc_p  = sum(p.numel() for p in encoder.parameters())
    flow_p = sum(p.numel() for p in flow_module.parameters())
    print(f"Encoder: {enc_p/1e6:.1f}M  |  FlowModule: {flow_p/1e6:.1f}M  "
          f"|  Total: {(enc_p+flow_p)/1e6:.1f}M")
    print(f"Train batches/epoch: {len(train_loader)}"
          f"  (bs={args.batch_size}, {len(train_dataset)} samples)")

    # ── 优化器 & 调度器 ─────────────────────────────────────────
    all_params = list(encoder.parameters()) + list(flow_module.parameters())
    optimizer  = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = GradScaler(enabled=use_amp)

    os.makedirs(args.save_dir, exist_ok=True)

    # ── 训练循环 ─────────────────────────────────────────────────
    best_pos_acc = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            encoder, flow_module, criterion, train_loader,
            optimizer, scaler, scheduler, device, epoch, args.epochs, use_amp,
        )
        dt = time.time() - t0

        log = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
               "time": dt, "lr": optimizer.param_groups[0]["lr"]}

        print(f"Epoch {epoch}/{args.epochs}: "
              f"loss={train_loss:.4f}  acc={train_acc:.1f}%  "
              f"lr={log['lr']:.2e}  {dt:.0f}s")

        # 评估
        if epoch % args.eval_steps == 0 or epoch == 1:
            eval_result = evaluate(encoder, flow_module, valid_loader,
                                   device, num_steps=args.eval_inf_steps)
            log.update(eval_result)
            print(f"  → Eval: pos_acc={eval_result['pos_acc']:.1f}%  "
                  f"perm_acc={eval_result['perm_acc']:.1f}%")

            if eval_result["pos_acc"] > best_pos_acc:
                best_pos_acc = eval_result["pos_acc"]
                torch.save({
                    "epoch": epoch,
                    "encoder_state_dict":    encoder.state_dict(),
                    "flow_module_state_dict": flow_module.state_dict(),
                    "optimizer_state_dict":  optimizer.state_dict(),
                    "pos_acc": best_pos_acc,
                    "args":    vars(args),
                }, os.path.join(args.save_dir, "best.pth"))
                print(f"  → Saved best (pos_acc={best_pos_acc:.1f}%)")

        history.append(log)
        with open(os.path.join(args.save_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    print(f"\nDone. Best pos_acc: {best_pos_acc:.1f}%")
    print(f"Checkpoint: {args.save_dir}/best.pth")


if __name__ == "__main__":
    main()
