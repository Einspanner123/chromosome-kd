"""
Reflow 训练脚本 — LDMDet

两阶段流程:
  Stage 1: 用训练好的模型多步 ODE 生成 (noise, detection) 配对
  Stage 2: 用配对数据重新训练模型，拉直 ODE 路径

用法:
  # Stage 1: 生成配对
  python projects/LDMDet/tools/reflow_train.py generate \
    --config projects/LDMDet/configs/ldmdet_flowdet_adaln_trd_full.py \
    --checkpoint work_dirs/ldmdet_flowdet_adaln_trd_full/best_coco_bbox_mAP_epoch_63.pth \
    --output-dir work_dirs/reflow_pairs/trd_full \
    --num-ode-steps 10

  # Stage 2: Reflow 训练
  python projects/LDMDet/tools/reflow_train.py train \
    --config projects/LDMDet/configs/ldmdet_flowdet_adaln_reflow.py \
    --reflow-pairs work_dirs/reflow_pairs/trd_full \
    --work-dir work_dirs/ldmdet_flowdet_adaln_reflow
"""

import argparse
import os
import sys

import torch
from mmengine.config import Config
from mmengine.runner import Runner

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))


def parse_args():
    parser = argparse.ArgumentParser(description="LDMDet Reflow Training")
    subparsers = parser.add_subparsers(dest="stage", required=True)

    gen_parser = subparsers.add_parser("generate", help="Stage 1: Generate pairs")
    gen_parser.add_argument("--config", required=True, help="Config file")
    gen_parser.add_argument("--checkpoint", required=True, help="Model checkpoint")
    gen_parser.add_argument("--output-dir", required=True, help="Output dir for pairs")
    gen_parser.add_argument("--num-ode-steps", type=int, default=10)
    gen_parser.add_argument("--device", type=str, default="cuda:0")
    gen_parser.add_argument("--use-train-set", action="store_true", help="Use train set instead of val set")

    train_parser = subparsers.add_parser("train", help="Stage 2: Reflow train")
    train_parser.add_argument("--config", required=True, help="Config file")
    train_parser.add_argument("--reflow-pairs", required=True, help="Reflow pairs dir")
    train_parser.add_argument("--work-dir", required=True, help="Work dir")
    train_parser.add_argument("--resume", action="store_true", help="Resume training")

    return parser.parse_args()


@torch.no_grad()
def generate_pairs(args):
    import projects.LDMDet.model
    import projects.LDMDet.hooks
    from mmdet.apis import init_detector
    from mmdet.utils import register_all_modules

    from projects.LDMDet.mods.structures import ImageMeta

    register_all_modules()
    model = init_detector(args.config, args.checkpoint, device=args.device)
    model.eval()

    cfg = model.cfg
    use_train = args.use_train_set
    if use_train:
        dataloader = Runner.build_dataloader(cfg.train_dataloader)
        print(f"Generating pairs from TRAIN set")
    else:
        dataloader = Runner.build_dataloader(cfg.val_dataloader)
        print(f"Generating pairs from VAL set")

    os.makedirs(args.output_dir, exist_ok=True)
    bbox_head = model.bbox_head
    device = torch.device(args.device)

    pair_count = 0
    for batch_idx, data_batch in enumerate(dataloader):
        data_preprocessed = model.data_preprocessor(data_batch, training=False)
        batch_inputs = data_preprocessed["inputs"]
        data_samples = data_preprocessed["data_samples"]

        img_metas = []
        for ds in data_samples:
            meta = ImageMeta(
                img_shape=ds.metainfo["img_shape"],
                ori_shape=ds.metainfo.get("ori_shape"),
                scale_factor=ds.metainfo.get("scale_factor"),
            )
            img_metas.append(meta)

        features = model.extract_feat(batch_inputs)

        bs = len(img_metas)
        z = bbox_head._sample_noise(bs, device)

        b = z.clone()
        times = torch.linspace(1.0, 0.0, args.num_ode_steps + 1, device=device)

        for i in range(args.num_ode_steps):
            t_curr = times[i].item()
            t_next = times[i + 1].item()
            _, _, x0_raw, _ = bbox_head._forward_at_t(features, b, t_curr, img_metas)
            b = bbox_head.rf.step(b, x0_raw, t_curr, t_next)

        save_path = os.path.join(args.output_dir, f"reflow_pairs_{pair_count:06d}.pt")
        torch.save({"z": z.cpu(), "b_pred": b.cpu()}, save_path)
        pair_count += 1

        if pair_count % 50 == 0:
            print(f"Generated pairs for {pair_count} batches")

    print(f"All {pair_count} pair files saved to {args.output_dir}")


def train_reflow(args):
    cfg = Config.fromfile(args.config)
    cfg.work_dir = args.work_dir

    cfg.model.bbox_head.reflow_pairs_dir = args.reflow_pairs

    if args.resume:
        cfg.resume = True

    runner = Runner.from_cfg(cfg)
    runner.train()


def main():
    args = parse_args()

    if args.stage == "generate":
        generate_pairs(args)
    elif args.stage == "train":
        train_reflow(args)


if __name__ == "__main__":
    main()
