"""ChromoGen生成入口

用法:
  python projects/ChromoGen/tools/generate.py \
    --checkpoint work_dirs/chromogen/final_model.pt \
    --num_images 100 \
    --output_dir generated/
"""

import argparse
import os
import sys

import torch
from torchvision.utils import save_image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from projects.ChromoGen.models.chromogen_pipeline import ChromoGenPipeline


def parse_args():
    parser = argparse.ArgumentParser(description='ChromoGen Generation')
    parser.add_argument(
        '--checkpoint', type=str, required=True, help='模型checkpoint路径'
    )
    parser.add_argument(
        '--num_images', type=int, default=100, help='生成图像数量'
    )
    parser.add_argument('--batch_size', type=int, default=4, help='批大小')
    parser.add_argument(
        '--num_inference_steps', type=int, default=50, help='DDIM采样步数'
    )
    parser.add_argument(
        '--guidance_scale', type=float, default=7.5, help='CFG缩放因子'
    )
    parser.add_argument(
        '--image_size', type=int, default=768, help='输出图像尺寸'
    )
    parser.add_argument(
        '--output_dir', type=str, default='generated/', help='输出目录'
    )
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument(
        '--save_bboxes', action='store_true', help='是否保存bbox结果'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(
        f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    )

    # 加载checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get('config', {})

    # 构建模型
    model = ChromoGenPipeline(
        vae_model=cfg.get('vae_model', 'stabilityai/sd-vae-ft-mse'),
        sample_size=cfg.get('sample_size', 96),
        unet_block_out_channels=cfg.get(
            'unet_block_out_channels', (320, 640, 1280, 1280)
        ),
        unet_attention_head_dim=cfg.get('unet_attention_head_dim', 8),
        cross_attention_dim=cfg.get('cross_attention_dim', 768),
        gradient_checkpointing=False,
        condition_embed_dim=cfg.get('condition_embed_dim', 768),
        condition_max_count=cfg.get('condition_max_count', 50),
        condition_dropout=0.0,  # 推理时不dropout
        enable_bbox_head=cfg.get('enable_bbox_head', False),
        bbox_feat_channels=cfg.get('bbox_feat_channels', 512),
        bbox_num_proposals=cfg.get('bbox_num_proposals', 100),
        bbox_num_heads=cfg.get('bbox_num_heads', 8),
        bbox_num_layers=cfg.get('bbox_num_layers', 3),
        bbox_snr_scale=cfg.get('bbox_snr_scale', 2.0),
        num_train_timesteps=cfg.get('num_train_timesteps', 1000),
        noise_schedule=cfg.get('noise_schedule', 'linear'),
        prediction_type=cfg.get('prediction_type', 'epsilon'),
        lambda_img=cfg.get('lambda_img', 1.0),
        lambda_bbox=cfg.get('lambda_bbox', 0.5),
        lambda_cls=cfg.get('lambda_cls', 0.5),
        cfg_dropout=0.0,
    )

    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()

    # 输出目录
    img_dir = os.path.join(args.output_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)

    all_results = []

    # 生成
    num_batches = (args.num_images + args.batch_size - 1) // args.batch_size
    img_count = 0

    for batch_idx in range(num_batches):
        B = min(args.batch_size, args.num_images - img_count)

        # 构造条件: 随机采样类别数量
        class_labels = torch.arange(24).unsqueeze(0).expand(B, -1).to(device)
        # 随机生成各类数量 (0-5)
        counts = torch.randint(0, 6, (B, 24), device=device)
        # 确保至少有1个目标
        counts[:, 0] = counts[:, 0].clamp(min=1)

        with torch.no_grad():
            result = model.generate(
                class_labels=class_labels,
                counts=counts,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                image_size=(args.image_size, args.image_size),
                return_bboxes=args.save_bboxes
                and cfg.get('enable_bbox_head', False),
            )

        # 保存图像
        for i in range(B):
            img_path = os.path.join(img_dir, f'gen_{img_count + i:05d}.png')
            save_image(result['images'][i], img_path)

            info = {
                'image_id': img_count + i,
                'image_path': img_path,
                'counts': counts[i].cpu().tolist(),
            }

            if 'pred_bboxes' in result:
                info['bboxes'] = result['pred_bboxes'][i].cpu().tolist()
                info['labels'] = result['pred_labels'][i].cpu().tolist()
                info['scores'] = result['pred_scores'][i].cpu().tolist()

            all_results.append(info)

        img_count += B
        print(f'Generated {img_count}/{args.num_images}')

    # 保存生成信息
    import json

    info_path = os.path.join(args.output_dir, 'generation_info.json')
    with open(info_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f'Done. Generated {img_count} images in {args.output_dir}')


if __name__ == '__main__':
    main()
