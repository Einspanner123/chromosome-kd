"""
推理可视化脚本: 用训练好的 LDMDetDiT 模型对验证集图像做检测并画框
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mmdet.apis import init_detector, inference_detector
from mmengine.config import Config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='projects/LDMDetDiT/configs/ldmdet_dit.py')
    parser.add_argument('--checkpoint', type=str,
                        default='work_dirs/ldmdet_dit_v7/epoch_3.pth')
    parser.add_argument('--image-dir', type=str,
                        default='data/Chromosome20240904_NoAug_NoResize_coco/valid')
    parser.add_argument('--output-dir', type=str,
                        default='work_dirs/ldmdet_dit_v7/inference_vis')
    parser.add_argument('--num-images', type=int, default=8)
    parser.add_argument('--score-thr', type=float, default=0.01)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 加载模型
    print(f'Loading model from {args.checkpoint}...')
    model = init_detector(args.config, args.checkpoint, device='cuda:0')
    model.eval()
    print('Model loaded.')

    # 类别名称
    class_names = [
        'A1', 'A2', 'A3', 'B4', 'B5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12',
        'D13', 'D14', 'D15', 'E16', 'E17', 'E18', 'F19', 'F20', 'G21', 'G22', 'X', 'Y'
    ]
    cmap = plt.cm.get_cmap('tab20', len(class_names))

    # 获取图像
    img_dir = Path(args.image_dir)
    image_files = sorted(list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')))
    image_files = image_files[:args.num_images]

    for img_path in image_files:
        print(f'\nProcessing: {img_path.name}')

        # 推理
        result = inference_detector(model, str(img_path))

        # 提取检测结果
        pred = result.pred_instances
        bboxes = pred.bboxes.cpu().numpy()   # (N, 4) xyxy
        scores = pred.scores.cpu().numpy()    # (N,)
        labels = pred.labels.cpu().numpy()    # (N,)

        # 过滤低分
        mask = scores >= args.score_thr
        bboxes = bboxes[mask]
        scores = scores[mask]
        labels = labels[mask]

        print(f'  Detected {len(bboxes)} boxes (score >= {args.score_thr})')
        if len(bboxes) > 0:
            print(f'  Score range: [{scores.min():.4f}, {scores.max():.4f}]')
            thresholds = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
            dist = ', '.join([f'>{t}: {(scores > t).sum()}' for t in thresholds])
            print(f'  Score distribution: {dist}')

            # 框尺寸统计
            widths = bboxes[:, 2] - bboxes[:, 0]
            heights = bboxes[:, 3] - bboxes[:, 1]
            areas = widths * heights
            print(f'  Box size: w=[{widths.min():.0f}, {widths.max():.0f}], '
                  f'h=[{heights.min():.0f}, {heights.max():.0f}], '
                  f'area=[{areas.min():.0f}, {areas.max():.0f}]')

        # 可视化
        from PIL import Image
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img)

        fig, ax = plt.subplots(1, 1, figsize=(14, 14))
        ax.imshow(img_np)

        for i in range(len(bboxes)):
            x1, y1, x2, y2 = bboxes[i]
            score = scores[i]
            label = labels[i]
            color = cmap(label)

            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 fill=False, edgecolor=color, linewidth=1.5)
            ax.add_patch(rect)
            ax.text(x1, y1 - 2, f'{class_names[label]}:{score:.2f}',
                    fontsize=7, color=color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.7))

        ax.set_title(f'{img_path.name}: {len(bboxes)} detections', fontsize=12)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f'{img_path.stem}_det.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    print(f'\nAll results saved to: {args.output_dir}')


if __name__ == '__main__':
    main()
