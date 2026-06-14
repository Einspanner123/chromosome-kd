"""ChromoGen统一评估入口

整合所有评估指标，输出完整评估报告。

用法:
  python projects/ChromoGen/tools/evaluate.py \
    --real_ann data/Chromosome20240904_NoAug_NoResize_coco/train/_annotations.coco.json \
    --real_img_dir data/Chromosome20240904_NoAug_NoResize_coco/train/ \
    --gen_dir generated/ \
    --gen_info generated/generation_info.json \
    --output_dir eval_results/

评估指标:
  图像质量:
    - FID (Fréchet Inception Distance)
    - IS (Inception Score)
    - LPIPS (感知相似度)
    - Diversity-LPIPS (生成多样性)

  BBox质量:
    - 类别分布匹配 (KL/JS散度)
    - 尺寸分布匹配 (Wasserstein距离)
    - 数量分布匹配
    - 空间分布匹配
    - 置信度分布

  下游任务 (可选):
    - LDMDet mAP对比
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from metrics.bbox_metrics import (
    compute_class_distribution,
    compute_count_distribution,
    compute_iou_quality,
    compute_size_distribution,
    compute_spatial_distribution,
)
from metrics.image_metrics import (
    compute_diversity_lpips,
    compute_fid,
    compute_is,
    compute_lpips,
)

# SwanLab集成
try:
    import swanlab

    HAS_SWANLAB = True
except ImportError:
    HAS_SWANLAB = False


def parse_args():
    parser = argparse.ArgumentParser(description='ChromoGen Evaluation')

    # 数据路径
    parser.add_argument(
        '--real_ann', type=str, required=True, help='真实COCO标注文件路径'
    )
    parser.add_argument(
        '--real_img_dir', type=str, required=True, help='真实图像目录'
    )
    parser.add_argument(
        '--gen_dir', type=str, required=True, help='生成图像目录'
    )
    parser.add_argument(
        '--gen_info', type=str, required=True, help='生成结果信息JSON路径'
    )

    # 评估选项
    parser.add_argument(
        '--eval_image',
        action='store_true',
        default=True,
        help='评估图像质量 (FID, IS, LPIPS)',
    )
    parser.add_argument(
        '--eval_bbox', action='store_true', default=True, help='评估BBox质量'
    )
    parser.add_argument(
        '--eval_downstream',
        action='store_true',
        default=False,
        help='评估下游任务 (需要LDMDet checkpoint)',
    )

    # 下游任务参数
    parser.add_argument('--ldmdet_config', type=str, default=None)
    parser.add_argument('--ldmdet_ckpt', type=str, default=None)

    # 通用参数
    parser.add_argument(
        '--score_threshold', type=float, default=0.3, help='BBox置信度阈值'
    )
    parser.add_argument(
        '--max_images', type=int, default=5000, help='最大评估图像数'
    )
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='eval_results/')
    parser.add_argument(
        '--swanlab_project',
        type=str,
        default='chromosome-kd',
        help='SwanLab项目名',
    )
    parser.add_argument(
        '--swanlab_experiment', type=str, default=None, help='SwanLab实验名'
    )
    parser.add_argument(
        '--no_swanlab', action='store_true', help='禁用SwanLab日志'
    )

    return parser.parse_args()


def evaluate_image_quality(args, use_swanlab: bool = False) -> dict:
    """评估图像质量"""
    import torch

    device = torch.device(
        f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    )

    results = {}
    print('\n' + '=' * 60)
    print('图像质量评估')
    print('=' * 60)

    # FID
    print('[1/4] 计算FID...')
    t0 = time.time()
    fid = compute_fid(
        real_dir=args.real_img_dir,
        gen_dir=args.gen_dir,
        batch_size=args.batch_size,
        max_images=args.max_images,
        device=device,
    )
    results['FID'] = fid
    print(f'  FID: {fid:.4f} ({time.time() - t0:.1f}s)')

    # IS
    print('[2/4] 计算IS...')
    t0 = time.time()
    is_mean, is_std = compute_is(
        gen_dir=args.gen_dir,
        batch_size=args.batch_size,
        max_images=args.max_images,
        device=device,
    )
    results['IS_mean'] = is_mean
    results['IS_std'] = is_std
    print(f'  IS: {is_mean:.4f} ± {is_std:.4f} ({time.time() - t0:.1f}s)')

    # LPIPS
    print('[3/4] 计算LPIPS...')
    t0 = time.time()
    lpips_val = compute_lpips(
        real_dir=args.real_img_dir,
        gen_dir=args.gen_dir,
        batch_size=args.batch_size,
        max_pairs=min(1000, args.max_images),
        device=device,
    )
    results['LPIPS'] = lpips_val
    print(f'  LPIPS: {lpips_val:.4f} ({time.time() - t0:.1f}s)')

    # Diversity-LPIPS
    print('[4/4] 计算生成多样性 (LPIPS)...')
    t0 = time.time()
    diversity = compute_diversity_lpips(
        gen_dir=args.gen_dir,
        batch_size=args.batch_size,
        max_pairs=500,
        device=device,
    )
    results['Diversity_LPIPS'] = diversity
    print(f'  Diversity-LPIPS: {diversity:.4f} ({time.time() - t0:.1f}s)')

    # SwanLab日志
    if use_swanlab:
        swanlab.log(
            {
                'eval/FID': fid,
                'eval/IS_mean': is_mean,
                'eval/IS_std': is_std,
                'eval/LPIPS': lpips_val,
                'eval/Diversity_LPIPS': diversity,
            }
        )

    return results


def evaluate_bbox_quality(args, use_swanlab: bool = False) -> dict:
    """评估BBox质量"""
    results = {}
    print('\n' + '=' * 60)
    print('BBox质量评估')
    print('=' * 60)

    # 类别分布
    print('[1/5] 计算类别分布匹配...')
    class_dist = compute_class_distribution(
        real_ann_path=args.real_ann,
        gen_info_path=args.gen_info,
        score_threshold=args.score_threshold,
    )
    results['class_distribution'] = class_dist
    print(f'  KL散度: {class_dist["kl_divergence"]:.4f}')
    print(f'  JS散度: {class_dist["js_divergence"]:.4f}')

    # 尺寸分布
    print('[2/5] 计算尺寸分布匹配...')
    size_dist = compute_size_distribution(
        real_ann_path=args.real_ann,
        gen_info_path=args.gen_info,
        score_threshold=args.score_threshold,
    )
    results['size_distribution'] = size_dist
    print(f'  Wasserstein-Width: {size_dist["wasserstein_width"]:.4f}')
    print(f'  Wasserstein-Height: {size_dist["wasserstein_height"]:.4f}')
    print(f'  尺寸分类匹配度: {size_dist["size_category_match"]:.4f}')

    # 数量分布
    print('[3/5] 计算数量分布匹配...')
    count_dist = compute_count_distribution(
        real_ann_path=args.real_ann,
        gen_info_path=args.gen_info,
        score_threshold=args.score_threshold,
    )
    results['count_distribution'] = count_dist
    print(f'  真实平均目标数: {count_dist["mean_count_real"]:.1f}')
    print(f'  生成平均目标数: {count_dist["mean_count_gen"]:.1f}')
    print(f'  Wasserstein距离: {count_dist["wasserstein_count"]:.4f}')

    # IoU/置信度质量
    print('[4/5] 计算BBox置信度分布...')
    iou_quality = compute_iou_quality(
        gen_info_path=args.gen_info,
        score_threshold=args.score_threshold,
    )
    results['bbox_quality'] = iou_quality
    print(f'  平均置信度: {iou_quality["mean_score"]:.4f}')
    print(f'  高置信度比例(>=0.7): {iou_quality["high_conf_ratio"]:.4f}')

    # 空间分布
    print('[5/5] 计算空间分布匹配...')
    spatial_dist = compute_spatial_distribution(
        real_ann_path=args.real_ann,
        gen_info_path=args.gen_info,
        score_threshold=args.score_threshold,
    )
    results['spatial_distribution'] = spatial_dist
    print(f'  空间KL散度: {spatial_dist["spatial_kl_divergence"]:.4f}')
    print(f'  空间JS散度: {spatial_dist["spatial_js_divergence"]:.4f}')

    # SwanLab日志
    if use_swanlab:
        swanlab.log(
            {
                'eval/class_kl': class_dist['kl_divergence'],
                'eval/class_js': class_dist['js_divergence'],
                'eval/size_wasserstein_w': size_dist['wasserstein_width'],
                'eval/size_wasserstein_h': size_dist['wasserstein_height'],
                'eval/size_category_match': size_dist['size_category_match'],
                'eval/count_wasserstein': count_dist['wasserstein_count'],
                'eval/count_mean_real': count_dist['mean_count_real'],
                'eval/count_mean_gen': count_dist['mean_count_gen'],
                'eval/spatial_kl': spatial_dist['spatial_kl_divergence'],
                'eval/spatial_js': spatial_dist['spatial_js_divergence'],
                'eval/bbox_mean_score': iou_quality['mean_score'],
                'eval/bbox_high_conf_ratio': iou_quality['high_conf_ratio'],
            }
        )

    return results


def evaluate_downstream(args) -> dict:
    """评估下游任务"""
    if not args.ldmdet_config or not args.ldmdet_ckpt:
        print(
            'Warning: LDMDet config/checkpoint not provided, skipping downstream evaluation'
        )
        return {}

    from metrics.downstream_metrics import evaluate_ldmdet

    print('\n' + '=' * 60)
    print('下游任务评估 (LDMDet)')
    print('=' * 60)

    metrics = evaluate_ldmdet(
        config_path=args.ldmdet_config,
        work_dir=os.path.join(args.output_dir, 'ldmdet_eval'),
        checkpoint_path=args.ldmdet_ckpt,
        data_root=os.path.dirname(args.real_ann),
        ann_file=args.real_ann,
        gpu_id=args.gpu,
    )
    print(f'  mAP: {metrics.get("coco/bbox_mAP", "N/A")}')

    return metrics


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # SwanLab初始化
    use_swanlab = HAS_SWANLAB and not args.no_swanlab
    if use_swanlab:
        experiment_name = args.swanlab_experiment or 'ChromoGen-Eval'
        swanlab.init(
            project=args.swanlab_project,
            experiment_name=experiment_name,
            config={
                'real_ann': args.real_ann,
                'gen_dir': args.gen_dir,
                'score_threshold': args.score_threshold,
                'max_images': args.max_images,
            },
        )
    elif not HAS_SWANLAB:
        print('SwanLab not installed, skipping experiment logging')

    all_results = {
        'config': {
            'real_ann': args.real_ann,
            'gen_dir': args.gen_dir,
            'gen_info': args.gen_info,
            'score_threshold': args.score_threshold,
        }
    }

    # 图像质量评估
    if args.eval_image:
        all_results['image_quality'] = evaluate_image_quality(
            args, use_swanlab
        )

    # BBox质量评估
    if args.eval_bbox:
        all_results['bbox_quality'] = evaluate_bbox_quality(args, use_swanlab)

    # 下游任务评估
    if args.eval_downstream:
        all_results['downstream'] = evaluate_downstream(args)

    # 保存结果
    result_path = os.path.join(args.output_dir, 'evaluation_report.json')
    with open(result_path, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 打印摘要
    print('\n' + '=' * 60)
    print('评估摘要')
    print('=' * 60)

    if 'image_quality' in all_results:
        iq = all_results['image_quality']
        print(
            f'  FID:              {iq.get("FID", "N/A"):.4f}'
            if isinstance(iq.get('FID'), float)
            else f'  FID:              {iq.get("FID", "N/A")}'
        )
        print(
            f'  IS:               {iq.get("IS_mean", "N/A"):.4f} ± {iq.get("IS_std", 0):.4f}'
            if isinstance(iq.get('IS_mean'), float)
            else f'  IS:               {iq.get("IS_mean", "N/A")}'
        )
        print(f'  LPIPS:            {iq.get("LPIPS", "N/A")}')
        print(f'  Diversity-LPIPS:  {iq.get("Diversity_LPIPS", "N/A")}')

    if 'bbox_quality' in all_results:
        bq = all_results['bbox_quality']
        cd = bq.get('class_distribution', {})
        sd = bq.get('size_distribution', {})
        cnt = bq.get('count_distribution', {})
        sp = bq.get('spatial_distribution', {})
        qu = bq.get('bbox_quality', {})
        print(f'  类别JS散度:       {cd.get("js_divergence", "N/A")}')
        print(f'  尺寸匹配度:       {sd.get("size_category_match", "N/A")}')
        print(f'  数量Wasserstein:  {cnt.get("wasserstein_count", "N/A")}')
        print(f'  空间JS散度:       {sp.get("spatial_js_divergence", "N/A")}')
        print(f'  平均置信度:       {qu.get("mean_score", "N/A")}')

    print(f'\n完整报告已保存: {result_path}')

    # 关闭SwanLab
    if use_swanlab:
        swanlab.finish()


if __name__ == '__main__':
    main()
