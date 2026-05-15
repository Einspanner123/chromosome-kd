import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from mmengine.dataset import Compose

from mmdet.apis import init_detector
from mmdet.utils import register_all_modules


def calculate_straightness(trajectory):
    """
    计算轨迹的直线度。
    直线度 = ||x_T - x_0|| / sum(||x_{t} - x_{t-1}||)
    值越接近 1.0，说明轨迹越直。
    """
    # trajectory: list of (logits, bboxes)
    # 我们只取 bboxes: [1, num_proposals, 4]
    all_bboxes = [step[1].cpu() for step in trajectory]
    all_bboxes = torch.stack(all_bboxes)  # [num_steps, 1, num_proposals, 4]
    all_bboxes = all_bboxes.squeeze(1)  # [num_steps, num_proposals, 4]

    num_steps = all_bboxes.shape[0]
    num_proposals = all_bboxes.shape[1]

    # 计算起点到终点的位移长度
    start_point = all_bboxes[0]
    end_point = all_bboxes[-1]
    displacement = torch.norm(
        end_point - start_point, dim=-1)  # [num_proposals]

    # 计算实际路径长度
    path_length = torch.zeros(num_proposals)
    for i in range(1, num_steps):
        step_dist = torch.norm(all_bboxes[i] - all_bboxes[i - 1], dim=-1)
        path_length += step_dist

    straightness = displacement / torch.clamp(path_length, min=1e-6)
    return straightness.numpy()


def analyze_straightness(config_path, checkpoint_path, img_path, out_dir):
    register_all_modules()
    model = init_detector(config_path, checkpoint_path, device='cuda:0')

    # 准备数据
    test_pipeline = model.cfg.test_dataloader.dataset.pipeline
    preprocess = Compose(test_pipeline)
    data_info = dict(img_path=img_path, img_id=0)
    data = preprocess(data_info)

    data_for_preprocessor = dict(
        inputs=[data['inputs'].to('cuda:0')],
        data_samples=[data['data_samples']])

    with torch.no_grad():
        preprocessed_data = model.data_preprocessor(
            data_for_preprocessor, training=False)
        batch_inputs = preprocessed_data['inputs']
        data_samples = preprocessed_data['data_samples']

        # 获取轨迹
        results = model.predict(
            batch_inputs, data_samples, return_trajectory=True)
        trajectory = results[0].metainfo['sampling_trajectory']

    # 计算直线度
    straightness_scores = calculate_straightness(trajectory)

    # 绘图
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.figure(figsize=(10, 6))
    plt.hist(
        straightness_scores,
        bins=50,
        alpha=0.75,
        color='blue',
        edgecolor='black')
    plt.title(f'Straightness Distribution ({model.bbox_head.diffusion_type})')
    plt.xlabel('Straightness Score (1.0 is perfectly straight)')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.3)

    save_path = os.path.join(out_dir, 'straightness_dist.png')
    plt.savefig(save_path)
    print(f'Straightness distribution plot saved to {save_path}')
    print(f'Mean Straightness: {np.mean(straightness_scores):.4f}')
    print(f'Median Straightness: {np.median(straightness_scores):.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Config file path')
    parser.add_argument('checkpoint', help='Checkpoint file path')
    parser.add_argument('img', help='Image file path')
    parser.add_argument(
        '--out-dir', default='straightness_analysis', help='Output directory')
    args = parser.parse_args()

    analyze_straightness(args.config, args.checkpoint, args.img, args.out_dir)
