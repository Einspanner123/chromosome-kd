import argparse
import os

import cv2
import torch
from mmengine.dataset import Compose

from mmdet.apis import init_detector
from mmdet.utils import register_all_modules


def visualize_trajectory(config_path, checkpoint_path, img_path, out_dir):
    register_all_modules()

    # 1. 初始化模型
    model = init_detector(config_path, checkpoint_path, device='cuda:0')
    cfg = model.cfg

    # 2. 准备图像数据
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # 获取预处理 pipeline
    test_pipeline = cfg.test_dataloader.dataset.pipeline
    preprocess = Compose(test_pipeline)

    data_info = dict(img_path=img_path, img_id=0)
    data = preprocess(data_info)

    # 3. 运行推理并获取轨迹
    with torch.no_grad():
        # 使用模型的 data_preprocessor 进行预处理（归一化、填充等）
        # data_preprocessor 期望一个 dict，包含 'inputs' 和 'data_samples'
        # inputs 应该是一个 list of tensor
        data_for_preprocessor = dict(
            inputs=[data['inputs'].to('cuda:0')],
            data_samples=[data['data_samples']])
        preprocessed_data = model.data_preprocessor(
            data_for_preprocessor, training=False)
        batch_inputs = preprocessed_data['inputs']
        data_samples = preprocessed_data['data_samples']

        # 调用我们新增的 return_trajectory 功能
        results = model.predict(
            batch_inputs, data_samples, return_trajectory=True)
        data_sample = results[0]
        trajectory = data_sample.metainfo['sampling_trajectory']

    # 4. 可视化每一帧
    import numpy as np

    img = cv2.imread(img_path)
    h_ori, w_ori = img.shape[:2]

    # 获取缩放因子以映射回原图
    # data_sample.metainfo['scale_factor'] 通常是 [w_scale, h_scale]
    scale_factor = data_sample.metainfo.get('scale_factor', [1.0, 1.0])
    if isinstance(scale_factor, (list, tuple, np.ndarray, torch.Tensor)):
        w_scale, h_scale = scale_factor[:2]
    else:
        w_scale = h_scale = scale_factor

    for i, (logits, bboxes) in enumerate(trajectory):
        # bboxes 是模型输出的 xyxy (在预处理后的图像空间)
        # logits: [1, num_proposals, num_classes]
        # bboxes: [1, num_proposals, 4]

        scores = torch.sigmoid(logits).max(-1)[0][0]

        frame = img.copy()
        # 降低阈值，以便在早期步骤也能看到一些框
        keep = scores > 0.1
        step_bboxes = bboxes[0][keep].cpu().numpy()
        step_scores = scores[keep].cpu().numpy()

        for bbox, score in zip(step_bboxes, step_scores):
            x1, y1, x2, y2 = bbox
            # 映射回原图尺寸: 预处理坐标 / 缩放因子
            x1, x2 = x1 / w_scale, x2 / w_scale
            y1, y2 = y1 / h_scale, y2 / h_scale

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # 确保在图像范围内
            x1, x2 = max(0, min(x1, w_ori)), max(0, min(x2, w_ori))
            y1, y2 = max(0, min(y1, h_ori)), max(0, min(y2, h_ori))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f'{score:.2f}',
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        out_path = os.path.join(out_dir, f'step_{i:03d}.png')
        cv2.imwrite(out_path, frame)
        print(f'Saved trajectory step {i} to {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Config file path')
    parser.add_argument('checkpoint', help='Checkpoint file path')
    parser.add_argument('img', help='Image file path')
    parser.add_argument(
        '--out-dir', default='trajectory_vis', help='Output directory')
    args = parser.parse_args()

    visualize_trajectory(args.config, args.checkpoint, args.img, args.out_dir)
