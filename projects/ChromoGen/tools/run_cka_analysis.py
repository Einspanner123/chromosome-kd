"""Phase 0 探针实验: CKA 分析运行脚本

测量 ChromoGen 特征与 ImageNet 预训练 ResNet 特征的 CKA 相似度。

实验设计:
  1. 对同一批图像, 分别提取 ChromoGen 特征和 ResNet 特征
  2. 对每对层计算 CKA
  3. 分析相似度模式

ChromoGen 层:
  - vae_latent (4ch, H/8)
  - down1 (320ch, H/8)
  - down2 (640ch, H/16)
  - down3 (1280ch, H/32)
  - mid (1280ch, H/64)

ResNet 层 (ImageNet 预训练, 检测特征参考):
  - layer1 (256ch, H/4)
  - layer2 (512ch, H/8)
  - layer3 (1024ch, H/16)
  - layer4 (2048ch, H/32)

用法:
  python projects/ChromoGen/tools/run_cka_analysis.py --gpu 0
"""

import argparse
import json
import os
import sys

import torch
from torchvision import models, transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from projects.ChromoGen.evaluation.cka import CKAAnalyzer
from projects.ChromoGen.evaluation.linear_probe import FeatureExtractor


class ResNetFeatureExtractor:
    """从 ImageNet 预训练 ResNet 提取多尺度特征 (作为检测特征参考)"""

    def __init__(self, device='cuda'):
        self.device = torch.device(device)
        # 加载 ImageNet 预训练 ResNet50
        self.model = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        # 冻结
        for param in self.model.parameters():
            param.requires_grad = False

        self._feature_buffer = {}
        self._register_hooks()

    def _register_hooks(self):
        """注册 hook 收集各层输出"""
        self.model.layer1.register_forward_hook(self._make_hook('res_layer1'))
        self.model.layer2.register_forward_hook(self._make_hook('res_layer2'))
        self.model.layer3.register_forward_hook(self._make_hook('res_layer3'))
        self.model.layer4.register_forward_hook(self._make_hook('res_layer4'))

    def _make_hook(self, name):
        def hook(module, input, output):
            self._feature_buffer[name] = output

        return hook

    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> dict:
        """提取多尺度特征

        Args:
            images: (B, 3, H, W) ImageNet 归一化图像

        Returns:
            feats: dict with 'res_layer1' to 'res_layer4'
        """
        images = images.to(self.device)
        self._feature_buffer.clear()
        _ = self.model(images)
        return {k: v.cpu() for k, v in self._feature_buffer.items()}


def flatten_and_pool(feat: torch.Tensor, output_size: int = 1) -> torch.Tensor:
    """自适应平均池化并展平为 (N, D)

    Args:
        feat: (N, C, H, W)
        output_size: 池化输出尺寸

    Returns:
        pooled: (N, C * output_size^2)
    """
    pooled = torch.nn.functional.adaptive_avg_pool2d(
        feat, output_size=output_size
    )
    return pooled.view(pooled.shape[0], -1)


def main():
    parser = argparse.ArgumentParser(description='CKA Analysis')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='work_dirs/chromogen_phase1/final_model.pt',
    )
    parser.add_argument(
        '--vae_path', type=str, default='work_dirs/chromogen_phase1/vae'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default='data/Chromosome20240904_NoAug_NoResize_coco/',
    )
    parser.add_argument('--num_images', type=int, default=100)
    parser.add_argument(
        '--output',
        type=str,
        default='work_dirs/chromogen_phase1/cka_results.json',
    )
    args = parser.parse_args()

    device = f'cuda:{args.gpu}'
    torch.manual_seed(42)

    # 1. 加载两个特征提取器
    print('Loading ChromoGen model...')
    chromogen_extractor = FeatureExtractor(
        checkpoint=args.checkpoint,
        vae_path=args.vae_path,
        device=device,
    )
    print('Loading ResNet50 (ImageNet pretrained)...')
    resnet_extractor = ResNetFeatureExtractor(device=device)

    # 2. 加载图像
    from PIL import Image
    from pycocotools.coco import COCO

    coco = COCO(os.path.join(args.data_root, 'train/_annotations.coco.json'))
    img_dir = os.path.join(args.data_root, 'train')
    image_ids = list(coco.imgs.keys())[: args.num_images]
    print(f'Using {len(image_ids)} images for CKA analysis')

    # ChromoGen 预处理 (-1 到 1)
    chromogen_transform = transforms.Compose(
        [
            transforms.Resize((768, 768)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    # ResNet 预处理 (ImageNet 归一化)
    resnet_transform = transforms.Compose(
        [
            transforms.Resize((768, 768)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    # 3. 提取特征
    chromogen_feats = {
        'vae_latent': [],
        'down1': [],
        'down2': [],
        'down3': [],
        'mid': [],
    }
    resnet_feats = {
        'res_layer1': [],
        'res_layer2': [],
        'res_layer3': [],
        'res_layer4': [],
    }

    print('Extracting features...')
    for i, img_id in enumerate(image_ids):
        img_info = coco.imgs[img_id]
        img_path = os.path.join(img_dir, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')

        # ChromoGen 特征
        cg_img = chromogen_transform(img).unsqueeze(0)
        cg_feats = chromogen_extractor(cg_img)
        for layer, feat in cg_feats.items():
            # 池化到 1x1 并展平
            chromogen_feats[layer].append(flatten_and_pool(feat, 1).squeeze(0))

        # ResNet 特征
        rn_img = resnet_transform(img).unsqueeze(0)
        rn_feats = resnet_extractor(rn_img)
        for layer, feat in rn_feats.items():
            resnet_feats[layer].append(flatten_and_pool(feat, 1).squeeze(0))

        if (i + 1) % 20 == 0:
            print(f'  Processed {i + 1}/{len(image_ids)} images')

    # 合并特征
    for d in [chromogen_feats, resnet_feats]:
        for k in d:
            d[k] = torch.stack(d[k], dim=0)

    print('\nFeature shapes:')
    for k, v in chromogen_feats.items():
        print(f'  ChromoGen {k}: {v.shape}')
    for k, v in resnet_feats.items():
        print(f'  ResNet {k}: {v.shape}')

    # 4. 计算 CKA 矩阵
    print('\nComputing CKA similarity matrix...')
    analyzer = CKAAnalyzer(device='cpu', n_permutations=200)

    sim_matrix = analyzer.compute_similarity_matrix(
        chromogen_feats, resnet_feats, method='linear'
    )

    cg_layers = list(chromogen_feats.keys())
    rn_layers = list(resnet_feats.keys())

    print(f'\n{"=" * 60}')
    print('CKA Similarity Matrix (Linear)')
    print(f'{"=" * 60}')
    # 表头
    header = f'{"ChromoGen":<15}'
    for rn in rn_layers:
        header += f'{rn:>14}'
    print(header)
    print('-' * (15 + 14 * len(rn_layers)))
    for i, cg in enumerate(cg_layers):
        row = f'{cg:<15}'
        for j in range(len(rn_layers)):
            row += f'{sim_matrix[i, j]:>14.4f}'
        print(row)

    # 5. 找最佳匹配
    print(f'\n{"=" * 60}')
    print('Best Matches (ChromoGen → ResNet)')
    print(f'{"=" * 60}')
    matches = analyzer.find_best_matches(
        chromogen_feats, resnet_feats, method='linear'
    )
    for cg_layer, info in matches.items():
        print(
            f'  {cg_layer:<15} → {info["best_match"]:<15} '
            f'(CKA={info["cka"]:.4f})'
        )

    # 6. 置换检验 (对最佳匹配)
    print(f'\n{"=" * 60}')
    print('Permutation Test (200 permutations)')
    print(f'{"=" * 60}')
    perm_results = {}
    for cg_layer, info in matches.items():
        rn_layer = info['best_match']
        X = chromogen_feats[cg_layer]
        Y = resnet_feats[rn_layer]
        cka, p_value = analyzer.permutation_test(X, Y, method='linear')
        perm_results[cg_layer] = {
            'best_match': rn_layer,
            'cka': cka,
            'p_value': p_value,
        }
        print(
            f'  {cg_layer:<15} vs {rn_layer:<15}: '
            f'CKA={cka:.4f}, p={p_value:.4f}'
        )

    # 7. 保存结果
    results = {
        'similarity_matrix': sim_matrix.tolist(),
        'chromogen_layers': cg_layers,
        'resnet_layers': rn_layers,
        'best_matches': matches,
        'permutation_tests': perm_results,
        'num_images': len(image_ids),
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {args.output}')


if __name__ == '__main__':
    main()
