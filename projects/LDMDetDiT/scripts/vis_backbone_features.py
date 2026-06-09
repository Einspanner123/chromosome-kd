"""
DINOv3-Small Backbone 特征可视化脚本

仅加载预训练 backbone，对染色体图像提取多尺度特征并可视化：
1. 各层特征图均值投影 (channel mean)
2. PCA 降维可视化 (多通道 -> RGB)
3. Attention Map (CLS token attention)
4. 特征相似度热力图 (cosine similarity between spatial positions)
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import Normalize
from PIL import Image
from sklearn.decomposition import PCA
from timm import create_model
from torchvision import transforms


def load_image(image_path: str, input_size: int = 512) -> tuple:
    """加载并预处理图像，返回 tensor 和原始 PIL 图像"""
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size

    transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    img_tensor = transform(img).unsqueeze(0)  # (1, 3, H, W)
    return img_tensor, img, (orig_h, orig_w)


def extract_features(model: torch.nn.Module, img_tensor: torch.Tensor) -> dict:
    """提取 DINOv3 (Eva 架构) 各 block 的特征

    DINOv3 实际是 Eva 架构:
    - patch_embed 输出 (B, H, W, C) 格式
    - 使用 RoPE (RotaryEmbeddingDinoV3) 而非简单 pos_embed
    - EvaBlock: norm1 -> attn -> drop_path1 + x -> norm2 -> mlp -> drop_path2 + x

    返回:
        features: dict, key 为 block 索引, value 为 (B, C, H, W) 特征图
        attn_weights: dict, key 为 block 索引, value 为 attention 权重
    """
    features = {}
    attn_weights = {}

    # Hook 来捕获各 block 的输出
    hooks = []

    def get_hook(name):
        def hook_fn(module, input, output):
            features[name] = output.detach()

        return hook_fn

    for i, block in enumerate(model.blocks):
        h = block.register_forward_hook(get_hook(f"block_{i}"))
        hooks.append(h)

    with torch.no_grad():
        # 直接用 model 的 forward_features 获取特征
        # 这样 RoPE 等位置编码会被正确处理
        _ = model.forward_features(img_tensor)

    # 移除 hooks
    for h in hooks:
        h.remove()

    # 计算空间尺寸
    patch_size = model.patch_embed.patch_size[0] if hasattr(model.patch_embed.patch_size, '__len__') else model.patch_embed.patch_size
    feat_h = feat_w = img_tensor.shape[-1] // patch_size

    # Eva block 输出格式: (B, N+1+num_registers, C)
    # DINOv3 有 CLS token + register tokens (4个), 所以 N+5
    # patch tokens 从 index 1 到 1+N
    embed_dim = model.embed_dim
    num_patches = feat_h * feat_w
    reshaped_features = {}
    for name, feat in features.items():
        B = feat.shape[0]
        # 去掉 CLS token 和 register tokens
        patch_feat = feat[:, 1 : 1 + num_patches, :]  # (B, H*W, C)
        patch_feat = patch_feat.reshape(B, feat_h, feat_w, embed_dim).permute(0, 3, 1, 2)  # (B, C, H, W)
        reshaped_features[name] = patch_feat

    # 提取 attention 权重 (通过 hook)
    # 在 attn forward 时捕获 qkv 输入, 然后手动计算近似 attention
    attn_weights = {}
    attn_inputs = {}

    def attn_pre_hook(name):
        def fn(module, args):
            attn_inputs[name] = args[0].detach()  # x_norm
            return args
        return fn

    def attn_post_hook(name):
        def fn(module, input, output):
            x_norm = attn_inputs[name]
            B, N_total, C = x_norm.shape
            num_heads = module.num_heads
            head_dim = C // num_heads

            qkv = module.qkv(x_norm)
            qkv = qkv.reshape(B, N_total, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)

            # 近似 attention (不含 RoPE), 用于可视化
            attn_score = q @ k.transpose(-2, -1) / (head_dim ** 0.5)
            attn_prob = attn_score.softmax(dim=-1)
            attn_weights[name] = attn_prob.detach()
        return fn

    pre_hooks = []
    post_hooks = []
    for i, block in enumerate(model.blocks):
        ph = block.attn.register_forward_pre_hook(attn_pre_hook(f"block_{i}"))
        poh = block.attn.register_forward_hook(attn_post_hook(f"block_{i}"))
        pre_hooks.append(ph)
        post_hooks.append(poh)

    with torch.no_grad():
        _ = model.forward_features(img_tensor)

    for h in pre_hooks + post_hooks:
        h.remove()

    return reshaped_features, attn_weights, feat_h, feat_w


def visualize_channel_mean(features: dict, save_path: str, img: Image.Image, feat_h: int, feat_w: int):
    """可视化各层特征图的通道均值投影"""
    n_layers = len(features)
    # 选择 4 个代表性层: 早期、浅层、中层、深层
    indices = [0, n_layers // 3, 2 * n_layers // 3, n_layers - 1]
    names = [f"Block {i}" for i in indices]

    fig, axes = plt.subplots(1, len(indices) + 1, figsize=(4 * (len(indices) + 1), 4))

    # 原图
    axes[0].imshow(img)
    axes[0].set_title("Original", fontsize=12)
    axes[0].axis("off")

    for ax, idx, name in zip(axes[1:], indices, names):
        feat = features[f"block_{idx}"].squeeze(0)  # (C, H, W)
        feat_mean = feat.mean(dim=0).cpu().numpy()  # (H, W)
        feat_mean = (feat_mean - feat_mean.min()) / (feat_mean.max() - feat_mean.min() + 1e-8)
        ax.imshow(feat_mean, cmap="viridis", interpolation="bilinear")
        ax.set_title(name, fontsize=12)
        ax.axis("off")

    plt.suptitle("DINOv3-Small Feature Maps (Channel Mean)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def visualize_pca(features: dict, save_path: str, img: Image.Image, feat_h: int, feat_w: int):
    """PCA 降维可视化: 多通道 -> RGB"""
    indices = [0, len(features) // 2, len(features) - 1]
    names = [f"Block {i}" for i in indices]

    fig, axes = plt.subplots(1, len(indices) + 1, figsize=(4 * (len(indices) + 1), 4))

    axes[0].imshow(img)
    axes[0].set_title("Original", fontsize=12)
    axes[0].axis("off")

    for ax, idx, name in zip(axes[1:], indices, names):
        feat = features[f"block_{idx}"].squeeze(0)  # (C, H, W)
        C, H, W = feat.shape
        feat_np = feat.cpu().numpy().reshape(C, H * W).T  # (H*W, C)

        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(feat_np)  # (H*W, 3)

        # 归一化到 [0, 1]
        pca_result = (pca_result - pca_result.min(axis=0)) / (pca_result.max(axis=0) - pca_result.min(axis=0) + 1e-8)
        rgb = pca_result.reshape(H, W, 3)

        ax.imshow(rgb, interpolation="bilinear")
        ax.set_title(f"{name}\n(explained: {pca.explained_variance_ratio_[:3].sum():.1%})", fontsize=10)
        ax.axis("off")

    plt.suptitle("DINOv3-Small Feature Maps (PCA -> RGB)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def visualize_attention(attn_weights: dict, save_path: str, img: Image.Image, feat_h: int, feat_w: int):
    """可视化 CLS token 对各 patch 的 attention map"""
    indices = [0, len(attn_weights) // 3, 2 * len(attn_weights) // 3, len(attn_weights) - 1]
    names = [f"Block {i}" for i in indices]
    num_patches = feat_h * feat_w

    fig, axes = plt.subplots(1, len(indices) + 1, figsize=(4 * (len(indices) + 1), 4))

    axes[0].imshow(img)
    axes[0].set_title("Original", fontsize=12)
    axes[0].axis("off")

    for ax, idx, name in zip(axes[1:], indices, names):
        attn = attn_weights[f"block_{idx}"]  # (B, num_heads, N_total, N_total)
        # CLS token (index 0) 对各 patch 的 attention
        # patch tokens 在 index 1 到 1+num_patches
        cls_attn = attn[0, :, 0, 1 : 1 + num_patches].mean(dim=0).cpu().numpy()  # (N,)
        cls_attn = cls_attn.reshape(feat_h, feat_w)

        # 上采样到原图大小
        cls_attn_resized = F.interpolate(
            torch.tensor(cls_attn).unsqueeze(0).unsqueeze(0),
            size=(img.size[1], img.size[0]),
            mode="bilinear",
        ).squeeze().numpy()

        ax.imshow(img)
        ax.imshow(cls_attn_resized, cmap="jet", alpha=0.5, interpolation="bilinear")
        ax.set_title(name, fontsize=12)
        ax.axis("off")

    plt.suptitle("DINOv3-Small CLS Attention Map", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def visualize_feature_similarity(features: dict, save_path: str, feat_h: int, feat_w: int):
    """可视化深层特征的空间相似度矩阵 (中心点 vs 全图)"""
    # 使用最后一层
    feat = features[f"block_{len(features) - 1}"].squeeze(0)  # (C, H, W)
    C, H, W = feat.shape

    # L2 归一化
    feat_norm = F.normalize(feat.reshape(C, H * W), dim=0)  # (C, H*W)

    # 计算中心点与全图的 cosine similarity
    center_idx = (H // 2) * W + (W // 2)
    center_feat = feat_norm[:, center_idx]  # (C,)
    sim_map = (feat_norm.T @ center_feat).cpu().numpy().reshape(H, W)

    # 同时画全局相似度矩阵 (采样)
    # 采样若干点计算 pairwise similarity
    n_samples = min(64, H * W)
    sample_indices = np.random.choice(H * W, n_samples, replace=False)
    sample_feats = feat_norm[:, sample_indices].T  # (n_samples, C)
    sim_matrix = (sample_feats @ sample_feats.T).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    im0 = axes[0].imshow(sim_map, cmap="RdYlBu_r", interpolation="bilinear")
    axes[0].set_title("Center Cosine Similarity", fontsize=12)
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(sim_matrix, cmap="RdYlBu_r", vmin=-1, vmax=1)
    axes[1].set_title(f"Pairwise Similarity ({n_samples} samples)", fontsize=12)
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    plt.suptitle(f"DINOv3-Small Feature Similarity (Block {len(features) - 1})", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="DINOv3-Small Backbone Feature Visualization")
    parser.add_argument(
        "--image-dir",
        type=str,
        default="data/Chromosome20240904_NoAug_NoResize_coco/valid",
        help="染色体图像目录",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="work_dirs/ldmdet_dit_v6/backbone_vis",
        help="可视化输出目录",
    )
    parser.add_argument("--num-images", type=int, default=4, help="可视化图像数量")
    parser.add_argument("--input-size", type=int, default=512, help="输入图像尺寸 (需被 patch_size=16 整除)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载 DINOv3-Small 预训练模型
    print("Loading DINOv3-Small pretrained model...")
    model = create_model("vit_small_patch16_dinov3.lvd1689m", pretrained=True)
    model = model.to(device)
    model.eval()
    print(f"  Model loaded on {device}")
    print(f"  Patch size: {model.patch_embed.patch_size}")
    print(f"  Embed dim: {model.embed_dim}")
    print(f"  Num blocks: {len(model.blocks)}")
    print(f"  Num heads: {model.blocks[0].attn.num_heads}")

    # 获取图像列表
    image_dir = Path(args.image_dir)
    image_files = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
    if not image_files:
        print(f"No images found in {args.image_dir}")
        return

    image_files = image_files[: args.num_images]
    print(f"Found {len(image_files)} images to visualize")

    for img_path in image_files:
        print(f"\nProcessing: {img_path.name}")
        img_tensor, img_pil, (orig_h, orig_w) = load_image(str(img_path), args.input_size)
        img_tensor = img_tensor.to(device)

        # 提取特征
        features, attn_weights, feat_h, feat_w = extract_features(model, img_tensor)
        print(f"  Extracted {len(features)} block features, spatial: {feat_h}x{feat_w}")

        # 特征统计
        for name in ["block_0", f"block_{len(features) // 2}", f"block_{len(features) - 1}"]:
            feat = features[name]
            print(f"  {name}: mean={feat.mean():.4f}, std={feat.std():.4f}, "
                  f"min={feat.min():.4f}, max={feat.max():.4f}")

        # 输出子目录
        stem = img_path.stem
        out_subdir = os.path.join(args.output_dir, stem)
        os.makedirs(out_subdir, exist_ok=True)

        # 1. 通道均值可视化
        visualize_channel_mean(
            features,
            os.path.join(out_subdir, f"{stem}_channel_mean.png"),
            img_pil,
            feat_h,
            feat_w,
        )

        # 2. PCA 降维可视化
        visualize_pca(
            features,
            os.path.join(out_subdir, f"{stem}_pca_rgb.png"),
            img_pil,
            feat_h,
            feat_w,
        )

        # 3. Attention Map 可视化
        visualize_attention(
            attn_weights,
            os.path.join(out_subdir, f"{stem}_attention.png"),
            img_pil,
            feat_h,
            feat_w,
        )

        # 4. 特征相似度可视化
        visualize_feature_similarity(
            features,
            os.path.join(out_subdir, f"{stem}_similarity.png"),
            feat_h,
            feat_w,
        )

    print(f"\nAll visualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
