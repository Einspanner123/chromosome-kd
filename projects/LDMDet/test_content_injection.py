"""测试内容注入是否能正常启动训练 (不需要 GPU, 检查架构放得通)"""
import sys
sys.path.insert(0, '/home/linkst/workplace/chromo/chromosome-kd')

import torch
torch.manual_seed(42)

from mmengine import Config
from mmengine.registry import MODELS

config_path = 'projects/LDMDet/configs/ldmdet_dit.py'
cfg = Config.fromfile(config_path)

cfg.model.bbox_head.num_proposals = 100

print("构建模型...")
model = MODELS.build(cfg.model)
model.eval()

print(f"模型参数总数: {sum(p.numel() for p in model.parameters()):,}")
print(f"可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# 模拟输入
bs = 1
device = next(model.parameters()).device
img = torch.randn(bs, 3, 800, 800, device=device)
img_metas = [{'img_shape': (800, 800), 'batch_input_shape': (800, 800),
               'scale_factor': (1.0, 1.0)} for _ in range(bs)]

print("\n=== 测试前向传播 ===")
with torch.no_grad():
    feats = model.extract_feat(img)
    loss_dict = model.bbox_head.loss(feats, img_metas, img_metas)

print(f"\nLoss: {loss_dict}")
print(f"总loss: {sum(v.item() for v in loss_dict.values() if isinstance(v, torch.Tensor)):.4f}")

print("\n=== 测试反向传播 ===")
for k, v in loss_dict.items():
    if isinstance(v, torch.Tensor):
        v.backward(retain_graph=True)
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                        for p in model.bbox_head.parameters()
                        if p.requires_grad)
        print(f"  {k}: backward 成功, 梯度非零: {has_grad}")

print("\n所有测试通过!")