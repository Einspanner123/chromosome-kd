"""三种 attention 模式精度对比（固定所有随机种子）"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ldmdet.tools.profile_training import build_model, generate_synthetic_batch
import torch, argparse, copy

args = argparse.Namespace(
    bs=8, num_proposals=500, num_heads=6, num_classes=24,
    feat_channels=256, pooler_resolution=7, img_size=512,
    snr_scale=2.0, diffusion_type='rectified_flow',
    rf_schedule='shifted', rf_shift=2.0,
    coupling='ghss', ot_epsilon=5.0, ot_num_iters=20,
    time_conditioning='scale_shift', scale_aware=False,
)
device = torch.device('cuda')

args.use_sdpa = False; args.attn_half = False
head_base = build_model(args).to(device)
state = copy.deepcopy(head_base.state_dict())

head_off = build_model(args).to(device)
head_off.load_state_dict(state); head_off.eval()

args.use_sdpa = True; args.attn_half = False
head_sdpa = build_model(args).to(device)
head_sdpa.load_state_dict(state); head_sdpa.eval()

args.use_sdpa = True; args.attn_half = True
head_fp16 = build_model(args).to(device)
head_fp16.load_state_dict(state); head_fp16.eval()

torch.manual_seed(0)
fixed_data = [generate_synthetic_batch(args, device) for _ in range(10)]
features, img_metas, gt_bboxes, gt_labels = fixed_data[0]

# 固定种子跑 loss（t 采样和 Sinkhorn 采样有随机性）
def get_losses(head, seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    with torch.no_grad():
        return head.loss(features, img_metas, gt_bboxes, gt_labels)

l_off = get_losses(head_off)
l_sdpa = get_losses(head_sdpa)
l_fp16 = get_losses(head_fp16)

print('=== Loss 绝对值 & 差异 (固定种子) ===')
print(f'{"Loss":<20} {"Baseline":>10} {"SDPA FP32":>10} {"SDPA FP16":>10} {"|d|FP32":>10} {"|d|FP16":>10}')
for k in l_off:
    vb = l_off[k].item()
    vs = l_sdpa[k].item()
    vf = l_fp16[k].item()
    ds = abs(vb - vs)
    df = abs(vb - vf)
    print(f'{k:<20} {vb:10.4f} {vs:10.4f} {vf:10.4f} {ds:10.6f} {df:10.6f}')
