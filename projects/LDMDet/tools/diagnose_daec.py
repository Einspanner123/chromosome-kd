"""DAEC (Detection-Aware Entropic Coupling) 深度诊断工具。

对比 SOTA 基准模型与当前的 DAEC + Contrastive 对齐模型。
测量 1: 检测头参数一致性审计。
测量 2: 同源染色体特征对齐审计 (核心：验证 Contrastive Loss 是否打破了信息截断)。
"""

import os
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
)

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOTA_CONFIG = os.path.join(
    _BASE,
    'configs',
    '_legacy',
    'ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5.py',
)
DAEC_CONFIG = os.path.join(
    _BASE, 'configs', 'recipes', 'ldmdet_daec_v2_semantic.py'
)

SOTA_CKPT = (
    'work_dirs/reproduce_0751_stochot_eps5_v2/best_coco_bbox_mAP_epoch_59.pth'
)
DAEC_CKPT = 'work_dirs/ldmdet_daec_v2_semantic/best_coco_bbox_mAP_epoch_48.pth'


def _compare_param_groups(sota_sd, target_sd, prefix, label):
    """对比指定 prefix 下的参数组余弦相似度。"""
    sota_keys = {k: v for k, v in sota_sd.items() if k.startswith(prefix)}
    target_keys = {k: v for k, v in target_sd.items() if k.startswith(prefix)}
    shared = set(sota_keys) & set(target_keys)
    if not shared:
        return 1.0

    cos_vals = []
    for k in sorted(shared):
        ps = sota_keys[k].float().flatten()
        pk = target_keys[k].float().flatten()
        if ps.numel() == 0 or pk.numel() == 0:
            continue
        if ps.norm() < 1e-6 or pk.norm() < 1e-6:
            continue
        cos = F.cosine_similarity(ps.unsqueeze(0), pk.unsqueeze(0)).item()
        cos_vals.append(cos)

    return sum(cos_vals) / len(cos_vals) if cos_vals else 1.0


def measurement_1():
    """测量 1: 检测头参数一致性审计"""
    print('\n' + '=' * 70)
    print('测量 1: 检测头参数一致性审计 (SOTA vs DAEC)')
    print('=' * 70)

    device = 'cpu'
    sota_sd = torch.load(
        SOTA_CKPT, map_location=device, weights_only=False
    ).get('state_dict', {})
    daec_sd = torch.load(
        DAEC_CKPT, map_location=device, weights_only=False
    ).get('state_dict', {})

    if not sota_sd or not daec_sd:
        print('❌ 权重加载失败。')
        return

    # 1. 整体 bbox_head
    head_cos = _compare_param_groups(
        sota_sd, daec_sd, 'bbox_head.head_series.', 'Overall Head'
    )
    print(f'总体 head_series 余弦相似度: {head_cos:.6f}')

    # 2. 子模块分析
    sub_prefixes = [
        ('cls_head', '分类头 (Cls Head)'),
        ('reg_head', '回归头 (Reg Head)'),
        ('self_attn', '自注意力 (Self Attn)'),
        ('inst_interact', '实例交互 (Inst Interact)'),
        ('adaln_mlp', '时间调节 (AdaLN MLP)'),
    ]

    print('\n子模块详细差异:')
    for key, label in sub_prefixes:
        prefix = f'bbox_head.head_series.0.{key}.'
        cos = _compare_param_groups(sota_sd, daec_sd, prefix, label)
        print(f'  {label:<20}: cos={cos:.6f}')


def measurement_2():
    """测量 2: 同源染色体特征对齐审计"""
    print('\n' + '=' * 70)
    print('测量 2: 同源染色体特征对齐审计 (验证 Contrastive 对齐效果)')
    print('=' * 70)

    device = torch.device('cpu')  # 强制使用 CPU 跑诊断
    torch.manual_seed(42)

    from mmengine.config import Config

    from projects.LDMDet.mods.diffusiondet_head import DiffusionDetHead
    from projects.LDMDet.mods.roi_extractor import SingleRoIExtractor
    from projects.LDMDet.mods.single_head import SingleDiffusionDetHead

    # 构建并加载模型
    def get_head(config_path, ckpt_path):
        cfg = Config.fromfile(config_path)
        head_cfg = cfg.model.bbox_head

        sh_cfg = head_cfg.single_head.copy()
        sh_cfg.pop('type', None)
        single_head = SingleDiffusionDetHead(**sh_cfg)

        re_cfg = head_cfg.roi_extractor.copy()
        re_cfg.pop('type', None)
        roi_extractor = SingleRoIExtractor(**re_cfg)

        head_kwargs = {
            k: v
            for k, v in head_cfg.items()
            if k not in ('type', 'single_head', 'roi_extractor', 'criterion')
        }
        head = DiffusionDetHead(
            single_head=single_head,
            roi_extractor=roi_extractor,
            **head_kwargs,
        )

        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        sd = ckpt.get('state_dict', ckpt)
        head_sd = {
            k.replace('bbox_head.', ''): v
            for k, v in sd.items()
            if k.startswith('bbox_head.')
        }
        head.load_state_dict(head_sd, strict=False)
        head.to(device)
        head.eval()
        return head

    print('正在加载 SOTA 模型...')
    sota_head = get_head(SOTA_CONFIG, SOTA_CKPT)
    print('正在加载 DAEC 模型...')
    daec_head = get_head(DAEC_CONFIG, DAEC_CKPT)

    # 构造测试输入
    gt_labels = torch.tensor(
        [0, 0, 21, 21, 22], device=device
    )  # A1, A1, G22, G22, X
    gt_boxes_cxcywh = torch.tensor(
        [
            [0.25, 0.25, 0.30, 0.20],
            [0.65, 0.25, 0.30, 0.20],
            [0.35, 0.55, 0.10, 0.08],
            [0.55, 0.55, 0.10, 0.08],
            [0.80, 0.70, 0.12, 0.10],
        ],
        device=device,
    )

    dummy_feat = [
        torch.randn(1, 256, 200, 200, device=device),
        torch.randn(1, 256, 100, 100, device=device),
        torch.randn(1, 256, 50, 50, device=device),
        torch.randn(1, 256, 25, 25, device=device),
    ]

    class _Meta:
        def __init__(self):
            self.img_shape = (800, 800)
            self.ori_shape = (800, 800)
            self.scale_factor = 1.0
            self.pad_shape = (800, 800)
            self.batch_input_shape = (800, 800)

    def run_inference(head):
        gt_diff = (gt_boxes_cxcywh * 2 - 1) * head.snr_scale
        noise = torch.randn(head.num_proposals, 4, device=device)
        t = torch.tensor([0.5], device=device)

        # 使用普通的 sinkhorn 匹配 (测量对齐度)
        cost = torch.cdist(noise, gt_diff, p=2)
        transport = head._sinkhorn_transport(cost)
        matched_gt_idx = transport.argmax(dim=1)

        x_start = gt_diff[matched_gt_idx]
        x_noisy, _ = head.rf.q_sample(x_start, x_noise=noise, t=t)
        curr_bboxes = head._raw_to_xyxy(x_noisy.unsqueeze(0), [_Meta()])
        time_emb = head.time_mlp(t * head.timesteps)

        # 跑第一层 head 提取特征
        result = head.head_series[0](
            dummy_feat, curr_bboxes, None, head.roi_extractor, time_emb
        )
        # result: (cls, reg, obj_feat, objness, vel)
        return result[2].squeeze(0), matched_gt_idx

    print('\n特征分析结果:')
    print(
        f'{"模型":<10} {"同源对":<10} {"类内cos":>10} {"类间cos":>10} {"对齐比":>10}'
    )
    print('-' * 55)

    for label, head in [('SOTA', sota_head), ('DAEC', daec_head)]:
        feat, matched_gt_idx = run_inference(head)

        gt_to_props = defaultdict(list)
        for pi in range(head.num_proposals):
            gt_to_props[matched_gt_idx[pi].item()].append(pi)

        # A1 对: (0, 1), G22 对: (2, 3)
        chromo_pairs = [(0, 1), (2, 3)]
        all_intra, all_inter = [], []

        for gti, gtj in chromo_pairs:
            pi, pj = gt_to_props[gti], gt_to_props[gtj]
            if len(pi) < 2 or len(pj) < 2:
                continue

            fi, fj = feat[pi], feat[pj]
            intra_mat = F.cosine_similarity(
                fi.unsqueeze(1), fi.unsqueeze(0), dim=2
            )
            mask = ~torch.eye(len(pi), dtype=torch.bool, device=device)
            intra_val = intra_mat[mask].mean().item()
            inter_val = (
                F.cosine_similarity(fi.unsqueeze(1), fj.unsqueeze(0), dim=2)
                .mean()
                .item()
            )

            all_intra.append(intra_val)
            all_inter.append(inter_val)

        avg_in = sum(all_intra) / len(all_intra) if all_intra else 0
        avg_out = sum(all_inter) / len(all_inter) if all_inter else 0
        ratio = avg_in / max(avg_out, 1e-8)
        print(
            f'{label:<10} {"Mixed":<10} {avg_in:>10.4f} {avg_out:>10.4f} {ratio:>10.2f}'
        )

    print('\n结论:')
    print(
        '1. 如果 DAEC 的对齐比显著高于 SOTA (例如 > 1.10), 说明 Contrastive Loss 成功打破了信息截断。'
    )
    print(
        '2. 如果 DAEC 的类内 cos 极高但类间也极高, 说明特征空间发生了坍缩, 需调低温度系数或权重。'
    )


def main():
    measurement_1()
    measurement_2()


if __name__ == '__main__':
    main()
