"""结构诊断: 从 A3 checkpoint 提取真实内部数据验证瓶颈假设

目的: 用实际推理数据 (而非代码推断) 验证 STRUCTURAL_IMPROVEMENT_ANALYSIS.md 中的 6 个瓶颈假设

诊断项:
  D1: RoI 空间信息利用率 — 7×7 RoI 特征的空间方差 vs DynamicConv squeeze 后的多样性
  D2: 时间条件化强度 — scale/shift 在不同 t 下的实际数值 (是否 ≈ 恒等变换)
  D3: 级联头贡献度 — 相邻头间 box_delta / cls_delta (识别冗余头)
  D4: 尺度-类别相关性 — 预测框尺寸 vs 预测类别的相关性 (模型是否已隐式学到)
  D5: 自注意力模式 — 注意力熵 + top-k 覆盖率 (是否发散)

Usage:
  python experiments/analysis/structural_diagnosis.py \
      --config experiments/configs/ldmdet/directions/mainline_ablation_24obj/a4_dpm_pp_24obj.py \
      --checkpoint work_dirs/a3_full_sota_24obj/best_coco_bbox_mAP_epoch_114.pth \
      --gpu-id 0 --max-images 50 \
      --output work_dirs/diagnosis/structural_diagnosis.json
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from mmengine.config import Config
from mmengine.hooks import Hook
from mmengine.runner import Runner


class StructuralDiagnosticHook(Hook):
    """结构诊断 Hook: 在推理时通过 PyTorch forward hooks 提取模型内部数据.

    挂载位置:
      - 每级 single_head.time_mlp: 捕获 time_emb → scale_shift (D2)
      - 每级 single_head.inst_interact (DynamicConv): 捕获 roi_features 输入 (D1)
      - 每级 single_head.self_attn: 捕获 attention weights (D5)
      - 每级 single_head.cls_head / reg_head: 捕获输出 (D3, D4)
      - head.py forward: 捕获 per-head cls_logits / pred_bboxes (D3)
    """

    def __init__(self, max_images=50):
        super().__init__()
        self.max_images = max_images
        self.image_count = 0
        self._hooks = []
        self._collected = defaultdict(list)  # key → list of per-image data

        # D5: 需要禁用 SDPA 才能获取 attention weights
        self._force_mha = True

    def _get_head(self, model):
        if hasattr(model, 'bbox_head'):
            return model.bbox_head
        if hasattr(model, 'module'):
            inner = model.module
            if hasattr(inner, 'bbox_head'):
                return inner.bbox_head
            if hasattr(inner, 'model') and hasattr(inner.model, 'bbox_head'):
                return inner.model.bbox_head
        return None

    def before_test(self, runner):
        """注册 PyTorch forward hooks 到 single_head 内部模块"""
        head = self._get_head(runner.model)
        if head is None:
            print('[DIAG] ERROR: cannot access bbox_head', flush=True)
            return

        print(f'[DIAG] head type: {type(head).__name__}', flush=True)
        print(f'[DIAG] num cascade heads: {len(head.head_series)}', flush=True)

        # D5: 禁用 SDPA, 强制使用 nn.MultiheadAttention (返回 weights)
        if self._force_mha:
            for i, sh in enumerate(head.head_series):
                sh.use_sdpa = False
                print(f'[DIAG] head{i}: use_sdpa forced to False for attention extraction', flush=True)

        # 为每级 cascade head 注册 hooks
        for i, sh in enumerate(head.head_series):
            self._register_single_head_hooks(i, sh)

        print(f'[DIAG] registered {len(self._hooks)} forward hooks', flush=True)

    def _register_single_head_hooks(self, head_idx, single_head):
        """为单个 cascade head 注册诊断 hooks"""

        # D2: time_mlp (scale_shift 模式) 或 adaln_mlp (adaln_zero 模式)
        tc = single_head.time_conditioning
        if tc == 'scale_shift' and hasattr(single_head, 'time_mlp'):
            target_mlp = single_head.time_mlp
            mode = 'scale_shift'
        elif tc == 'adaln_zero' and hasattr(single_head, 'adaln_mlp'):
            target_mlp = single_head.adaln_mlp
            mode = 'adaln_zero'
        else:
            print(f'[DIAG] head{head_idx}: unknown time_conditioning={tc}, skip D2', flush=True)
            target_mlp = None
            mode = None

        if target_mlp is not None:
            def make_time_hook(hidx, m):
                def hook(module, inp, out):
                    # inp[0] = time_emb (after SiLU, [bs, 1024])
                    # out = scale_shift [bs, 512] or adaln_params [bs, 1536]
                    if self.image_count >= self.max_images:
                        return
                    time_emb = inp[0].detach().cpu()
                    output = out.detach().cpu()
                    if m == 'scale_shift':
                        scale, shift = output.chunk(2, dim=-1)
                        self._collected[f'D2/head{hidx}/scale'].append(scale.numpy())
                        self._collected[f'D2/head{hidx}/shift'].append(shift.numpy())
                    elif m == 'adaln_zero':
                        # 6 chunks: gamma1,beta1,alpha1,gamma2,beta2,alpha2
                        chunks = output.chunk(6, dim=-1)
                        alphas = torch.cat([chunks[2], chunks[5]], dim=-1)  # alpha1, alpha2
                        self._collected[f'D2/head{hidx}/adaln_alphas'].append(alphas.numpy())
                return hook
            self._hooks.append(target_mlp.register_forward_hook(make_time_hook(head_idx, mode)))

        # D1: DynamicConv — 捕获 roi_features 输入
        dynconv = single_head.inst_interact
        def make_dynconv_hook(hidx):
            def hook(module, inp, out):
                if self.image_count >= self.max_images:
                    return
                # inp = (proposals [1, N, 256], roi_features [49, N, 256])
                # out = [1, N, 256]
                if len(inp) >= 2:
                    roi_feat = inp[1].detach()  # [49, N, 256]
                    # D1a: RoI 空间方差 (7×7 内部的方差)
                    # roi_feat: [49, N, 256] → reshape [7, 7, N, 256]
                    N = roi_feat.shape[1]
                    roi_spatial = roi_feat.permute(1, 2, 0).reshape(N, 256, 7, 7)
                    spatial_var = roi_spatial.var(dim=[2, 3]).mean().item()  # 标量
                    # D1b: DynamicConv 输出多样性 (proposal 间方差)
                    out_feat = out.detach().squeeze(0)  # [N, 256]
                    proposal_div = out_feat.var(dim=0).mean().item()
                    self._collected[f'D1/head{hidx}/roi_spatial_var'].append(spatial_var)
                    self._collected[f'D1/head{hidx}/dynconv_out_diversity'].append(proposal_div)
            return hook
        self._hooks.append(dynconv.register_forward_hook(make_dynconv_hook(head_idx)))

        # D5: self_attn — 捕获 attention weights
        attn = single_head.self_attn
        def make_attn_hook(hidx):
            def hook(module, inp, out):
                if self.image_count >= self.max_images:
                    return
                # nn.MultiheadAttention 返回 (attn_output, attn_weights)
                # attn_weights: [bs*num_heads, seq, seq] (need need_weights=True)
                if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
                    weights = out[1].detach().cpu()  # [bs, num_heads, N, N] or [N, N]
                    if weights.dim() == 3:
                        weights = weights.unsqueeze(0)
                    # 注意力熵
                    probs = weights.clamp(min=1e-8)
                    entropy = -(probs * probs.log()).sum(-1).mean().item()
                    # Top-10 覆盖率
                    topk_cov = probs.topk(10, dim=-1)[0].sum(-1).mean().item()
                    self._collected[f'D5/head{hidx}/attn_entropy'].append(entropy)
                    self._collected[f'D5/head{hidx}/attn_topk10_coverage'].append(topk_cov)
            return hook
        self._hooks.append(attn.register_forward_hook(make_attn_hook(head_idx)))

        # D4: cls_head + reg_head — 捕获输出用于尺度-类别相关性
        def make_cls_hook(hidx):
            def hook(module, inp, out):
                if self.image_count >= self.max_images:
                    return
                # out: [N, num_classes]
                self._collected[f'D4/head{hidx}/cls_logits'].append(out.detach().cpu().numpy())
            return hook
        self._hooks.append(single_head.cls_head.register_forward_hook(make_cls_hook(head_idx)))

        def make_reg_hook(hidx):
            def hook(module, inp, out):
                if self.image_count >= self.max_images:
                    return
                # out: [N, 4] bbox deltas
                self._collected[f'D4/head{hidx}/reg_deltas'].append(out.detach().cpu().numpy())
            return hook
        self._hooks.append(single_head.reg_head.register_forward_hook(make_reg_hook(head_idx)))

        # D4-fix: 捕获 single_head 整体输出 (含 pred_bboxes xyxy) 用于真实尺寸-类别相关性
        def make_head_output_hook(hidx):
            def hook(module, inp, out):
                if self.image_count >= self.max_images:
                    return
                # out = (class_logits [bs, N, C], pred_bboxes [bs, N, 4] xyxy, fc_feature)
                if isinstance(out, tuple) and len(out) >= 2:
                    cls_logits = out[0].detach().cpu().numpy()  # [bs, N, C]
                    pred_bboxes = out[1].detach().cpu().numpy()  # [bs, N, 4] xyxy
                    self._collected[f'D4fix/head{hidx}/cls_logits'].append(cls_logits)
                    self._collected[f'D4fix/head{hidx}/pred_bboxes'].append(pred_bboxes)
            return hook
        self._hooks.append(single_head.register_forward_hook(make_head_output_hook(head_idx)))

    def after_test_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        self.image_count += len(outputs) if isinstance(outputs, list) else 1
        if self.image_count <= 3:
            print(f'[DIAG] processed {self.image_count} images so far', flush=True)

    def after_test(self, runner):
        """清理 hooks"""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        print(f'[DIAG] hooks removed. collected {len(self._collected)} keys', flush=True)

    def compute_stats(self):
        """汇总所有诊断数据"""
        stats = {'n_images': self.image_count}

        # D1: RoI 空间信息利用率
        d1 = {}
        for key, vals in self._collected.items():
            if not key.startswith('D1/'):
                continue
            arr = np.array(vals)
            d1[key] = {
                'n': len(arr),
                'mean': float(arr.mean()),
                'std': float(arr.std()),
                'median': float(np.median(arr)),
            }
        stats['D1_roi_spatial_info'] = d1

        # D2: 时间条件化强度
        d2 = {}
        for key, vals in self._collected.items():
            if not key.startswith('D2/'):
                continue
            arr = np.array(vals)  # [n_images, bs, dim]
            # scale 应偏离 1, shift 应偏离 0
            if 'scale' in key:
                deviation = np.abs(arr - 1.0).mean()
                d2[key] = {'mean_abs_deviation_from_1': float(deviation),
                           'mean': float(arr.mean()), 'std': float(arr.std())}
            elif 'shift' in key:
                magnitude = np.abs(arr).mean()
                d2[key] = {'mean_abs_magnitude': float(magnitude),
                           'mean': float(arr.mean()), 'std': float(arr.std())}
            elif 'adaln_alphas' in key:
                # alpha 应偏离 0 (零初始化), 非 0 表示时间条件化生效
                magnitude = np.abs(arr).mean()
                d2[key] = {'mean_abs_alpha': float(magnitude),
                           'mean': float(arr.mean()), 'std': float(arr.std())}
        stats['D2_time_conditioning'] = d2

        # D3: 级联头贡献度 (从 D4 的 cls_logits 和 reg_deltas 推断)
        # 注意: D3 需要同一张图内不同 head 的输出配对, 这里简化为统计 head 间统计量差异
        d3 = {}
        for hidx in range(6):
            cls_key = f'D4/head{hidx}/cls_logits'
            reg_key = f'D4/head{hidx}/reg_deltas'
            if cls_key in self._collected:
                cls_arr = np.array(self._collected[cls_key])
                d3[f'head{hidx}/cls_logits_mean'] = float(cls_arr.mean())
                d3[f'head{hidx}/cls_logits_std'] = float(cls_arr.std())
            if reg_key in self._collected:
                reg_arr = np.array(self._collected[reg_key])
                d3[f'head{hidx}/reg_deltas_mean'] = float(reg_arr.mean())
                d3[f'head{hidx}/reg_deltas_std'] = float(reg_arr.std())
        stats['D3_cascade_head_contribution'] = d3

        # D4: 尺度-类别相关性
        d4 = {}
        # 用最后一级 head 的 cls_logits + reg_deltas 做相关性分析
        cls_key = 'D4/head5/cls_logits'
        reg_key = 'D4/head5/reg_deltas'
        if cls_key in self._collected and reg_key in self._collected:
            all_cls = np.concatenate(self._collected[cls_key], axis=0)  # [total_N, 24]
            all_reg = np.concatenate(self._collected[reg_key], axis=0)  # [total_N, 4]
            pred_cls = all_cls.argmax(axis=-1)  # [total_N]
            # reg_deltas: [dx, dy, dw, dh], dw/dh 是对数尺度
            pred_dw = all_reg[:, 2]
            pred_dh = all_reg[:, 3]
            # 对每个类, 统计 dw/dh 的均值 (反映尺寸)
            class_sizes = {}
            for c in range(24):
                mask = pred_cls == c
                if mask.sum() > 5:
                    class_sizes[f'class{c}'] = {
                        'n': int(mask.sum()),
                        'mean_dw': float(pred_dw[mask].mean()),
                        'mean_dh': float(pred_dh[mask].mean()),
                        'mean_log_size': float((pred_dw[mask] + pred_dh[mask]).mean() / 2),
                    }
            d4['per_class_size'] = class_sizes
            # 类间尺寸方差 (越大 → 尺寸-类别相关性越强)
            if class_sizes:
                mean_sizes = [v['mean_log_size'] for v in class_sizes.values()]
                d4['inter_class_size_std'] = float(np.std(mean_sizes))
                d4['n_classes_with_data'] = len(class_sizes)
        stats['D4_scale_class_correlation'] = d4

        # D4-fix: 基于实际预测框尺寸 (xyxy) 的尺度-类别相关性
        d4fix = {}
        cls_key = 'D4fix/head5/cls_logits'
        bbox_key = 'D4fix/head5/pred_bboxes'
        if cls_key in self._collected and bbox_key in self._collected:
            # cls_logits: list of [bs, N, C], pred_bboxes: list of [bs, N, 4] xyxy
            all_cls = np.concatenate(
                [a.reshape(-1, a.shape[-1]) for a in self._collected[cls_key]], axis=0
            )  # [total_N, C]
            all_bbox = np.concatenate(
                [b.reshape(-1, b.shape[-1]) for b in self._collected[bbox_key]], axis=0
            )  # [total_N, 4] xyxy
            # 只看高置信度预测 (sigmoid > 0.3)
            scores = 1.0 / (1.0 + np.exp(-all_cls))  # sigmoid
            max_scores = scores.max(axis=-1)
            conf_mask = max_scores > 0.3
            all_cls = all_cls[conf_mask]
            all_bbox = all_bbox[conf_mask]
            pred_cls = all_cls.argmax(axis=-1)
            # 计算实际框面积 = w * h (像素²)
            w = all_bbox[:, 2] - all_bbox[:, 0]
            h = all_bbox[:, 3] - all_bbox[:, 1]
            areas = w * h
            log_areas = np.log(areas.clip(min=1.0))
            class_sizes = {}
            for c in range(24):
                mask = pred_cls == c
                if mask.sum() > 5:
                    class_sizes[f'class{c}'] = {
                        'n': int(mask.sum()),
                        'mean_area_px2': float(areas[mask].mean()),
                        'mean_log_area': float(log_areas[mask].mean()),
                        'mean_w': float(w[mask].mean()),
                        'mean_h': float(h[mask].mean()),
                        'std_log_area': float(log_areas[mask].std()),
                    }
            d4fix['per_class_size'] = class_sizes
            if class_sizes:
                mean_log_areas = [v['mean_log_area'] for v in class_sizes.values()]
                d4fix['inter_class_log_area_std'] = float(np.std(mean_log_areas))
                d4fix['inter_class_log_area_range'] = float(max(mean_log_areas) - min(mean_log_areas))
                d4fix['n_classes_with_data'] = len(class_sizes)
                d4fix['n_confident_preds'] = int(conf_mask.sum())
        stats['D4fix_scale_class_correlation'] = d4fix

        # D5: 自注意力模式
        d5 = {}
        for key, vals in self._collected.items():
            if not key.startswith('D5/'):
                continue
            arr = np.array(vals)
            d5[key] = {
                'n': len(arr),
                'mean': float(arr.mean()),
                'std': float(arr.std()),
                'median': float(np.median(arr)),
            }
        stats['D5_attention_pattern'] = d5

        return stats


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description='结构诊断: 从 checkpoint 提取内部数据')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint path')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-images', type=int, default=50)
    parser.add_argument('--output', required=True, help='Output JSON path')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    set_seed(args.seed)

    cfg = Config.fromfile(args.config)
    cfg.work_dir = os.path.dirname(args.checkpoint) or 'work_dirs/diagnosis'

    # 切换为 val 模式
    cfg.test_dataloader = cfg.val_dataloader
    cfg.test_evaluator = cfg.val_evaluator

    # 禁用 SwanLab
    cfg.vis_backends = [dict(type='LocalVisBackend')]
    if 'visualizer' in cfg:
        cfg.visualizer['vis_backends'] = [dict(type='LocalVisBackend')]

    # 限制评估图片数 (通过修改 dataloader 的 dataset)
    # 注意: mmengine 的 test_dataloader 可能不支持直接限制, 用 Hook 计数
    cfg.custom_hooks = []

    diag_hook = StructuralDiagnosticHook(max_images=args.max_images)

    runner = Runner.from_cfg(cfg)
    runner.load_checkpoint(args.checkpoint)
    runner.register_hook(diag_hook, priority='LOW')

    print(f'[DIAG] checkpoint: {args.checkpoint}', flush=True)
    print(f'[DIAG] max_images: {args.max_images}', flush=True)
    print(f'[DIAG] solver={cfg.model.bbox_head.get("solver_type", "unknown")}, '
          f'steps={cfg.model.bbox_head.get("sampling_timesteps", "unknown")}', flush=True)

    runner.test()

    stats = diag_hook.compute_stats()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f'\n[DIAG] === 诊断结果摘要 ===', flush=True)
    print(f'[DIAG] 处理图片数: {stats["n_images"]}', flush=True)

    # D1 摘要
    d1 = stats.get('D1_roi_spatial_info', {})
    if d1:
        print(f'\n[D1] RoI 空间信息利用率:', flush=True)
        for hidx in range(6):
            sv_key = f'D1/head{hidx}/roi_spatial_var'
            dv_key = f'D1/head{hidx}/dynconv_out_diversity'
            if sv_key in d1 and dv_key in d1:
                print(f'  head{hidx}: roi_spatial_var={d1[sv_key]["mean"]:.4f}  '
                      f'dynconv_out_diversity={d1[dv_key]["mean"]:.4f}', flush=True)

    # D2 摘要
    d2 = stats.get('D2_time_conditioning', {})
    if d2:
        print(f'\n[D2] 时间条件化强度:', flush=True)
        for hidx in range(6):
            for suffix in ['scale', 'shift', 'adaln_alphas']:
                key = f'D2/head{hidx}/{suffix}'
                if key in d2:
                    if suffix == 'scale':
                        print(f'  head{hidx}: scale |deviation from 1|={d2[key]["mean_abs_deviation_from_1"]:.4f}  '
                              f'(mean={d2[key]["mean"]:.4f}, std={d2[key]["std"]:.4f})', flush=True)
                    elif suffix == 'shift':
                        print(f'  head{hidx}: shift |magnitude|={d2[key]["mean_abs_magnitude"]:.4f}  '
                              f'(mean={d2[key]["mean"]:.4f}, std={d2[key]["std"]:.4f})', flush=True)
                    elif suffix == 'adaln_alphas':
                        print(f'  head{hidx}: adaln |alpha|={d2[key]["mean_abs_alpha"]:.4f}  '
                              f'(mean={d2[key]["mean"]:.4f}, std={d2[key]["std"]:.4f})', flush=True)

    # D3 摘要
    d3 = stats.get('D3_cascade_head_contribution', {})
    if d3:
        print(f'\n[D3] 级联头统计量 (cls_logits / reg_deltas):', flush=True)
        for hidx in range(6):
            ck = f'head{hidx}/cls_logits_mean'
            rk = f'head{hidx}/reg_deltas_mean'
            if ck in d3 and rk in d3:
                print(f'  head{hidx}: cls_mean={d3[ck]:.4f} (std={d3[f"head{hidx}/cls_logits_std"]:.4f})  '
                      f'reg_mean={d3[rk]:.4f} (std={d3[f"head{hidx}/reg_deltas_std"]:.4f})', flush=True)

    # D4 摘要 (旧, 基于 reg_deltas — 有缺陷)
    d4 = stats.get('D4_scale_class_correlation', {})
    if d4:
        print(f'\n[D4] 尺度-类别相关性 (基于 reg_deltas, 有缺陷):', flush=True)
        print(f'  inter_class_size_std={d4.get("inter_class_size_std", "N/A")}', flush=True)

    # D4-fix 摘要 (基于实际预测框尺寸)
    d4fix = stats.get('D4fix_scale_class_correlation', {})
    if d4fix:
        print(f'\n[D4-fix] 尺度-类别相关性 (基于实际预测框面积):', flush=True)
        print(f'  inter_class_log_area_std={d4fix.get("inter_class_log_area_std", "N/A")}', flush=True)
        print(f'  inter_class_log_area_range={d4fix.get("inter_class_log_area_range", "N/A")}', flush=True)
        print(f'  n_confident_preds={d4fix.get("n_confident_preds", 0)}', flush=True)
        pcs = d4fix.get('per_class_size', {})
        if pcs:
            class_names = ['A1','A2','A3','B4','B5','C6','C7','C8','C9','C10','C11','C12',
                           'D13','D14','D15','E16','E17','E18','F19','F20','G21','G22','X','Y']
            for c in sorted(pcs.keys(), key=lambda x: int(x.replace('class', ''))):
                cidx = int(c.replace('class', ''))
                name = class_names[cidx] if cidx < 24 else c
                print(f'  {name}({c}): log_area={pcs[c]["mean_log_area"]:.2f}  '
                      f'w={pcs[c]["mean_w"]:.1f} h={pcs[c]["mean_h"]:.1f}  '
                      f'(n={pcs[c]["n"]})', flush=True)

    # D5 摘要
    d5 = stats.get('D5_attention_pattern', {})
    if d5:
        print(f'\n[D5] 自注意力模式:', flush=True)
        for hidx in range(6):
            ek = f'D5/head{hidx}/attn_entropy'
            tk = f'D5/head{hidx}/attn_topk10_coverage'
            if ek in d5 and tk in d5:
                print(f'  head{hidx}: entropy={d5[ek]["mean"]:.4f} (std={d5[ek]["std"]:.4f})  '
                      f'topk10_cov={d5[tk]["mean"]:.4f}', flush=True)

    print(f'\n[DIAG] 完整结果已保存到: {args.output}', flush=True)


if __name__ == '__main__':
    main()
