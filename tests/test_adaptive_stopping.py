import torch
import torch.nn as nn

from ldmdet.core.head import DiffusionDetHead


class _IdentitySampler:
    def xyxy_to_raw(self, boxes, img_metas):
        return boxes


def _minimal_head():
    head = DiffusionDetHead.__new__(DiffusionDetHead)
    nn.Module.__init__(head)
    head.adaptive_stop_geo_threshold = 0.1
    head.adaptive_stop_cls_threshold = 0.02
    head.adaptive_stop_score_threshold = 0.5
    head.adaptive_stop_min_box_scale = 0.0
    head.adaptive_stop_topk = 1
    head._sampler = _IdentitySampler()
    return head


def test_adaptive_stop_mask_combines_all_safety_gates():
    head = _minimal_head()
    metrics = {
        'geo_residual': torch.tensor([0.09, 0.11, 0.09]),
        'cls_residual': torch.tensor([0.01, 0.01, 0.03]),
        'mean_topk_score': torch.tensor([0.8, 0.8, 0.8]),
        'lower_box_scale': torch.tensor([0.1, 0.1, 0.1]),
    }
    assert head._adaptive_stop_mask(metrics).tolist() == [True, False, False]


def test_cascade_consistency_uses_final_top_scoring_proposal():
    head = _minimal_head()
    # [heads, batch, proposals, coordinates/classes]
    boxes = torch.tensor([
        [[[0., 0., 1., 1.], [0., 0., 2., 2.]]],
        [[[0., 0., 1., 1.], [0., 0., 4., 4.]]],
    ])
    logits = torch.tensor([
        [[[0., 0.], [1., 0.]]],
        [[[0., 0.], [3., 0.]]],
    ])
    metrics = head._cascade_consistency_metrics(
        logits, boxes, [{'img_shape': (4, 4)}])
    # Proposal 1 wins top-1; coordinate delta [0,0,2,2] has RMS sqrt(2).
    assert torch.allclose(
        metrics['geo_residual'], torch.tensor([2.0 ** 0.5]), atol=1e-6)
    assert metrics['cls_residual'].item() > 0
    assert metrics['mean_topk_score'].item() > 0.9
    assert metrics['lower_box_scale'].item() > 0
