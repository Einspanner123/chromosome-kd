"""测试 bbox 坐标转换"""

import torch
from ldmdet.utils.box_ops import bbox2roi, bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh


class TestBoxOps:
    def test_xyxy_to_cxcywh(self):
        bbox = torch.tensor([[100.0, 100.0, 200.0, 200.0], [50.0, 50.0, 150.0, 150.0]])
        result = bbox_xyxy_to_cxcywh(bbox)
        expected = torch.tensor([[150.0, 150.0, 100.0, 100.0], [100.0, 100.0, 100.0, 100.0]])
        assert torch.allclose(result, expected)

    def test_cxcywh_to_xyxy(self):
        bbox = torch.tensor([[150.0, 150.0, 100.0, 100.0], [100.0, 100.0, 100.0, 100.0]])
        result = bbox_cxcywh_to_xyxy(bbox)
        expected = torch.tensor([[100.0, 100.0, 200.0, 200.0], [50.0, 50.0, 150.0, 150.0]])
        assert torch.allclose(result, expected)

    def test_roundtrip(self):
        original = torch.tensor([[10.0, 20.0, 110.0, 220.0], [0.0, 0.0, 50.0, 80.0]])
        cxcywh = bbox_xyxy_to_cxcywh(original)
        xyxy = bbox_cxcywh_to_xyxy(cxcywh)
        assert torch.allclose(original, xyxy)

    def test_bbox2roi(self):
        bboxes = [torch.tensor([[10, 20, 30, 40]]), torch.tensor([[50, 60, 70, 80], [90, 100, 110, 120]])]
        rois = bbox2roi(bboxes)
        assert rois.shape == (3, 5)  # 3 boxes, [img_id, x1, y1, x2, y2]
        assert rois[0, 0] == 0  # first box from image 0
        assert rois[1, 0] == 1  # second box from image 1
