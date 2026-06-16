"""测试 ldmdet.utils.box_ops — 坐标转换与 ROI 工具"""

import torch
import pytest
from ldmdet.utils.box_ops import bbox_xyxy_to_cxcywh, bbox_cxcywh_to_xyxy, bbox2roi


class TestBboxXyxyToCxcywh:
    def test_basic(self):
        xyxy = torch.tensor([[10.0, 20.0, 30.0, 60.0]])
        cxcywh = bbox_xyxy_to_cxcywh(xyxy)
        expected = torch.tensor([[20.0, 40.0, 20.0, 40.0]])
        assert torch.allclose(cxcywh, expected, atol=1e-5)

    def test_batch(self):
        xyxy = torch.tensor([
            [0.0, 0.0, 100.0, 200.0],
            [10.0, 20.0, 30.0, 60.0],
        ])
        cxcywh = bbox_xyxy_to_cxcywh(xyxy)
        assert cxcywh.shape == (2, 4)
        expected = torch.tensor([
            [50.0, 100.0, 100.0, 200.0],
            [20.0, 40.0, 20.0, 40.0],
        ])
        assert torch.allclose(cxcywh, expected, atol=1e-5)

    def test_deterministic(self):
        xyxy = torch.rand(100, 4)
        xyxy[:, 2:] += xyxy[:, :2]
        r1 = bbox_xyxy_to_cxcywh(xyxy)
        r2 = bbox_xyxy_to_cxcywh(xyxy)
        assert torch.allclose(r1, r2, atol=1e-7)


class TestBboxCxcywhToXyxy:
    def test_basic(self):
        cxcywh = torch.tensor([[20.0, 40.0, 20.0, 40.0]])
        xyxy = bbox_cxcywh_to_xyxy(cxcywh)
        expected = torch.tensor([[10.0, 20.0, 30.0, 60.0]])
        assert torch.allclose(xyxy, expected, atol=1e-5)

    def test_batch(self):
        cxcywh = torch.tensor([
            [50.0, 100.0, 100.0, 200.0],
            [20.0, 40.0, 20.0, 40.0],
        ])
        xyxy = bbox_cxcywh_to_xyxy(cxcywh)
        assert xyxy.shape == (2, 4)


class TestRoundtrip:
    def test_xyxy_cxcywh_roundtrip(self):
        """xyxy → cxcywh → xyxy 往返一致"""
        xyxy = torch.rand(100, 4)
        xyxy[:, 2:] += xyxy[:, :2]  # 确保 x2 > x1, y2 > y1
        recovered = bbox_cxcywh_to_xyxy(bbox_xyxy_to_cxcywh(xyxy))
        assert torch.allclose(xyxy, recovered, atol=1e-5)

    def test_cxcywh_xyxy_roundtrip(self):
        """cxcywh → xyxy → cxcywh 往返一致"""
        cxcywh = torch.rand(100, 4)
        cxcywh[:, 2:] += 0.1  # 确保 w, h > 0
        recovered = bbox_xyxy_to_cxcywh(bbox_cxcywh_to_xyxy(cxcywh))
        assert torch.allclose(cxcywh, recovered, atol=1e-5)


class TestBbox2roi:
    def test_basic(self):
        bboxes_list = [
            torch.tensor([[10.0, 20.0, 30.0, 60.0], [5.0, 5.0, 15.0, 25.0]]),
            torch.tensor([[100.0, 200.0, 300.0, 400.0]]),
        ]
        rois = bbox2roi(bboxes_list)
        assert rois.shape == (3, 5)
        # batch_idx
        assert rois[0, 0].item() == 0
        assert rois[1, 0].item() == 0
        assert rois[2, 0].item() == 1
        # 坐标
        assert torch.allclose(rois[0, 1:], bboxes_list[0][0])
        assert torch.allclose(rois[2, 1:], bboxes_list[0][0]) is False  # 不同图
        assert torch.allclose(rois[2, 1:], bboxes_list[1][0])

    def test_single_image(self):
        bboxes_list = [torch.rand(10, 4)]
        rois = bbox2roi(bboxes_list)
        assert rois.shape == (10, 5)
        assert (rois[:, 0] == 0).all()

    def test_empty(self):
        bboxes_list = [torch.zeros(0, 4), torch.rand(5, 4)]
        rois = bbox2roi(bboxes_list)
        assert rois.shape == (5, 5)
        assert (rois[:, 0] == 1).all()
