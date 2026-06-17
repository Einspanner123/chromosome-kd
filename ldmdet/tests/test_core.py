"""测试 ldmdet.core — DynamicConv, SingleRoIExtractor, SingleDiffusionDetHead, DiffusionDetHead"""

import torch
import pytest
from ldmdet.core.dynamic_conv import DynamicConv
from ldmdet.core.roi_extractor import SingleRoIExtractor
from ldmdet.core.single_head import SingleDiffusionDetHead
from ldmdet.core.head import DiffusionDetHead
from ldmdet.criterion.criterion import DiffusionDetCriterion
from ldmdet.criterion.matcher import DiffusionDetMatcher
from ldmdet.criterion.losses import FocalLoss, L1Loss, GIoULoss
from ldmdet.data.structures import ImageMeta


class TestDynamicConv:
    @pytest.fixture
    def dyn_conv(self):
        return DynamicConv(feat_channels=64, dynamic_dim=32, dynamic_num=2, pooler_resolution=7)

    def test_output_shape(self, dyn_conv):
        proposals = torch.randn(1, 50, 64)
        roi_feats = torch.randn(49, 50, 64)  # 7*7=49
        out = dyn_conv(proposals, roi_feats)
        assert out.shape == (1, 50, 64)

    def test_deterministic(self, dyn_conv):
        proposals = torch.randn(1, 50, 64)
        roi_feats = torch.randn(49, 50, 64)
        out1 = dyn_conv(proposals, roi_feats)
        out2 = dyn_conv(proposals, roi_feats)
        assert torch.allclose(out1, out2, atol=1e-5)

    def test_gradient_flow(self, dyn_conv):
        proposals = torch.randn(1, 50, 64, requires_grad=True)
        roi_feats = torch.randn(49, 50, 64, requires_grad=True)
        out = dyn_conv(proposals, roi_feats)
        out.sum().backward()
        assert proposals.grad is not None
        assert roi_feats.grad is not None

    def test_different_sizes(self, dyn_conv):
        # 不同数量的提案
        proposals = torch.randn(1, 10, 64)
        roi_feats = torch.randn(49, 10, 64)
        out = dyn_conv(proposals, roi_feats)
        assert out.shape == (1, 10, 64)


class TestSingleRoIExtractor:
    @pytest.fixture
    def roi_extractor(self):
        return SingleRoIExtractor(
            roi_layer={'type': 'RoIAlign', 'output_size': 7, 'sampling_ratio': 2, 'aligned': True},
            out_channels=64,
            featmap_strides=[4, 8, 16, 32],
            finest_scale=56,
        )

    def _make_feats(self, bs=1, h=64, w=64, channels=64):
        """生成多尺度特征图"""
        return tuple([
            torch.randn(bs, channels, h // s, w // s)
            for s in [4, 8, 16, 32]
        ])

    def test_output_shape(self, roi_extractor):
        feats = self._make_feats()
        rois = torch.tensor([[0, 10.0, 20.0, 50.0, 80.0], [0, 30.0, 40.0, 100.0, 120.0]])
        out = roi_extractor(feats, rois)
        assert out.shape == (2, 64, 7, 7)

    def test_map_roi_levels(self, roi_extractor):
        rois = torch.tensor([
            [0, 10.0, 20.0, 50.0, 80.0],
            [0, 10.0, 20.0, 500.0, 800.0],
        ])
        levels = roi_extractor.map_roi_levels(rois, num_levels=4)
        assert levels.shape == (2,)
        assert (levels >= 0).all() and (levels < 4).all()
        # 大框应映射到更高层级
        assert levels[1] >= levels[0]

    def test_roi_rescale(self, roi_extractor):
        rois = torch.tensor([[0, 10.0, 20.0, 50.0, 80.0]])
        rescaled = roi_extractor.roi_rescale(rois, scale_factor=2.0)
        # 中心不变，尺寸翻倍
        cx_orig = (10.0 + 50.0) / 2
        cy_orig = (20.0 + 80.0) / 2
        cx_new = (rescaled[0, 1] + rescaled[0, 3]) / 2
        cy_new = (rescaled[0, 2] + rescaled[0, 4]) / 2
        assert abs(cx_new.item() - cx_orig) < 1e-3
        assert abs(cy_new.item() - cy_orig) < 1e-3
        # 宽高翻倍
        w_orig = 50.0 - 10.0
        w_new = rescaled[0, 3] - rescaled[0, 1]
        assert abs(w_new.item() - 2 * w_orig) < 1e-3


class TestSingleDiffusionDetHead:
    @pytest.fixture
    def single_head(self):
        return SingleDiffusionDetHead(
            num_classes=24,
            feat_channels=64,
            dim_feedforward=128,
            num_cls_convs=1,
            num_reg_convs=1,
            num_heads=4,
            pooler_resolution=7,
            dynamic_dim=32,
            dynamic_num=2,
            time_conditioning='scale_shift',
        )

    @pytest.fixture
    def single_head_adaln(self):
        return SingleDiffusionDetHead(
            num_classes=24,
            feat_channels=64,
            dim_feedforward=128,
            num_cls_convs=1,
            num_reg_convs=1,
            num_heads=4,
            pooler_resolution=7,
            dynamic_dim=32,
            dynamic_num=2,
            time_conditioning='adaln_zero',
        )

    def _make_inputs(self, bs=2, num_boxes=10, feat_channels=64):
        features = tuple([
            torch.randn(bs, feat_channels, 64 // s, 64 // s)
            for s in [4, 8, 16, 32]
        ])
        bboxes = torch.rand(bs, num_boxes, 4) * 100
        bboxes[:, :, 2:] += bboxes[:, :, :2]
        time_emb = torch.randn(bs, feat_channels * 4)
        pooler = SingleRoIExtractor(
            roi_layer={'type': 'RoIAlign', 'output_size': 7, 'sampling_ratio': 2, 'aligned': True},
            out_channels=feat_channels,
            featmap_strides=[4, 8, 16, 32],
        )
        return features, bboxes, time_emb, pooler

    def test_forward_scale_shift(self, single_head):
        features, bboxes, time_emb, pooler = self._make_inputs()
        cls_logits, pred_bboxes, proposals = single_head(features, bboxes, None, pooler, time_emb)
        assert cls_logits.shape == (2, 10, 24)
        assert pred_bboxes.shape == (2, 10, 4)
        assert proposals.shape == (1, 20, 64)

    def test_forward_adaln_zero(self, single_head_adaln):
        features, bboxes, time_emb, pooler = self._make_inputs()
        cls_logits, pred_bboxes, proposals = single_head_adaln(features, bboxes, None, pooler, time_emb)
        assert cls_logits.shape == (2, 10, 24)
        assert pred_bboxes.shape == (2, 10, 4)

    def test_deterministic(self, single_head):
        single_head.eval()
        features, bboxes, time_emb, pooler = self._make_inputs()
        with torch.no_grad():
            out1 = single_head(features, bboxes, None, pooler, time_emb)
            out2 = single_head(features, bboxes, None, pooler, time_emb)
        assert torch.allclose(out1[0], out2[0], atol=1e-5)
        assert torch.allclose(out1[1], out2[1], atol=1e-5)

    def test_gradient_flow(self, single_head):
        features, bboxes, time_emb, pooler = self._make_inputs()
        cls_logits, pred_bboxes, _ = single_head(features, bboxes, None, pooler, time_emb)
        loss = cls_logits.sum() + pred_bboxes.sum()
        loss.backward()
        for p in single_head.parameters():
            if p.requires_grad:
                assert p.grad is not None


class TestDiffusionDetHead:
    @pytest.fixture
    def head(self):
        single_head = SingleDiffusionDetHead(
            num_classes=24,
            feat_channels=64,
            dim_feedforward=128,
            num_cls_convs=1,
            num_reg_convs=1,
            num_heads=4,
            pooler_resolution=7,
            dynamic_dim=32,
            dynamic_num=2,
            time_conditioning='scale_shift',
        )
        roi_extractor = SingleRoIExtractor(
            roi_layer={'type': 'RoIAlign', 'output_size': 7, 'sampling_ratio': 2, 'aligned': True},
            out_channels=64,
            featmap_strides=[4, 8, 16, 32],
        )
        matcher = DiffusionDetMatcher(
            cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, candidate_topk=5,
        )
        criterion = DiffusionDetCriterion(
            num_classes=24,
            matcher=matcher,
            loss_cls=FocalLoss(loss_weight=2.0),
            loss_bbox=L1Loss(loss_weight=5.0),
            loss_giou=GIoULoss(loss_weight=2.0),
            deep_supervision=True,
        )
        return DiffusionDetHead(
            num_classes=24,
            feat_channels=64,
            num_proposals=50,
            num_heads=3,
            snr_scale=2.0,
            timesteps=1000,
            sampling_timesteps=1,
            solver_type='euler',
            diffusion_type='rectified_flow',
            rf_schedule='shifted',
            rf_shift=2.0,
            single_head=single_head,
            roi_extractor=roi_extractor,
            criterion=criterion,
            deep_supervision=True,
            use_nms=True,
            nms_thr=0.5,
            score_thr=0.05,
        )

    def _make_features(self, bs=2, channels=64):
        return tuple([
            torch.randn(bs, channels, 64 // s, 64 // s)
            for s in [4, 8, 16, 32]
        ])

    def _make_img_metas(self, bs=2):
        return [ImageMeta(img_shape=(256, 256)) for _ in range(bs)]

    def test_forward_shape(self, head):
        features = self._make_features()
        bboxes = torch.rand(2, 50, 4) * 200
        bboxes[:, :, 2:] += bboxes[:, :, :2]
        t = torch.full((2,), 500.0)
        all_cls, all_bbox, proposals = head(features, bboxes, t)
        assert all_cls.shape[0] == 3  # num_heads
        assert all_cls.shape[1] == 2  # bs
        assert all_cls.shape[2] == 50  # num_proposals
        assert all_bbox.shape == (3, 2, 50, 4)

    def test_loss(self, head):
        features = self._make_features()
        img_metas = self._make_img_metas()
        gt_bboxes = [torch.rand(5, 4) * 200 for _ in range(2)]
        for bb in gt_bboxes:
            bb[:, 2:] += bb[:, :2]
        gt_labels = [torch.randint(0, 24, (5,)) for _ in range(2)]
        losses = head.loss(features, img_metas, gt_bboxes, gt_labels)
        assert 'loss_cls' in losses
        assert 'loss_bbox' in losses
        assert 'loss_giou' in losses
        for v in losses.values():
            assert torch.isfinite(v)

    def test_predict(self, head):
        features = self._make_features()
        img_metas = self._make_img_metas()
        results = head.predict(features, img_metas, rescale=False)
        assert len(results) == 2  # batch size

    def test_no_deep_supervision(self):
        single_head = SingleDiffusionDetHead(
            num_classes=24, feat_channels=64, dim_feedforward=128,
            num_cls_convs=1, num_reg_convs=1, num_heads=4,
            pooler_resolution=7, dynamic_dim=32, dynamic_num=2,
        )
        roi_extractor = SingleRoIExtractor(
            roi_layer={'type': 'RoIAlign', 'output_size': 7, 'sampling_ratio': 2, 'aligned': True},
            out_channels=64, featmap_strides=[4, 8, 16, 32],
        )
        head = DiffusionDetHead(
            num_classes=24, feat_channels=64, num_proposals=50, num_heads=3,
            snr_scale=2.0, timesteps=1000, sampling_timesteps=1, solver_type='euler',
            diffusion_type='rectified_flow', rf_schedule='shifted', rf_shift=2.0,
            single_head=single_head, roi_extractor=roi_extractor,
            deep_supervision=False, use_nms=True, nms_thr=0.5, score_thr=0.05,
        )
        features = self._make_features()
        bboxes = torch.rand(2, 50, 4) * 200
        bboxes[:, :, 2:] += bboxes[:, :, :2]
        t = torch.full((2,), 500.0)
        all_cls, all_bbox, proposals = head(features, bboxes, t)
        assert all_cls.shape[0] == 1  # 无 deep supervision 只返回最后一层


class TestDiffusionDetHeadDDPM:
    """测试 DiffusionDetHead 的 DDPM 路径"""

    @pytest.fixture
    def ddpm_head(self):
        single_head = SingleDiffusionDetHead(
            num_classes=24, feat_channels=64, dim_feedforward=128,
            num_cls_convs=1, num_reg_convs=1, num_heads=4,
            pooler_resolution=7, dynamic_dim=32, dynamic_num=2,
            time_conditioning='scale_shift',
        )
        roi_extractor = SingleRoIExtractor(
            roi_layer={'type': 'RoIAlign', 'output_size': 7, 'sampling_ratio': 2, 'aligned': True},
            out_channels=64, featmap_strides=[4, 8, 16, 32],
        )
        matcher = DiffusionDetMatcher(
            cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, candidate_topk=5,
        )
        criterion = DiffusionDetCriterion(
            num_classes=24, matcher=matcher,
            loss_cls=FocalLoss(loss_weight=2.0),
            loss_bbox=L1Loss(loss_weight=5.0),
            loss_giou=GIoULoss(loss_weight=2.0),
            deep_supervision=True,
        )
        return DiffusionDetHead(
            num_classes=24, feat_channels=64, num_proposals=50, num_heads=3,
            snr_scale=2.0, timesteps=1000, sampling_timesteps=4,
            solver_type='euler', diffusion_type='ddpm',
            single_head=single_head, roi_extractor=roi_extractor,
            criterion=criterion, deep_supervision=True,
            use_nms=True, nms_thr=0.5, score_thr=0.05,
        )

    def _make_features(self, bs=2, channels=64):
        return tuple([
            torch.randn(bs, channels, 64 // s, 64 // s)
            for s in [4, 8, 16, 32]
        ])

    def _make_img_metas(self, bs=2):
        return [ImageMeta(img_shape=(256, 256)) for _ in range(bs)]

    def test_ddpm_loss(self, ddpm_head):
        features = self._make_features()
        img_metas = self._make_img_metas()
        gt_bboxes = [torch.rand(5, 4) * 200 for _ in range(2)]
        for bb in gt_bboxes:
            bb[:, 2:] += bb[:, :2]
        gt_labels = [torch.randint(0, 24, (5,)) for _ in range(2)]
        losses = ddpm_head.loss(features, img_metas, gt_bboxes, gt_labels)
        assert 'loss_cls' in losses
        for v in losses.values():
            assert torch.isfinite(v)

    def test_ddpm_predict(self, ddpm_head):
        features = self._make_features()
        img_metas = self._make_img_metas()
        results = ddpm_head.predict(features, img_metas, rescale=False)
        assert len(results) == 2

    def test_ddpm_q_sample(self, ddpm_head):
        """测试 DDPM 前向加噪 q_sample"""
        x_start = torch.randn(2, 50, 4)
        t = torch.tensor([100, 500])
        x_noisy = ddpm_head.q_sample(x_start, t)
        assert x_noisy.shape == x_start.shape
        assert torch.isfinite(x_noisy).all()


class TestDiffusionDetHeadSolvers:
    """测试 DiffusionDetHead 的 Heun 和 DPM-Solver++ 推理路径"""

    def _make_head(self, solver_type, sampling_timesteps=4):
        single_head = SingleDiffusionDetHead(
            num_classes=24, feat_channels=64, dim_feedforward=128,
            num_cls_convs=1, num_reg_convs=1, num_heads=4,
            pooler_resolution=7, dynamic_dim=32, dynamic_num=2,
        )
        roi_extractor = SingleRoIExtractor(
            roi_layer={'type': 'RoIAlign', 'output_size': 7, 'sampling_ratio': 2, 'aligned': True},
            out_channels=64, featmap_strides=[4, 8, 16, 32],
        )
        return DiffusionDetHead(
            num_classes=24, feat_channels=64, num_proposals=50, num_heads=3,
            snr_scale=2.0, timesteps=1000, sampling_timesteps=sampling_timesteps,
            solver_type=solver_type, diffusion_type='rectified_flow',
            rf_schedule='linear', single_head=single_head,
            roi_extractor=roi_extractor, deep_supervision=True,
            use_nms=True, nms_thr=0.5, score_thr=0.05,
        )

    def _make_inputs(self, bs=2, channels=64):
        features = tuple([
            torch.randn(bs, channels, 64 // s, 64 // s)
            for s in [4, 8, 16, 32]
        ])
        img_metas = [ImageMeta(img_shape=(256, 256)) for _ in range(bs)]
        return features, img_metas

    def test_heun_solver(self):
        head = self._make_head('heun', sampling_timesteps=4)
        features, img_metas = self._make_inputs()
        results = head.predict(features, img_metas, rescale=False)
        assert len(results) == 2

    def test_dpm_solver_pp(self):
        head = self._make_head('dpm_solver_pp', sampling_timesteps=4)
        features, img_metas = self._make_inputs()
        results = head.predict(features, img_metas, rescale=False)
        assert len(results) == 2

    def test_dpm_solver_pp_3(self):
        head = self._make_head('dpm_solver_pp_3', sampling_timesteps=4)
        features, img_metas = self._make_inputs()
        results = head.predict(features, img_metas, rescale=False)
        assert len(results) == 2


class TestDiffusionDetHeadPredictTrajectory:
    """测试 DiffusionDetHead.predict(return_trajectory=True)"""

    def test_return_trajectory(self):
        single_head = SingleDiffusionDetHead(
            num_classes=24, feat_channels=64, dim_feedforward=128,
            num_cls_convs=1, num_reg_convs=1, num_heads=4,
            pooler_resolution=7, dynamic_dim=32, dynamic_num=2,
        )
        roi_extractor = SingleRoIExtractor(
            roi_layer={'type': 'RoIAlign', 'output_size': 7, 'sampling_ratio': 2, 'aligned': True},
            out_channels=64, featmap_strides=[4, 8, 16, 32],
        )
        head = DiffusionDetHead(
            num_classes=24, feat_channels=64, num_proposals=50, num_heads=3,
            snr_scale=2.0, timesteps=1000, sampling_timesteps=4,
            solver_type='euler', diffusion_type='rectified_flow',
            rf_schedule='linear', single_head=single_head,
            roi_extractor=roi_extractor, deep_supervision=True,
            use_nms=True, nms_thr=0.5, score_thr=0.05,
        )
        features = tuple([
            torch.randn(1, 64, 64 // s, 64 // s)
            for s in [4, 8, 16, 32]
        ])
        img_metas = [ImageMeta(img_shape=(256, 256))]
        results, trajectory = head.predict(features, img_metas, rescale=False, return_trajectory=True)
        assert len(results) == 1
        assert len(trajectory) > 0
        for cls_logits, pred_bboxes in trajectory:
            assert cls_logits.shape[0] == 1
            assert pred_bboxes.shape[0] == 1


class TestDiffusionDetHeadOTCoupling:
    """测试 DiffusionDetHead 的 OT coupling 路径"""

    def test_ot_coupling_loss(self):
        from ldmdet.coupling import build_coupling
        single_head = SingleDiffusionDetHead(
            num_classes=24, feat_channels=64, dim_feedforward=128,
            num_cls_convs=1, num_reg_convs=1, num_heads=4,
            pooler_resolution=7, dynamic_dim=32, dynamic_num=2,
        )
        roi_extractor = SingleRoIExtractor(
            roi_layer={'type': 'RoIAlign', 'output_size': 7, 'sampling_ratio': 2, 'aligned': True},
            out_channels=64, featmap_strides=[4, 8, 16, 32],
        )
        matcher = DiffusionDetMatcher(
            cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, candidate_topk=5,
        )
        criterion = DiffusionDetCriterion(
            num_classes=24, matcher=matcher,
            loss_cls=FocalLoss(loss_weight=2.0),
            loss_bbox=L1Loss(loss_weight=5.0),
            loss_giou=GIoULoss(loss_weight=2.0),
            deep_supervision=True,
        )
        coupling = build_coupling('sinkhorn_stochastic', epsilon=5.0, num_iters=10)
        head = DiffusionDetHead(
            num_classes=24, feat_channels=64, num_proposals=50, num_heads=3,
            snr_scale=2.0, timesteps=1000, sampling_timesteps=1,
            solver_type='euler', diffusion_type='rectified_flow',
            rf_schedule='linear', single_head=single_head,
            roi_extractor=roi_extractor, criterion=criterion,
            coupling=coupling, deep_supervision=True,
            use_nms=True, nms_thr=0.5, score_thr=0.05,
        )
        features = tuple([
            torch.randn(2, 64, 64 // s, 64 // s)
            for s in [4, 8, 16, 32]
        ])
        img_metas = [ImageMeta(img_shape=(256, 256)) for _ in range(2)]
        gt_bboxes = [torch.rand(5, 4) * 200 for _ in range(2)]
        for bb in gt_bboxes:
            bb[:, 2:] += bb[:, :2]
        gt_labels = [torch.randint(0, 24, (5,)) for _ in range(2)]
        losses = head.loss(features, img_metas, gt_bboxes, gt_labels)
        for v in losses.values():
            assert torch.isfinite(v)


class TestSingleRoIExtractorEdgeCases:
    """测试 SingleRoIExtractor 边界情况"""

    @pytest.fixture
    def roi_extractor(self):
        return SingleRoIExtractor(
            roi_layer={'type': 'RoIAlign', 'output_size': 7, 'sampling_ratio': 2, 'aligned': True},
            out_channels=64,
            featmap_strides=[4, 8, 16, 32],
            finest_scale=56,
        )

    def _make_feats(self, bs=1, h=64, w=64, channels=64):
        return tuple([
            torch.randn(bs, channels, h // s, w // s)
            for s in [4, 8, 16, 32]
        ])

    def test_single_level_feats(self):
        """单级特征图"""
        roi_extractor = SingleRoIExtractor(
            roi_layer={'type': 'RoIAlign', 'output_size': 7, 'sampling_ratio': 2, 'aligned': True},
            out_channels=64,
            featmap_strides=[4],
        )
        feats = (torch.randn(1, 64, 16, 16),)
        rois = torch.tensor([[0, 10.0, 20.0, 50.0, 80.0]])
        out = roi_extractor(feats, rois)
        assert out.shape == (1, 64, 7, 7)

    def test_roi_scale_factor(self, roi_extractor):
        """roi_scale_factor 参数"""
        feats = self._make_feats()
        rois = torch.tensor([[0, 10.0, 20.0, 50.0, 80.0]])
        out_normal = roi_extractor(feats, rois)
        out_scaled = roi_extractor(feats, rois, roi_scale_factor=2.0)
        assert out_normal.shape == out_scaled.shape

    def test_empty_rois(self, roi_extractor):
        """空 ROI 列表"""
        feats = self._make_feats()
        rois = torch.zeros(0, 5)
        out = roi_extractor(feats, rois)
        assert out.shape == (0, 64, 7, 7)


class TestDiffusionDetHeadAdvancedParams:
    """DiffusionDetHead 高级参数 smoke test — 验证参数传递不报错"""

    def _make_base_components(self):
        single_head = SingleDiffusionDetHead(
            num_classes=24, feat_channels=64, dim_feedforward=128,
            num_cls_convs=1, num_reg_convs=1, num_heads=4,
            pooler_resolution=7, dynamic_dim=32, dynamic_num=2,
        )
        roi_extractor = SingleRoIExtractor(
            roi_layer={'type': 'RoIAlign', 'output_size': 7, 'sampling_ratio': 2, 'aligned': True},
            out_channels=64, featmap_strides=[4, 8, 16, 32],
        )
        matcher = DiffusionDetMatcher(
            cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, candidate_topk=5,
        )
        criterion = DiffusionDetCriterion(
            num_classes=24, matcher=matcher,
            loss_cls=FocalLoss(loss_weight=2.0),
            loss_bbox=L1Loss(loss_weight=5.0),
            loss_giou=GIoULoss(loss_weight=2.0),
            deep_supervision=True,
        )
        return single_head, roi_extractor, criterion

    def _make_features(self, bs=2, channels=64):
        return tuple([
            torch.randn(bs, channels, 64 // s, 64 // s)
            for s in [4, 8, 16, 32]
        ])

    def _make_img_metas(self, bs=2):
        return [ImageMeta(img_shape=(256, 256)) for _ in range(bs)]

    def test_filter_unknown_false(self):
        """filter_unknown=False 不报错"""
        single_head, roi_extractor, criterion = self._make_base_components()
        head = DiffusionDetHead(
            num_classes=24, feat_channels=64, num_proposals=50, num_heads=3,
            snr_scale=2.0, timesteps=1000, sampling_timesteps=1,
            solver_type='euler', diffusion_type='rectified_flow',
            rf_schedule='linear', single_head=single_head,
            roi_extractor=roi_extractor, criterion=criterion,
            filter_unknown=False,
        )
        features = self._make_features()
        img_metas = self._make_img_metas()
        gt_bboxes = [torch.rand(5, 4) * 200 for _ in range(2)]
        for bb in gt_bboxes:
            bb[:, 2:] += bb[:, :2]
        gt_labels = [torch.randint(0, 24, (5,)) for _ in range(2)]
        losses = head.loss(features, img_metas, gt_bboxes, gt_labels)
        for v in losses.values():
            assert torch.isfinite(v)

    def test_gt_reweight_false(self):
        """gt_reweight=False 不报错"""
        single_head, roi_extractor, criterion = self._make_base_components()
        head = DiffusionDetHead(
            num_classes=24, feat_channels=64, num_proposals=50, num_heads=3,
            snr_scale=2.0, timesteps=1000, sampling_timesteps=1,
            solver_type='euler', diffusion_type='rectified_flow',
            rf_schedule='linear', single_head=single_head,
            roi_extractor=roi_extractor, criterion=criterion,
            gt_reweight=False,
        )
        features = self._make_features()
        img_metas = self._make_img_metas()
        gt_bboxes = [torch.rand(5, 4) * 200 for _ in range(2)]
        for bb in gt_bboxes:
            bb[:, 2:] += bb[:, :2]
        gt_labels = [torch.randint(0, 24, (5,)) for _ in range(2)]
        losses = head.loss(features, img_metas, gt_bboxes, gt_labels)
        for v in losses.values():
            assert torch.isfinite(v)

    def test_pre_noise_layer(self):
        """pre_noise_layer 参数传递不报错"""
        single_head, roi_extractor, criterion = self._make_base_components()
        head = DiffusionDetHead(
            num_classes=24, feat_channels=64, num_proposals=50, num_heads=3,
            snr_scale=2.0, timesteps=1000, sampling_timesteps=1,
            solver_type='euler', diffusion_type='rectified_flow',
            rf_schedule='linear', single_head=single_head,
            roi_extractor=roi_extractor, criterion=criterion,
            pre_noise_layer=3,
        )
        assert head.pre_noise_layer == 3

    def test_rf_schedule_power(self):
        """rf_schedule='power' 推理路径不报错"""
        single_head, roi_extractor, _ = self._make_base_components()
        head = DiffusionDetHead(
            num_classes=24, feat_channels=64, num_proposals=50, num_heads=3,
            snr_scale=2.0, timesteps=1000, sampling_timesteps=4,
            solver_type='euler', diffusion_type='rectified_flow',
            rf_schedule='power', rf_power=2.0,
            single_head=single_head, roi_extractor=roi_extractor,
            use_nms=True, nms_thr=0.5, score_thr=0.05,
        )
        features = self._make_features()
        img_metas = self._make_img_metas()
        results = head.predict(features, img_metas, rescale=False)
        assert len(results) == 2

    def test_loss_aux_dict(self):
        """loss_aux 参数传递不报错"""
        single_head, roi_extractor, criterion = self._make_base_components()
        head = DiffusionDetHead(
            num_classes=24, feat_channels=64, num_proposals=50, num_heads=3,
            snr_scale=2.0, timesteps=1000, sampling_timesteps=1,
            solver_type='euler', diffusion_type='rectified_flow',
            rf_schedule='linear', single_head=single_head,
            roi_extractor=roi_extractor, criterion=criterion,
            loss_aux={'counting': 1.0},
        )
        assert head.loss_aux == {'counting': 1.0}

    def test_box_renewal_false(self):
        """box_renewal=False 推理路径不报错"""
        single_head, roi_extractor, _ = self._make_base_components()
        head = DiffusionDetHead(
            num_classes=24, feat_channels=64, num_proposals=50, num_heads=3,
            snr_scale=2.0, timesteps=1000, sampling_timesteps=4,
            solver_type='euler', diffusion_type='rectified_flow',
            rf_schedule='linear', single_head=single_head,
            roi_extractor=roi_extractor,
            box_renewal=False,
            use_nms=True, nms_thr=0.5, score_thr=0.05,
        )
        features = self._make_features()
        img_metas = self._make_img_metas()
        results = head.predict(features, img_metas, rescale=False)
        assert len(results) == 2

    def test_use_ensemble_false(self):
        """use_ensemble=False 参数传递正确 (当前 predict 需 use_ensemble=True)"""
        single_head, roi_extractor, _ = self._make_base_components()
        head = DiffusionDetHead(
            num_classes=24, feat_channels=64, num_proposals=50, num_heads=3,
            snr_scale=2.0, timesteps=1000, sampling_timesteps=4,
            solver_type='euler', diffusion_type='rectified_flow',
            rf_schedule='linear', single_head=single_head,
            roi_extractor=roi_extractor,
            use_ensemble=False,
            use_nms=True, nms_thr=0.5, score_thr=0.05,
        )
        # 验证参数传递
        assert head.use_ensemble is False
        assert head._sampler.use_ensemble is False
