"""插桩分析模块测试 — 红绿重构 Red 阶段

验证 3 个插桩分析脚本能准确反映其分析目的:
1. TrajectoryAnalyzer: 扩散采样轨迹 (每步 box 演化, cls 收敛, box_renewal 修正)
2. RoIFeatureAnalyzer: RoI 特征区分度 (同组类别相似度, 尺度范数, 误差类型可分性)
3. HeadOutputAnalyzer: cls/reg 头输出分布 (logits 分布, delta 幅度, 梯度比例)
"""

import pytest
import torch

from experiments.analysis.instrumentation.trajectory_analyzer import (
    TrajectoryAnalyzer,
    TrajectoryCollector,
)
from experiments.analysis.instrumentation.feature_analyzer import (
    RoIFeatureAnalyzer,
    RoIFeatureCollector,
)
from experiments.analysis.instrumentation.head_analyzer import (
    HeadOutputAnalyzer,
    HeadOutputCollector,
)


# ──────────────────────────────────────────────
# 1. 扩散采样轨迹分析
# ──────────────────────────────────────────────

class TestTrajectoryCollector:
    """采集扩散采样每步的中间状态"""

    def test_collect_returns_per_step(self):
        """应返回每个采样步的 cls_logits 和 pred_bboxes"""
        collector = TrajectoryCollector()
        # 模拟 3 个采样步
        for step in range(3):
            collector.record(
                step_idx=step,
                t_curr=1.0 - step * 0.3,
                t_next=0.7 - step * 0.3,
                cls_logits=torch.randn(2, 10, 24),
                pred_bboxes=torch.randn(2, 10, 4),
                x0_raw=torch.randn(2, 10, 4),
                x_raw_before_step=torch.randn(2, 10, 4),
                x_raw_after_step=torch.randn(2, 10, 4),
            )
        traj = collector.trajectory
        assert len(traj) == 3, "应记录 3 个采样步"
        step0 = traj[0]
        assert 'step_idx' in step0
        assert 't_curr' in step0
        assert 'cls_logits' in step0
        assert 'pred_bboxes' in step0
        assert 'x0_raw' in step0
        assert 'x_raw_before_step' in step0
        assert 'x_raw_after_step' in step0

    def test_collect_records_renewal_delta(self):
        """应记录 box_renewal 的修正量 (若提供)"""
        collector = TrajectoryCollector()
        collector.record(
            step_idx=0, t_curr=1.0, t_next=0.7,
            cls_logits=torch.randn(2, 10, 24),
            pred_bboxes=torch.randn(2, 10, 4),
            x0_raw=torch.randn(2, 10, 4),
            x_raw_before_step=torch.randn(2, 10, 4),
            x_raw_after_step=torch.randn(2, 10, 4),
            x_raw_after_renewal=torch.randn(2, 10, 4),
        )
        assert 'x_raw_after_renewal' in collector.trajectory[0]


class TestTrajectoryAnalyzer:
    """分析扩散采样轨迹, 挖掘扩散过程内在问题"""

    @pytest.fixture
    def trajectory(self):
        """构造模拟轨迹: 3 步, box 逐渐收敛"""
        steps = []
        for step in range(3):
            # pred_bboxes 随步数逐渐接近目标 [0.5, 0.5, 0.2, 0.2]
            target = torch.tensor([0.5, 0.5, 0.2, 0.2])
            noise = torch.randn(2, 10, 4) * (0.3 - step * 0.1)
            steps.append({
                'step_idx': step,
                't_curr': 1.0 - step * 0.3,
                't_next': 0.7 - step * 0.3,
                'cls_logits': torch.randn(2, 10, 24) * (1.0 - step * 0.2),
                'pred_bboxes': target + noise,
                'x0_raw': target + noise * 0.5,
                'x_raw_before_step': torch.randn(2, 10, 4),
                'x_raw_after_step': torch.randn(2, 10, 4),
            })
        return steps

    def test_box_evolution_analysis(self, trajectory):
        """应分析 box 随采样步的演化 (IoU 变化, 距离变化)"""
        analyzer = TrajectoryAnalyzer(trajectory)
        result = analyzer.analyze_box_evolution()
        assert 'per_step_iou_to_final' in result, "应有每步 IoU 到最终预测"
        assert 'per_step_l1_to_final' in result, "应有每步 L1 到最终预测"
        assert 'convergence_step' in result, "应识别收敛步 (IoU 不再显著变化)"
        assert len(result['per_step_iou_to_final']) == 3

    def test_cls_convergence_analysis(self, trajectory):
        """应分析 cls_logits 随采样步的收敛性"""
        analyzer = TrajectoryAnalyzer(trajectory)
        result = analyzer.analyze_cls_convergence()
        assert 'per_step_entropy' in result, "应有每步熵 (置信度变化)"
        assert 'per_step_top1_top2_gap' in result, "应有每步 top1-top2 间距 (分类确定性)"
        assert 'is_converged' in result, "应判断是否收敛"
        assert len(result['per_step_entropy']) == 3

    def test_x0_prediction_quality(self, trajectory):
        """应分析 x0 预测质量 (x0_raw vs 最终 pred_bboxes)"""
        analyzer = TrajectoryAnalyzer(trajectory)
        result = analyzer.analyze_x0_quality()
        assert 'per_step_x0_to_final_iou' in result, "应有每步 x0 到最终 IoU"
        assert 'x0_stability' in result, "应有 x0 稳定性 (步间方差)"
        assert 'early_x0_quality' in result, "应有早期 x0 质量 (前几步)"

    def test_renewal_analysis(self, trajectory):
        """应分析 box_renewal 的修正幅度"""
        # 添加 renewal 信息
        for s in trajectory:
            s['x_raw_after_renewal'] = s['x_raw_after_step'] + torch.randn(2, 10, 4) * 0.01
        analyzer = TrajectoryAnalyzer(trajectory)
        result = analyzer.analyze_renewal()
        assert 'per_step_renewal_delta' in result, "应有每步 renewal 修正幅度"
        assert 'renewal_effective' in result, "应判断 renewal 是否有效 (修正量显著)"
        assert len(result['per_step_renewal_delta']) == 3

    def test_full_report(self, trajectory):
        """应生成完整报告"""
        analyzer = TrajectoryAnalyzer(trajectory)
        report = analyzer.full_report()
        assert 'box_evolution' in report
        assert 'cls_convergence' in report
        assert 'x0_quality' in report
        assert 'renewal' in report
        assert 'analysis' in report, "应有文字分析结论"


# ──────────────────────────────────────────────
# 2. RoI 特征区分度分析
# ──────────────────────────────────────────────

class TestRoIFeatureCollector:
    """采集 RoI 特征及对应标签"""

    def test_collect_features_and_labels(self):
        """应采集 RoI 特征和对应类别标签"""
        collector = RoIFeatureCollector()
        collector.record(
            roi_features=torch.randn(10, 256),
            labels=torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]),
            scales=torch.tensor([100.0, 200.0, 500.0, 1000.0, 2000.0,
                                  100.0, 200.0, 500.0, 1000.0, 2000.0]),
        )
        assert collector.features.shape == (10, 256)
        assert collector.labels.shape == (10,)
        assert collector.scales.shape == (10,)

    def test_collect_with_error_type(self):
        """应支持采集误差类型标签 (TP/Cls/Loc/Both) 用于特征可分性分析"""
        collector = RoIFeatureCollector()
        collector.record(
            roi_features=torch.randn(10, 256),
            labels=torch.zeros(10, dtype=torch.long),
            scales=torch.ones(10) * 100,
            error_types=['TP'] * 5 + ['Cls'] * 3 + ['Loc'] * 2,
        )
        assert collector.error_types is not None
        assert len(collector.error_types) == 10


class TestRoIFeatureAnalyzer:
    """分析 RoI 特征区分度, 挖掘分类瓶颈根因"""

    @pytest.fixture
    def collector(self):
        """构造模拟特征: 同组类相似, 不同组类可分"""
        torch.manual_seed(42)
        collector = RoIFeatureCollector()
        # 5 个类, 每类 20 样本
        # 类 0,1 同组 (特征接近), 类 2,3,4 各自独立
        base = torch.randn(5, 256)
        base[1] = base[0] + torch.randn(256) * 0.05  # 类 1 接近类 0
        feats, labels, scales = [], [], []
        for cls_idx in range(5):
            for _ in range(20):
                feats.append(base[cls_idx] + torch.randn(256) * 0.1)
                labels.append(cls_idx)
                scales.append(100.0 * (cls_idx + 1))
        collector.record(
            roi_features=torch.stack(feats),
            labels=torch.tensor(labels),
            scales=torch.tensor(scales),
        )
        return collector

    def test_same_group_similarity(self, collector):
        """应分析同组类别 (特征接近的类) 的余弦相似度"""
        analyzer = RoIFeatureAnalyzer(collector)
        result = analyzer.analyze_same_group_similarity(group_mapping={0: [0, 1], 1: [2], 2: [3], 3: [4]})
        assert 'per_group_intra_similarity' in result, "应有组内相似度"
        assert 'cross_group_similarity' in result, "应有跨组相似度 (对照)"
        assert 'same_group_too_similar' in result, "应判断同组是否过于相似"
        # 类 0,1 是同组, 应比跨组更相似
        assert result['per_group_intra_similarity'][0] > result['cross_group_similarity']

    def test_scale_feature_norm(self, collector):
        """应分析不同尺度目标的特征范数"""
        analyzer = RoIFeatureAnalyzer(collector)
        result = analyzer.analyze_scale_feature_norm()
        assert 'per_scale_norm' in result, "应有每尺度区间特征范数"
        assert 'small_vs_large_norm_ratio' in result, "应有小/大目标范数比"
        assert 'scale_affects_feature' in result, "应判断尺度是否影响特征"

    def test_error_type_separability(self, collector):
        """应分析 TP/Cls/Loc 误差类型在特征空间的可分性"""
        # 添加误差类型
        n = len(collector.labels)
        collector.error_types = ['TP'] * (n // 2) + ['Cls'] * (n // 4) + ['Loc'] * (n - n // 2 - n // 4)
        analyzer = RoIFeatureAnalyzer(collector)
        result = analyzer.analyze_error_type_separability()
        assert 'cls_error_cluster_center' in result, "应有 Cls 误差特征中心"
        assert 'tp_cluster_center' in result, "应有 TP 特征中心"
        assert 'cls_tp_separation' in result, "应有 Cls-TP 在特征空间的分离度"
        assert 'is_separable' in result, "应判断误差类型是否可分"

    def test_full_report(self, collector):
        """应生成完整报告"""
        analyzer = RoIFeatureAnalyzer(collector)
        report = analyzer.full_report(group_mapping={0: [0, 1], 1: [2], 2: [3], 3: [4]})
        assert 'same_group_similarity' in report
        assert 'scale_feature_norm' in report
        assert 'analysis' in report


# ──────────────────────────────────────────────
# 3. cls/reg 头输出分布分析
# ──────────────────────────────────────────────

class TestHeadOutputCollector:
    """采集 cls_head 和 reg_head 的输入输出"""

    def test_collect_cls_and_reg(self):
        """应采集 cls_head 和 reg_head 的输入特征和输出"""
        collector = HeadOutputCollector()
        collector.record(
            fc_feature=torch.randn(10, 256),
            cls_logits=torch.randn(10, 24),
            reg_deltas=torch.randn(10, 4),
            labels=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        )
        assert collector.fc_features.shape == (10, 256)
        assert collector.cls_logits.shape == (10, 24)
        assert collector.reg_deltas.shape == (10, 4)
        assert collector.labels.shape == (10,)


class TestHeadOutputAnalyzer:
    """分析 cls/reg 头输出分布, 挖掘头内在问题"""

    @pytest.fixture
    def collector(self):
        torch.manual_seed(42)
        collector = HeadOutputCollector()
        n = 100
        # 模拟: 类别 0,1 的 logits 接近 (分类困难)
        cls_logits = torch.randn(n, 24)
        cls_logits[:, 1] = cls_logits[:, 0] + torch.randn(n) * 0.1
        collector.record(
            fc_feature=torch.randn(n, 256),
            cls_logits=cls_logits,
            reg_deltas=torch.randn(n, 4) * 0.5,
            labels=torch.randint(0, 24, (n,)),
        )
        return collector

    def test_cls_logits_distribution(self, collector):
        """应分析 cls_logits 的 per-class 分布"""
        analyzer = HeadOutputAnalyzer(collector)
        result = analyzer.analyze_cls_distribution()
        assert 'per_class_mean_logit' in result, "应有每类平均 logit"
        assert 'per_class_std_logit' in result, "应有每类 logit 标准差"
        assert 'class_collapse' in result, "应检测类别坍塌 (某些类 logit 恒低)"
        assert 'confusing_pairs' in result, "应识别易混淆类对 (logits 高度相关)"
        assert len(result['per_class_mean_logit']) == 24

    def test_reg_delta_distribution(self, collector):
        """应分析 reg_delta 的幅度分布"""
        analyzer = HeadOutputAnalyzer(collector)
        result = analyzer.analyze_reg_distribution()
        assert 'delta_mean' in result, "应有 delta 均值"
        assert 'delta_std' in result, "应有 delta 标准差"
        assert 'per_dim_mean' in result, "应有每维 delta 均值 (4 维)"
        assert 'is_conservative' in result, "应判断回归是否保守 (delta 太小)"
        assert len(result['per_dim_mean']) == 4

    def test_cls_reg_gradient_ratio(self, collector):
        """应分析 cls vs reg 头的梯度比例 (若提供梯度)"""
        # 模拟梯度
        n = len(collector.labels)
        collector.cls_grad_norm = torch.rand(n) * 0.1
        collector.reg_grad_norm = torch.rand(n) * 0.5
        analyzer = HeadOutputAnalyzer(collector)
        result = analyzer.analyze_gradient_ratio()
        assert 'mean_cls_grad' in result
        assert 'mean_reg_grad' in result
        assert 'cls_reg_ratio' in result, "应有 cls/reg 梯度比"
        assert 'dominant_head' in result, "应判断哪个头主导"

    def test_hard_sample_analysis(self, collector):
        """应分析困难样本 (同组混淆) 的 logits 特征"""
        analyzer = HeadOutputAnalyzer(collector)
        result = analyzer.analyze_hard_samples(group_mapping={0: [0, 1], 1: [2]})
        assert 'hard_sample_count' in result
        assert 'hard_sample_confidence' in result, "困难样本置信度 (是否过高)"
        assert 'hard_sample_top2_gap' in result, "困难样本 top1-top2 间距 (是否过小)"
        assert 'overconfident' in result, "应判断困难样本是否过度自信"

    def test_full_report(self, collector):
        """应生成完整报告"""
        analyzer = HeadOutputAnalyzer(collector)
        report = analyzer.full_report(group_mapping={0: [0, 1], 1: [2]})
        assert 'cls_distribution' in report
        assert 'reg_distribution' in report
        assert 'hard_samples' in report
        assert 'analysis' in report
