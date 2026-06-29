"""插桩分析模块 — 挖掘模型模块内在问题

三个分析器, 对应三个最有价值的白盒插桩点:
1. TrajectoryAnalyzer: 扩散采样轨迹 (box 演化, cls 收敛, x0 质量, renewal 修正)
2. RoIFeatureAnalyzer: RoI 特征区分度 (同组相似度, 尺度范数, 误差可分性)
3. HeadOutputAnalyzer: cls/reg 头输出分布 (logits 分布, delta 幅度, 梯度比例, 困难样本)
"""

from .trajectory_analyzer import TrajectoryAnalyzer, TrajectoryCollector
from .feature_analyzer import RoIFeatureAnalyzer, RoIFeatureCollector
from .head_analyzer import HeadOutputAnalyzer, HeadOutputCollector

__all__ = [
    'TrajectoryAnalyzer', 'TrajectoryCollector',
    'RoIFeatureAnalyzer', 'RoIFeatureCollector',
    'HeadOutputAnalyzer', 'HeadOutputCollector',
]
