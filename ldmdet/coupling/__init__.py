"""耦合策略模块 — 训练时将噪声提案与 GT 框配对。

每种耦合策略是独立的 CouplingStrategy 子类，可插拔、可独立测试。
"""

from ldmdet.coupling.base import CouplingStrategy, build_coupling  # noqa: F401
