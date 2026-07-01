"""特征桥接模块 — 方向六：生成模型感知迁移

将 ChromoGen 生成模型的特征迁移到 LDMDet 检测器。
"""

from ldmdet.feature_bridge.chromogen_extractor import (  # noqa: F401
    ChromoGenFeatureExtractor,
    UNetFeatureConfig,
)
from ldmdet.feature_bridge.chromogen_unet import ChromoGenUNet  # noqa: F401
from ldmdet.feature_bridge.chromogen_vae import ChromoGenVAE  # noqa: F401
from ldmdet.feature_bridge.feature_bridge_module import (  # noqa: F401
    FeatureBridgeModule,
)
from ldmdet.feature_bridge.cross_attn_bridge import (  # noqa: F401
    CrossAttnFeatureBridgeModule,
)
from ldmdet.feature_bridge.simple_gate_bridge import (  # noqa: F401
    SimpleGateFeatureBridgeModule,
)
