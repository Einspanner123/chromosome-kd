"""
KaryoFlow 离散流匹配排列生成器 (核心模块)

MDLM 风格: 随机 mask 排列的部分位置 → 预测被 mask 的值。
直接预测排列 (slot i → detection j)，不使用 Lehmer code。

每个位置的词表统一为 N (= num_slots)，输出是匹配分数 score(slot_i, det_j)。
推理时从全 [MASK] 开始，迭代按置信度 unmask → 最终生成完整排列。

关键设计: 输出 logits 通过 query 与 memory 的点积计算（而非固定 Linear head），
使模型天然学习 "哪个染色体特征最适合放在哪个槽位"。
"""

import torch
import torch.nn as nn
from torch import Tensor

from .constants import NUM_SLOTS
from .modules import AdaLNZeroLayer, SinusoidalPositionEmbeddings


class KaryoFlowModule(nn.Module):
    """离散流匹配排列生成器 (直接排列预测版)

    每个 slot 预测 "哪个检测结果应该放在这个位置"，
    词表大小 = num_slots (统一)。

    Args:
        d_model: 特征维度
        num_layers: Transformer 解码器层数
        nhead: 注意力头数
        dim_feedforward: FFN 中间层维度
        num_slots: 核型图槽位数 (46)
        dropout: dropout 率
    """

    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 6,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        num_slots: int = NUM_SLOTS,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_slots = num_slots
        self.mask_token_id = num_slots  # [MASK] token

        # 排列值嵌入 (0 ~ num_slots-1 + [MASK])
        self.perm_embed = nn.Embedding(num_slots + 1, d_model)

        # 槽位位置嵌入 (编码 Denver 分类先验)
        self.slot_embed = nn.Embedding(num_slots, d_model)

        # 时间嵌入
        time_dim = d_model * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(d_model),
            nn.Linear(d_model, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Transformer 解码器
        self.layers = nn.ModuleList([
            AdaLNZeroLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                time_dim=time_dim,
            )
            for _ in range(num_layers)
        ])

        # 输出头: 直接 Linear 预测每个 slot 的分配概率
        # 不使用点积匹配 (会导致模式坍塌)
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_slots),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # 重新零初始化 AdaLN-Zero 的 scale/shift (但保留 gate 的小正数)
        for layer in self.layers:
            w = layer.adaln_mlp[-1].weight
            b = layer.adaln_mlp[-1].bias
            nn.init.zeros_(w)
            nn.init.zeros_(b)
            # gate (α) bias 初始化为 0.1 (让信息从一开始就能流过)
            d = self.d_model
            with torch.no_grad():
                b[2*d:3*d] = 0.1   # α₁ (self-attn)
                b[5*d:6*d] = 0.1   # α₂ (cross-attn)
                b[8*d:9*d] = 0.1   # α₃ (FFN)

    def forward(
        self,
        chrom_features: Tensor,
        perm_noisy: Tensor,
        t: Tensor,
    ) -> Tensor:
        """
        Args:
            chrom_features: (B, N, d) — 染色体编码特征
            perm_noisy: (B, N) — 带 [MASK] 的排列 (long), 值 ∈ {0..N-1, MASK}
            t: (B,) — masking 比例 [0, 1]

        Returns:
            logits: (B, N, N) — 每个 slot 对每个检测的匹配分数
        """
        B, N = perm_noisy.shape

        # 嵌入排列值
        perm_emb = self.perm_embed(perm_noisy)  # (B, N, d)

        # 槽位位置嵌入
        slot_ids = torch.arange(N, device=perm_noisy.device)
        slot_emb = self.slot_embed(slot_ids).unsqueeze(0)  # (1, N, d)

        # Query = 排列值嵌入 + 槽位位置嵌入
        query = perm_emb + slot_emb  # (B, N, d)

        # Memory = 染色体特征 (不变)
        memory = chrom_features  # (B, N, d)

        # 时间嵌入
        time_emb = self.time_mlp(t)  # (B, time_dim)

        # Transformer 解码
        for layer in self.layers:
            query = layer(query, memory, time_emb)

        # 输出: 直接 Linear head
        logits = self.output_head(query)  # (B, N, N)

        return logits
