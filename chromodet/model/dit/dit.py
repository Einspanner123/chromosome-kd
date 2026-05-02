from chromodet.model.dit.blocks import *

class STVAEEncoder3D(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入层：3通道→128通道（时空卷积）
        self.conv_in = nn.Conv3d(3, 128, kernel_size=3, stride=1, padding=1)  # [B, 3, T, H, W] → [B, 128, T, H, W]
        
        # 下采样块堆叠（4层）：每次压缩时空维度（T/2, H/2, W/2），通道数翻倍或保持
        self.down_blocks = nn.ModuleList([
            # 第1层：128→128（时间步T不变，空间H/W压缩1/2）
            STVAEDownBlock3D(in_channels=128, out_channels=128, tempc_kernel_size=3),  # [B, 128, T, H, W] → [B, 128, T, H/2, W/2]
            # 第2层：128→256（时空均压缩1/2）
            STVAEDownBlock3D(in_channels=128, out_channels=256, tempc_kernel_size=3),  # [B, 128, T, H/2, W/2] → [B, 256, T/2, H/4, W/4]
            # 第3层：256→256（时空均压缩1/2）
            STVAEDownBlock3D(in_channels=256, out_channels=256, tempc_kernel_size=3),  # [B, 256, T/2, H/4, W/4] → [B, 256, T/4, H/8, W/8]
            # 第4层：256→512（时空均压缩1/2）
            STVAEDownBlock3D(in_channels=256, out_channels=512, tempc_kernel_size=3)   # [B, 256, T/4, H/8, W/8] → [B, 512, T/8, H/16, W/16]
        ])
        
        # 输出层：512→32（潜在空间均值+方差，各16通道）
        self.conv_out = nn.Conv3d(512, 32, kernel_size=3, stride=1, padding=1)  # [B, 512, T/8, H/16, W/16] → [B, 32, T/8, H/16, W/16]
        # 分割为均值（16通道）和方差（16通道）
        self.split = lambda x: (x[:, :16], x[:, 16:])  # 最终输出：[B, 16, T/8, H/16, W/16]（均值）和同维度方差

    def forward(self, x):
        feat = self.conv_in(x)
        for blk in self.down_blocks:
            feat = blk(feat)
        feat = self.conv_out(feat)
        # assert feat.shape == (2,32,2,16,16), f"{feat.shape=}"
        mean, log_var = self.split(feat)
        latent_z = mean + torch.exp(0.5*log_var) * torch.randn_like(mean)
        return latent_z

class STVAEDecoder3D(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入层：16（潜在通道）→512通道
        self.conv_in = nn.Conv3d(16, 512, kernel_size=3, stride=1, padding=1)  # [B, 16, T/8, H/16, W/16] → [B, 512, T/8, H/16, W/16]
        
        # 上采样块堆叠（4层）：每次恢复时空维度（T×2, H×2, W×2），通道数减半或保持
        self.up_blocks = nn.ModuleList([
            # 第1层：512→256（时空均恢复×2）
            STVAEUpBlock3D(in_channels=512, out_channels=256, tempc_kernel_size=3),  # [B, 512, T/8, H/16, W/16] → [B, 256, T/4, H/8, W/8]
            # 第2层：256→256（时空均恢复×2）
            STVAEUpBlock3D(in_channels=256, out_channels=256, tempc_kernel_size=3),  # [B, 256, T/4, H/8, W/8] → [B, 256, T/2, H/4, W/4]
            # 第3层：256→128（时空均恢复×2）
            STVAEUpBlock3D(in_channels=256, out_channels=128, tempc_kernel_size=3),  # [B, 256, T/2, H/4, W/4] → [B, 128, T, H/2, W/2]
            # 第4层：128→128（空间恢复×2，时间不变）
            STVAEUpBlock3D(in_channels=128, out_channels=128, tempc_kernel_size=3)   # [B, 128, T, H/2, W/2] → [B, 128, T, H, W]
        ])
        
        # 输出层：128→3（恢复RGB通道）
        self.conv_out = nn.Conv3d(128, 3, kernel_size=3, stride=1, padding=1)  # [B, 128, T, H, W] → [B, 3, T, H, W]

    def forward(self, x):
        x = self.conv_in(x)
        for blk in self.up_blocks:
            x = blk (x)

        return x
        
class LaVinDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_size = (2, 2)  # 空间分块大小（H和W方向）
        self.latent_channels = 16  # 潜在特征通道数
        self.hidden_size = 2304  # Transformer隐藏层维度
        
        # 潜在特征→patch嵌入（将空间维度转为序列长度）
        self.proj_in = nn.Conv2d(
            in_channels=16*4,  # 16通道×时间步4（T'=4）
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )  # [B, 16*4, H', W'] → [B, 2304, H'/2, W'/2]

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.reshape(B, C*T, H, W)
        x = self.proj_in(x)
        return x.flatten(2).transpose(1,2)

# # 数据交互示例（输入潜在特征：[2, 16, 4, 64, 64]，T'=4, H'=64, W'=64）
# x = latent_z  # [B=2, C=16, T'=4, H'=64, W'=64]
# # 时间维度合并到通道：[2, 16×4=64, 64, 64]
# x = x.permute(0, 1, 2, 3, 4).reshape(2, 16*4, 64, 64)
# # 空间分块：64×64 → 32×32个patch（每个2×2）
# x = dit.proj_in(x)  # [2, 2304, 32, 32]
# # 转为序列：[B, 序列长度, 隐藏维度]，序列长度=32×32=1024
# x = x.flatten(2).transpose(1, 2)  # [2, 1024, 2304]

if __name__ == "__main__":
    encoder = STVAEEncoder3D()
    dummy_input = torch.randn(2,3,16,256,256)
    out: torch.Tensor = encoder(dummy_input)
    print(out.shape) # (2, 16, 1, 16, 16)
    decoder = STVAEDecoder3D()
    out = decoder(out)
    print(out.shape)
    