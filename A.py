import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """两个 3×3 卷积块 + BN + ReLU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

# --------------------------------------------
# Encoder：4 层 + Bottleneck，带三处残差
# --------------------------------------------
class ResidualUNetEncoder(nn.Module):
    def __init__(self, in_ch=1, feats=[64,128,256,512,1024]):
        super().__init__()
        self.pool = nn.MaxPool2d(2,2)

        self.l1 = ConvBlock(in_ch,      feats[0])  # L1
        self.l2 = ConvBlock(feats[0],   feats[1])  # L2
        self.l3 = ConvBlock(feats[1],   feats[2])  # L3
        self.l4 = ConvBlock(feats[2],   feats[3])  # L4
        self.l5 = ConvBlock(feats[3],   feats[4])  # L5

        # Bottleneck，没有再下采，直接融合
        self.bottleneck = ConvBlock(feats[4] , feats[4])

        # 对齐残差：1→2, 3→4, 1→3
        self.align12 = nn.Conv2d(feats[0], feats[1], 1)
        self.align34 = nn.Conv2d(feats[2], feats[3], 1)
        self.align15 = nn.Conv2d(feats[0], feats[4], 1)
        self.align13 =  nn.Conv2d(feats[2],feats[4], 1)
            
    def forward(self, x):
        # L1
        e1 = self.l1(x)          # [B, f0, H,   W  ]
        p1 = self.pool(e1)       # [B, f0, H/2, W/2]

        # L2 + L1→L2 残差
        e2_raw = self.l2(p1)     # [B, f1, H/2, W/2]
        r12 = self.align12(e1)   # [B, f1, H,   W  ]
        r12 = F.interpolate(r12, size=e2_raw.shape[2:],
                            mode='bilinear', align_corners=False)
        e2   = e2_raw + r12
        p2   = self.pool(e2)     # [B, f1, H/4, W/4]

        # L3
        e3 = self.l3(p2)         # [B, f2, H/4, W/4]
        p3 = self.pool(e3)       # [B, f2, H/8, W/8]

        # L4 + L3→L4 残差
        e4_raw = self.l4(p3)     # [B, f3, H/8, W/8]
        r34    = self.align34(e3)# [B, f3, H/4, W/4]
        r34    = F.interpolate(r34, size=e4_raw.shape[2:],
                              mode='bilinear', align_corners=False)
        e4  = e4_raw + r34
        p5 = self.pool(e4) 
        
        # L5
        e5_raw = self.l5(p5) 
        
       
        r15 = self.align15(e1)   # [B, f2, H,   W  ]
        r15 = F.interpolate(r15, size=e5_raw.shape[2:],
                            mode='bilinear', align_corners=False)
        # 再和 1, 5 相加
        Add15 = e5_raw + r15           # [B, f2, H/4, W/4]
        # 下采到 e4_raw 的空间
        
        r35 = self.align13(e3)
        r15 = F.interpolate(r35, size=r15.shape[2:],
                              mode='bilinear', align_corners=False)
        # 拼接 e4 和 r13_4
        bt_in = Add15+ r15   # [B, f3+f2, H/8, W/8]
        b     = self.bottleneck(bt_in)         # [B, f3,   H/8, W/8]
        
        return [e1,e2,e3,e4], b


# --------------------------------------------
# Decoder：4 层上采样 + 拼接 skip
# --------------------------------------------
class ResidualUNetDecoder(nn.Module):
    def __init__(self, feats=[64,128,256,512,1024]):
        super().__init__()
        C1, C2, C3, C4, C5 = feats  # 64, 128, 256, 512

        # 从 b (通道 C4) 上采到 C4 → 拼接 e4 (C4) → 得到 2*C4 通道
        self.up4  = nn.ConvTranspose2d(C5, C4, 2, stride=2)
        self.dec4 = ConvBlock(2*C4, C4)   # <- 这里 ch_in=2*C4

        # 接下来同理：
        self.up3  = nn.ConvTranspose2d(C4, C3, 2, stride=2)
        self.dec3 = ConvBlock(2*C3, C3)

        self.up2  = nn.ConvTranspose2d(C3, C2, 2, stride=2)
        self.dec2 = ConvBlock(2*C2, C2)

        self.up1  = nn.ConvTranspose2d(C2, C1, 2, stride=2)
        self.dec1 = ConvBlock(2*C1, C1)

        self.final = nn.Conv2d(C1, 1, 1)

    def forward(self, skips, b):
        e1,e2,e3,e4 = skips

        # D4
        d4 = self.up4(b)                        # [B, C4, ...]
        # e4 = F.interpolate(e4, size=d4.shape[2:], mode='bilinear', align_corners=False)
        d4 = self.dec4(torch.cat([d4,e4], dim=1))  # [B, 2*C4, ...] -> [B, C4, ...]

        # D3
        d3 = self.up3(d4)
        # e3 = F.interpolate(e3, size=d3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3,e3], dim=1))  # [B, 2*C3, ...] -> [B, C3, ...]

        # D2
        d2 = self.up2(d3)
        # e2 = F.interpolate(e2, size=d2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2,e2], dim=1))

        # D1
        d1 = self.up1(d2)
        # e1 = F.interpolate(e1, size=d1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1,e1], dim=1))

        return torch.sigmoid(self.final(d1))

# --------------------------------------------
# 完整模型

# --------------------------------------------
class ResidualUNet(nn.Module):
    def __init__(self, in_ch=1, feats=[64,128,256,512,1024]):
        super().__init__()
        self.enc = ResidualUNetEncoder(in_ch, feats)
        self.dec = ResidualUNetDecoder(feats)

    def forward(self, x):
        skips, bottom = self.enc(x)
        
        out = self.dec(skips, bottom)
        print("out",out.shape)
        return out 





class UNet(nn.Module):
    """
    标准 U-Net
    
    Args:
        in_ch:    输入通道数（例如 1）
        out_ch:   输出通道数（分割类别数，比如 1）
        features: 每一层的通道数列表，比如 [64,128,256,512]
    """
    def __init__(self, in_ch=1, out_ch=1, features=[64,128,256,512]):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Encoder ---
        self.enc1 = ConvBlock(in_ch,        features[0])
        self.enc2 = ConvBlock(features[0],  features[1])
        self.enc3 = ConvBlock(features[1],  features[2])
        self.enc4 = ConvBlock(features[2],  features[3])



        # Bottleneck
        self.bottleneck = ConvBlock(features[3], features[3]*2)

        # --- Decoder ---
        self.up4 = nn.ConvTranspose2d(features[3]*2, features[3], kernel_size=2, stride=2)
        self.dec4 = ConvBlock(features[3]*2, features[3])

        self.up3 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.dec3 = ConvBlock(features[2]*2, features[2])

        self.up2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.dec2 = ConvBlock(features[1]*2, features[1])

        self.up1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.dec1 = ConvBlock(features[0]*2, features[0])

        # Final 1×1 卷积输出
        self.final_conv = nn.Conv2d(features[0], out_ch, kernel_size=1)

    def forward(self, x):
        # Encoder 路径
        e1 = self.enc1(x)      # [B, f0, H,   W  ]
        p1 = self.pool(e1)
        
        e2 = self.enc2(p1)     # [B, f1, H/2, W/2]
        p2 = self.pool(e2)
        
        e3 = self.enc3(p2)     # [B, f2, H/4, W/4]
        p3 = self.pool(e3)
        
        e4 = self.enc4(p3)     # [B, f3, H/8, W/8]
        p4 = self.pool(e4)

        # Bottleneck
        b = self.bottleneck(p4)  # [B, f3*2, H/16, W/16]

        # Decoder 路径 + 跳跃连接(concat)
        d4 = self.up4(b)                    # [B, f3, H/8,  W/8 ]
        d4 = torch.cat([d4, e4], dim=1)     # [B, f3*2, H/8,  W/8 ]
        d4 = self.dec4(d4)                  # [B, f3,   H/8,  W/8 ]

        d3 = self.up3(d4)                   # [B, f2,   H/4,  W/4 ]
        d3 = torch.cat([d3, e3], dim=1)     # [B, f2*2, H/4,  W/4 ]
        d3 = self.dec3(d3)                  # [B, f2,   H/4,  W/4 ]

        d2 = self.up2(d3)                   # [B, f1,   H/2,  W/2 ]
        d2 = torch.cat([d2, e2], dim=1)     # [B, f1*2, H/2,  W/2 ]
        d2 = self.dec2(d2)                  # [B, f1,   H/2,  W/2 ]

        d1 = self.up1(d2)                   # [B, f0,   H,    W   ]
        d1 = torch.cat([d1, e1], dim=1)     # [B, f0*2, H,    W   ]
        d1 = self.dec1(d1)                  # [B, f0,   H,    W   ]

        # 输出
        return self.final_conv(d1)          # [B, out_ch, H, W]





# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


class FPNFusion(nn.Module):
    def __init__(self, in_channels=[128, 512, 1024], out_channels=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1) for in_ch in in_channels
        ])
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in in_channels
        ])

    def forward(self, features):
        feats = [lateral(feat) for lateral, feat in zip(self.lateral_convs, features)]
        p3 = feats[2]
        p2 = feats[1] + F.interpolate(p3, size=feats[1].shape[2:], mode='bilinear', align_corners=False)
        p1 = feats[0] + F.interpolate(p2, size=feats[0].shape[2:], mode='bilinear', align_corners=False)
        p3 = self.smooth_convs[2](p3)
        p2 = self.smooth_convs[1](p2)
        p1 = self.smooth_convs[0](p1)
        return [p1, p2, p3]

class FPNUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=[64,128,256,512]):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc1 = ConvBlock(in_ch,        features[0])
        self.enc2 = ConvBlock(features[0],  features[1])
        self.enc3 = ConvBlock(features[1],  features[2])
        self.enc4 = ConvBlock(features[2],  features[3])

        self.bottleneck = ConvBlock(features[3], features[3]*2)

        self.fpn = FPNFusion(in_channels=[features[1], features[3], features[3]*2], out_channels=256)
        self.fpn_reduce = nn.Conv2d(256*3, features[1], kernel_size=1)

        self.up4 = nn.ConvTranspose2d(features[3]*2, features[3], kernel_size=2, stride=2)
        self.dec4 = ConvBlock(features[3]*2, features[3])

        self.up3 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.dec3 = ConvBlock(features[2]*2, features[2])

        self.up2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.dec2 = ConvBlock(features[1]*2, features[1])

        self.up1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.dec1 = ConvBlock(features[0]*2, features[0])

        self.final_conv = nn.Conv2d(features[0], out_ch, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool(e1)

        e2 = self.enc2(p1)
        p2 = self.pool(e2)

        e3 = self.enc3(p2)
        p3 = self.pool(e3)

        e4 = self.enc4(p3)
        p4 = self.pool(e4)

        b = self.bottleneck(p4)

        fpn_feats = self.fpn([e2, e4, b])
        fpn_concat = torch.cat([
            F.interpolate(fpn_feats[1], size=fpn_feats[0].shape[2:], mode='bilinear'),
            F.interpolate(fpn_feats[2], size=fpn_feats[0].shape[2:], mode='bilinear'),
            fpn_feats[0]
        ], dim=1)
        fpn_output = self.fpn_reduce(fpn_concat)

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, fpn_output], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.final_conv(d1)



#######################################################################################################



# --- FPN 融合模块 ---
class FPNFusionV(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1) for in_ch in in_channels
        ])

    def forward(self, feats):
        # feats: List of feature maps at different levels
        return [conv(f) for conv, f in zip(self.lateral_convs, feats)]

# --- 主干网络 ---
class FPNUNetV2(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc1 = ConvBlock(in_ch, features[0])
        self.enc2 = ConvBlock(features[0], features[1])
        self.enc3 = ConvBlock(features[1], features[2])
        self.enc4 = ConvBlock(features[2], features[3])

        self.bottleneck = ConvBlock(features[3], features[3]*2)

        # FPN modules
        self.fpn1 = FPNFusionV(in_channels=[features[0], features[1], features[2]], out_channels=128)
        self.fpn1_reduce = nn.Conv2d(128*3, features[0], kernel_size=1)

        self.fpn2 = FPNFusionV(in_channels=[features[1], features[3], features[3]*2], out_channels=256)
        self.fpn2_reduce = nn.Conv2d(256*3, features[1], kernel_size=1)

        self.up4 = nn.ConvTranspose2d(features[3]*2, features[3], kernel_size=2, stride=2)
        self.dec4 = ConvBlock(features[3]*2, features[3])

        self.up3 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.dec3 = ConvBlock(features[2]*2 + features[0], features[2])

        self.up2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.dec2 = ConvBlock(features[1]*2, features[1])

        self.up1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.dec1 = ConvBlock(features[0]*2, features[0])

        self.final_conv = nn.Conv2d(features[0], out_ch, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool(e1)

        e2 = self.enc2(p1)
        p2 = self.pool(e2)

        e3 = self.enc3(p2)
        p3 = self.pool(e3)

        e4 = self.enc4(p3)
        p4 = self.pool(e4)

        b = self.bottleneck(p4)

        # FPN1: e1, e2, e3
        fpn1_feats = self.fpn1([e1, e2, e3])
        fpn1_concat = torch.cat([
            F.interpolate(fpn1_feats[1], size=fpn1_feats[0].shape[2:], mode='bilinear', align_corners=False),
            F.interpolate(fpn1_feats[2], size=fpn1_feats[0].shape[2:], mode='bilinear', align_corners=False),
            fpn1_feats[0]
        ], dim=1)
        fpn1_output = self.fpn1_reduce(fpn1_concat)

        # FPN2: e2, e4, b
        fpn2_feats = self.fpn2([e2, e4, b])
        fpn2_concat = torch.cat([
            F.interpolate(fpn2_feats[1], size=fpn2_feats[0].shape[2:], mode='bilinear', align_corners=False),
            F.interpolate(fpn2_feats[2], size=fpn2_feats[0].shape[2:], mode='bilinear', align_corners=False),
            fpn2_feats[0]
        ], dim=1)
        fpn2_output = self.fpn2_reduce(fpn2_concat)

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        # 插值对齐
        fpn1_output = F.interpolate(fpn1_output, size=d3.shape[2:], mode='bilinear', align_corners=False)
        e3_resized = F.interpolate(e3, size=d3.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, e3_resized, fpn1_output], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        fpn2_output = F.interpolate(fpn2_output, size=d2.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, fpn2_output], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        e1_resized = F.interpolate(e1, size=d1.shape[2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, e1_resized], dim=1)
        d1 = self.dec1(d1)

        return self.final_conv(d1)






import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------
# Conv Block for Encoder/Decoder
# --------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

# --------------------
# Top-Down FPN Module
# --------------------
class TopDownFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1) for in_ch in in_channels_list
        ])
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in in_channels_list
        ])

    def forward(self, features):
        features = features[::-1]  # start from deepest layer (top-down)
        lateral = [l(f) for l, f in zip(self.lateral_convs, features)]
        fpn_outputs = []
        x = lateral[0]
        fpn_outputs.append(self.smooth_convs[0](x))
        for i in range(1, len(lateral)):
            x = F.interpolate(x, size=lateral[i].shape[2:], mode='nearest') + lateral[i]
            fpn_outputs.append(self.smooth_convs[i](x))
        return fpn_outputs[::-1]  # from shallow to deep

# --------------------
# FPN-UNet V3
# --------------------
class FPNUNetV3(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        self.enc1 = ConvBlock(in_ch, features[0])
        self.enc2 = ConvBlock(features[0], features[1])
        self.enc3 = ConvBlock(features[1], features[2])
        self.enc4 = ConvBlock(features[2], features[3])
        self.bottleneck = ConvBlock(features[3], features[3]*2)

        # FPN for all encoder levels including bottleneck (reverse channel order)
        self.fpn = TopDownFPN(
            in_channels_list=[features[3]*2, features[3], features[2], features[1], features[0]],
            out_channels=128
        )

        # Decoder
        self.up4 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(128 * 2, 128)

        self.up3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(128 * 2, 128)

        self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128 * 2, 128)

        self.up1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128 * 2, 128)

        self.final_conv = nn.Conv2d(128, out_ch, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        # FPN features: [P1, P2, P3, P4, P5]
        fpn_feats = self.fpn([e1, e2, e3, e4, b])

        # Decoder
        d4 = self.up4(fpn_feats[4])  # up from P5
        d4 = torch.cat([d4, fpn_feats[3]], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, fpn_feats[2]], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, fpn_feats[1]], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, fpn_feats[0]], dim=1)
        d1 = self.dec1(d1)

        return self.final_conv(d1)

# # 示例用法：
# model = FPNUNetV3(in_ch=1, out_ch=1)
# x = torch.randn(1, 1, 256, 256)
# out = model(x)
# print(out.shape)  # torch.Size([1, 1, 256, 256])






















from pathlib import Path

# 定义新的带有 Encoder 残差结构的 FPNUNetV3_CBAM 模块代码

import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x))
        max = self.fc(self.max_pool(x))
        out = avg + max
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class TopDownFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1) for in_ch in in_channels_list
        ])
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in in_channels_list
        ])

    def forward(self, features):
        features = features[::-1]
        lateral = [l(f) for l, f in zip(self.lateral_convs, features)]
        fpn_outputs = []
        x = lateral[0]
        fpn_outputs.append(self.smooth_convs[0](x))
        for i in range(1, len(lateral)):
            x = F.interpolate(x, size=lateral[i].shape[2:], mode='nearest') + lateral[i]
            fpn_outputs.append(self.smooth_convs[i](x))
        return fpn_outputs[::-1]

class FPNUNetV3_CBAM_Residual(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc1 = ConvBlock(in_ch, features[0])
        self.enc2 = ConvBlock(features[0], features[1])
        self.enc3 = ConvBlock(features[1], features[2])
        self.enc4 = ConvBlock(features[2], features[3])
        self.bottleneck = ConvBlock(features[3], features[3]*2)

        self.e1_adapter = nn.Conv2d(features[0], features[1], kernel_size=1)
        self.e2_adapter = nn.Conv2d(features[1], features[2], kernel_size=1)

        self.fpn = TopDownFPN(
            in_channels_list=[features[3]*2, features[3], features[2], features[1], features[0]],
            out_channels=128
        )

        self.up4 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.cbam4 = CBAM(128*2 + features[3])
        self.dec4 = ConvBlock(128*2 + features[3], 128)

        self.up3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.cbam3 = CBAM(128*2 + features[2])
        self.dec3 = ConvBlock(128*2 + features[2], 128)

        self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.cbam2 = CBAM(128*2 + features[1])
        self.dec2 = ConvBlock(128*2 + features[1], 128)

        self.up1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.cbam1 = CBAM(128*2 + features[0])
        self.dec1 = ConvBlock(128*2 + features[0], 128)

        self.final_conv = nn.Conv2d(128, out_ch, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))

        e3_input = self.pool(e2)
        e1_resized = F.interpolate(e1, size=e3_input.shape[2:], mode='bilinear', align_corners=False)
        e1_resized = self.e1_adapter(e1_resized)
        e3 = self.enc3(e3_input + e1_resized)

        e4_input = self.pool(e3)
        e2_resized = F.interpolate(e2, size=e4_input.shape[2:], mode='bilinear', align_corners=False)
        e2_resized = self.e2_adapter(e2_resized)
        e4 = self.enc4(e4_input + e2_resized)

        b = self.bottleneck(self.pool(e4))

        fpn_feats = self.fpn([e1, e2, e3, e4, b])

        d4 = self.up4(fpn_feats[4])
        d4 = torch.cat([d4, fpn_feats[3], e4], dim=1)
        d4 = self.cbam4(d4)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, fpn_feats[2], e3], dim=1)
        d3 = self.cbam3(d3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, fpn_feats[1], e2], dim=1)
        d2 = self.cbam2(d2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, fpn_feats[0], e1], dim=1)
        d1 = self.cbam1(d1)
        d1 = self.dec1(d1)

        return self.final_conv(d1)





#========================================================================================

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x))
        max = self.fc(self.max_pool(x))
        out = avg + max
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class TopDownFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1) for in_ch in in_channels_list
        ])
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in in_channels_list
        ])

    def forward(self, features):
        features = features[::-1]
        lateral = [l(f) for l, f in zip(self.lateral_convs, features)]
        fpn_outputs = []
        x = lateral[0]
        fpn_outputs.append(self.smooth_convs[0](x))
        for i in range(1, len(lateral)):
            x = F.interpolate(x, size=lateral[i].shape[2:], mode='nearest') + lateral[i]
            fpn_outputs.append(self.smooth_convs[i](x))
        return fpn_outputs[::-1]

class FPNUNetV3_CBAM_Residual_2(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc1 = ConvBlock(in_ch, features[0])
        self.enc2 = ConvBlock(features[0], features[1])
        self.enc3 = ConvBlock(features[1], features[2])
        self.enc4 = ConvBlock(features[2], features[3])
        self.bottleneck = ConvBlock(features[3], features[3]*2)

        self.e1_adapter = nn.Conv2d(features[0], features[0], kernel_size=1)
        self.e2_adapter = nn.Conv2d(features[1], features[1], kernel_size=1)
        self.fuse3 = nn.Conv2d(features[1] + features[0], features[1], kernel_size=1)
        self.fuse4 = nn.Conv2d(features[2] + features[1], features[2], kernel_size=1)

        self.fpn = TopDownFPN(
            in_channels_list=[features[3]*2, features[3], features[2], features[1], features[0]],
            out_channels=128
        )

        self.up4 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.cbam4 = CBAM(128*2 + features[3])
        self.dec4 = ConvBlock(128*2 + features[3], 128)

        self.up3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.cbam3 = CBAM(128*2 + features[2])
        self.dec3 = ConvBlock(128*2 + features[2], 128)

        self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.cbam2 = CBAM(128*2 + features[1])
        self.dec2 = ConvBlock(128*2 + features[1], 128)

        self.up1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.cbam1 = CBAM(128*2 + features[0])
        self.dec1 = ConvBlock(128*2 + features[0], 128)

        self.final_conv = nn.Conv2d(128, out_ch, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))

        e3_input = self.pool(e2)
        e1_resized = F.interpolate(self.e1_adapter(e1), size=e3_input.shape[2:], mode='bilinear', align_corners=False)
        combined3 = torch.cat([e3_input, e1_resized], dim=1)
        e3 = self.enc3(self.fuse3(combined3))

        e4_input = self.pool(e3)
        e2_resized = F.interpolate(self.e2_adapter(e2), size=e4_input.shape[2:], mode='bilinear', align_corners=False)
        combined4 = torch.cat([e4_input, e2_resized], dim=1)
        e4 = self.enc4(self.fuse4(combined4))

        b = self.bottleneck(self.pool(e4))

        fpn_feats = self.fpn([e1, e2, e3, e4, b])

        d4 = self.up4(fpn_feats[4])
        d4 = torch.cat([d4, fpn_feats[3], e4], dim=1)
        d4 = self.cbam4(d4)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, fpn_feats[2], e3], dim=1)
        d3 = self.cbam3(d3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, fpn_feats[1], e2], dim=1)
        d2 = self.cbam2(d2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, fpn_feats[0], e1], dim=1)
        d1 = self.cbam1(d1)
        d1 = self.dec1(d1)

        return self.final_conv(d1)










#////////////////-----------------------------------------------------------------------------


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg = self.fc(self.avg_pool(x))
        max = self.fc(self.max_pool(x))
        out = avg + max
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)
    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class TopDownFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1) for in_ch in in_channels_list
        ])
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in in_channels_list
        ])
    def forward(self, features):
        features = features[::-1]
        lateral = [l(f) for l, f in zip(self.lateral_convs, features)]
        fpn_outputs = []
        x = lateral[0]
        fpn_outputs.append(self.smooth_convs[0](x))
        for i in range(1, len(lateral)):
            x = F.interpolate(x, size=lateral[i].shape[2:], mode='nearest') + lateral[i]
            fpn_outputs.append(self.smooth_convs[i](x))
        return fpn_outputs[::-1]

class FPNUNetV3_CBAM_AttnGate(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        self.enc1 = ConvBlock(in_ch, features[0])
        self.enc2 = ConvBlock(features[0], features[1])
        self.enc3 = ConvBlock(features[1], features[2])
        self.enc4 = ConvBlock(features[2], features[3])
        self.bottleneck = ConvBlock(features[3], features[3]*2)

        self.fpn = TopDownFPN(
            in_channels_list=[features[3]*2, features[3], features[2], features[1], features[0]],
            out_channels=128
        )

        self.att4 = AttentionGate(128, features[3], 64)
        self.att3 = AttentionGate(128, features[2], 32)
        self.att2 = AttentionGate(128, features[1], 16)
        self.att1 = AttentionGate(128, features[0], 8)

        self.up4 = nn.ConvTranspose2d(128, 128, 2, 2)
        self.cbam4 = CBAM(128 + features[3])
        self.dec4 = ConvBlock(128 + features[3], 128)

        self.up3 = nn.ConvTranspose2d(128, 128, 2, 2)
        self.cbam3 = CBAM(128 + features[2])
        self.dec3 = ConvBlock(128 + features[2], 128)

        self.up2 = nn.ConvTranspose2d(128, 128, 2, 2)
        self.cbam2 = CBAM(128 + features[1])
        self.dec2 = ConvBlock(128 + features[1], 128)

        self.up1 = nn.ConvTranspose2d(128, 128, 2, 2)
        self.cbam1 = CBAM(128 + features[0])
        self.dec1 = ConvBlock(128 + features[0], 128)

        self.final_conv = nn.Conv2d(128, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        fpn_feats = self.fpn([e1, e2, e3, e4, b])

        g4 = self.att4(fpn_feats[3], e4)
        d4 = self.up4(fpn_feats[4])
        d4 = torch.cat([d4, g4], dim=1)
        d4 = self.cbam4(d4)
        d4 = self.dec4(d4)

        g3 = self.att3(fpn_feats[2], e3)
        d3 = self.up3(d4)
        d3 = torch.cat([d3, g3], dim=1)
        d3 = self.cbam3(d3)
        d3 = self.dec3(d3)

        g2 = self.att2(fpn_feats[1], e2)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, g2], dim=1)
        d2 = self.cbam2(d2)
        d2 = self.dec2(d2)

        g1 = self.att1(fpn_feats[0], e1)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, g1], dim=1)
        d1 = self.cbam1(d1)
        d1 = self.dec1(d1)

        return self.final_conv(d1)

# ========================================================================================


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg = self.fc(self.avg_pool(x))
        max = self.fc(self.max_pool(x))
        out = avg + max
        return self.sigmoid(out)

class CBAM_Light(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
    def forward(self, x):
        return x * self.ca(x)

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3)
        )
    def forward(self, x):
        return self.conv(x)

class TopDownFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels_list
        ])
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list
        ])
    def forward(self, features):
        features = features[::-1]
        lateral = [l(f) for l, f in zip(self.lateral_convs, features)]
        fpn_outputs = []
        x = lateral[0]
        fpn_outputs.append(self.smooth_convs[0](x))
        for i in range(1, len(lateral)):
            x = F.interpolate(x, size=lateral[i].shape[2:], mode='nearest') + lateral[i]
            fpn_outputs.append(self.smooth_convs[i](x))
        return fpn_outputs[::-1]

class FPNUNet_Light(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        self.enc1 = ConvBlock(in_ch, features[0])
        self.enc2 = ConvBlock(features[0], features[1])
        self.enc3 = ConvBlock(features[1], features[2])
        self.enc4 = ConvBlock(features[2], features[3])
        self.bottleneck = ConvBlock(features[3], features[3]*2)

        self.fpn = TopDownFPN(
            in_channels_list=[features[3]*2, features[3], features[2], features[1], features[0]],
            out_channels=128
        )

        self.att4 = AttentionGate(128, features[3], 64)
        self.att3 = AttentionGate(128, features[2], 32)

        self.up4 = nn.ConvTranspose2d(128, 128, 2, 2)
        self.cbam4 = CBAM_Light(128 + features[3])
        self.dec4 = ConvBlock(128 + features[3], 128)

        self.up3 = nn.ConvTranspose2d(128, 128, 2, 2)
        self.cbam3 = CBAM_Light(128 + features[2])
        self.dec3 = ConvBlock(128 + features[2], 128)

        self.up2 = nn.ConvTranspose2d(128, 128, 2, 2)
        self.dec2 = ConvBlock(128 + features[1], 128)

        self.up1 = nn.ConvTranspose2d(128, 128, 2, 2)
        self.dec1 = ConvBlock(128 + features[0], 128)

        self.final_conv = nn.Conv2d(128, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        fpn_feats = self.fpn([e1, e2, e3, e4, b])

        g4 = self.att4(fpn_feats[3], e4)
        d4 = self.up4(fpn_feats[4])
        d4 = torch.cat([d4, g4], dim=1)
        d4 = self.cbam4(d4)
        d4 = self.dec4(d4)

        g3 = self.att3(fpn_feats[2], e3)
        d3 = self.up3(d4)
        d3 = torch.cat([d3, g3], dim=1)
        d3 = self.cbam3(d3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.final_conv(d1)



# =========================================


class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.residual = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.residual(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)

class CBAM_Light(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x))
        max = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg + max)

class TopDownFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels_list
        ])
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list
        ])

    def forward(self, features):
        features = features[::-1]
        lateral = [l(f) for l, f in zip(self.lateral_convs, features)]
        fpn_outputs = []
        x = lateral[0]
        fpn_outputs.append(self.smooth_convs[0](x))
        for i in range(1, len(lateral)):
            x = F.interpolate(x, size=lateral[i].shape[2:], mode='nearest') + lateral[i]
            fpn_outputs.append(self.smooth_convs[i](x))
        return fpn_outputs[::-1]

class FPNUNet_Light(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        self.enc1 = ResidualConvBlock(in_ch, features[0])
        self.enc2 = ResidualConvBlock(features[0], features[1])
        self.enc3 = ResidualConvBlock(features[1], features[2])
        self.enc4 = ResidualConvBlock(features[2], features[3])
        self.bottleneck = ResidualConvBlock(features[3], features[3]*2)

        self.e1_adapter = nn.Conv2d(features[0], features[1], 1)
        self.fuse3 = nn.Conv2d(features[1]*2, features[1], 1)
        self.e2_adapter = nn.Conv2d(features[1], features[2], 1)
        self.fuse4 = nn.Conv2d(features[2]*2, features[2], 1)

        self.fpn = TopDownFPN(
            in_channels_list=[features[3]*2, features[3], features[2], features[1], features[0]],
            out_channels=128
        )

        self.cbam4 = CBAM_Light(128 + 128 + features[3])
        self.dec4 = ResidualConvBlock(128 + 128 + features[3], 128)

        self.cbam3 = CBAM_Light(128 + 128 + features[2])
        self.dec3 = ResidualConvBlock(128 + 128 + features[2], 128)

        self.cbam2 = CBAM_Light(128 + 128 + features[1])
        self.dec2 = ResidualConvBlock(128 + 128 + features[1], 128)

        self.cbam1 = CBAM_Light(128 + 128 + features[0])
        self.dec1 = ResidualConvBlock(128 + 128 + features[0], 128)

        self.up4 = nn.ConvTranspose2d(128, 128, 2, 2)
        self.up3 = nn.ConvTranspose2d(128, 128, 2, 2)
        self.up2 = nn.ConvTranspose2d(128, 128, 2, 2)
        self.up1 = nn.ConvTranspose2d(128, 128, 2, 2)

        self.final_conv = nn.Conv2d(128, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))

        e3_input = self.pool(e2)
        e1_resized = F.interpolate(self.e1_adapter(e1), size=e3_input.shape[2:], mode='bilinear', align_corners=False)
        combined3 = torch.cat([e3_input, e1_resized], dim=1)
        e3 = self.enc3(self.fuse3(combined3))

        e4_input = self.pool(e3)
        e2_resized = F.interpolate(self.e2_adapter(e2), size=e4_input.shape[2:], mode='bilinear', align_corners=False)
        combined4 = torch.cat([e4_input, e2_resized], dim=1)
        e4 = self.enc4(self.fuse4(combined4))

        b = self.bottleneck(self.pool(e4))

        fpn_feats = self.fpn([e1, e2, e3, e4, b])

        d4 = self.up4(fpn_feats[4])
        d4 = torch.cat([d4, fpn_feats[3], e4], dim=1)
        d4 = self.cbam4(d4)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, fpn_feats[2], e3], dim=1)
        d3 = self.cbam3(d3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, fpn_feats[1], e2], dim=1)
        d2 = self.cbam2(d2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, fpn_feats[0], e1], dim=1)
        d1 = self.cbam1(d1)
        d1 = self.dec1(d1)

        return self.final_conv(d1)



# =====================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.residual = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.residual(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)

class TopDownFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels_list
        ])
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list
        ])

    def forward(self, features):
        features = features[::-1]
        lateral = [l(f) for l, f in zip(self.lateral_convs, features)]
        fpn_outputs = []
        x = lateral[0]
        fpn_outputs.append(self.smooth_convs[0](x))
        for i in range(1, len(lateral)):
            x = F.interpolate(x, size=lateral[i].shape[2:], mode='nearest') + lateral[i]
            fpn_outputs.append(self.smooth_convs[i](x))
        return fpn_outputs[::-1]

class FPNUNet_LLight(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        self.enc1 = ResidualConvBlock(in_ch, features[0])
        self.enc2 = ResidualConvBlock(features[0], features[1])
        self.enc3 = ResidualConvBlock(features[1], features[2])
        self.enc4 = ResidualConvBlock(features[2], features[3])
        self.bottleneck = ResidualConvBlock(features[3], features[3]*2)

        self.fpn = TopDownFPN(
            in_channels_list=[features[3]*2, features[3], features[2], features[1], features[0]],
            out_channels=128
        )

        self.dec4 = ResidualConvBlock(768, 128)
        self.dec3 = ResidualConvBlock(512, 128)
        self.dec2 = ResidualConvBlock(384, 128)
        self.dec1 = ResidualConvBlock(320, 128)

        self.up4 = nn.ConvTranspose2d(128, 128, 2, 2)
        self.up3 = nn.ConvTranspose2d(128, 128, 2, 2)
        self.up2 = nn.ConvTranspose2d(128, 128, 2, 2)
        self.up1 = nn.ConvTranspose2d(128, 128, 2, 2)

        self.final_conv = nn.Conv2d(128, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        fpn_feats = self.fpn([e1, e2, e3, e4, b])

        d4 = self.up4(fpn_feats[4])
        d4 = torch.cat([d4, fpn_feats[3], e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, fpn_feats[2], e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, fpn_feats[1], e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, fpn_feats[0], e1], dim=1)
        d1 = self.dec1(d1)

        return self.final_conv(d1)


# ==========================================================================



import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.residual = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.residual(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)

class TopDownFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels_list
        ])
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list
        ])

    def forward(self, features):
        features = features[::-1]
        lateral = [l(f) for l, f in zip(self.lateral_convs, features)]
        fpn_outputs = []
        x = lateral[0]
        fpn_outputs.append(self.smooth_convs[0](x))
        for i in range(1, len(lateral)):
            x = F.interpolate(x, size=lateral[i].shape[2:], mode='nearest') + lateral[i]
            fpn_outputs.append(self.smooth_convs[i](x))
        return fpn_outputs[::-1]

class FPNUNet_Lightt(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        self.enc1 = ResidualConvBlock(in_ch, features[0])
        self.enc2 = ResidualConvBlock(features[0], features[1])
        self.enc3 = ResidualConvBlock(features[1], features[2])
        self.enc4 = ResidualConvBlock(features[2], features[3])
        self.bottleneck = ResidualConvBlock(features[3], features[3]*2)

        self.fpn = TopDownFPN(
            in_channels_list=[features[3]*2, features[3], features[2], features[1], features[0]],
            out_channels=128
        )

        self.dec4 = ResidualConvBlock(768, 128)
        self.dec3 = ResidualConvBlock(512, 128)
        self.dec2 = ResidualConvBlock(384, 128)
        self.dec1 = ResidualConvBlock(320, 128)

        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 128, kernel_size=1)
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 128, kernel_size=1)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 128, kernel_size=1)
        )
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 128, kernel_size=1)
        )

        self.final_conv = nn.Conv2d(128, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        fpn_feats = self.fpn([e1, e2, e3, e4, b])

        d4 = self.up4(fpn_feats[4])
        d4 = torch.cat([d4, fpn_feats[3], e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, fpn_feats[2], e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, fpn_feats[1], e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, fpn_feats[0], e1], dim=1)
        d1 = self.dec1(d1)

        return self.final_conv(d1)

# =========================================================



class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.residual = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.residual(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)

class TopDownFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels_list
        ])
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list
        ])

    def forward(self, features):
        features = features[::-1]
        lateral = [l(f) for l, f in zip(self.lateral_convs, features)]
        fpn_outputs = []
        x = lateral[0]
        fpn_outputs.append(self.smooth_convs[0](x))
        for i in range(1, len(lateral)):
            x = F.interpolate(x, size=lateral[i].shape[2:], mode='nearest') + lateral[i]
            fpn_outputs.append(self.smooth_convs[i](x))
        return fpn_outputs[::-1]

class FPNUNet_Liight(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        self.enc1 = ResidualConvBlock(in_ch, features[0])
        self.enc2 = ResidualConvBlock(features[0], features[1])
        self.enc3 = ResidualConvBlock(features[1], features[2])
        self.enc4 = ResidualConvBlock(features[2], features[3])
        self.bottleneck = ResidualConvBlock(features[3], features[3]*2)

        self.fpn = TopDownFPN(
            in_channels_list=[features[3]*2, features[3], features[2], features[1], features[0]],
            out_channels=128
        )

        # channel reducers for encoder features
        self.e4_reducer = nn.Conv2d(512, 64, 1)
        self.e3_reducer = nn.Conv2d(256, 64, 1)
        self.e2_reducer = nn.Conv2d(128, 64, 1)
        self.e1_reducer = nn.Conv2d(64, 64, 1)

        self.dec4 = ResidualConvBlock(128 + 128 + 64, 128)
        self.dec3 = ResidualConvBlock(128 + 128 + 64, 128)
        self.dec2 = ResidualConvBlock(128 + 128 + 64, 128)
        self.dec1 = ResidualConvBlock(128 + 128 + 64, 128)

        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 128, 1)
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 128, 1)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 128, 1)
        )
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 128, 1)
        )

        self.final_conv = nn.Conv2d(128, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        fpn_feats = self.fpn([e1, e2, e3, e4, b])

        d4 = self.up4(fpn_feats[4])
        d4 = torch.cat([d4, fpn_feats[3], self.e4_reducer(e4)], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, fpn_feats[2], self.e3_reducer(e3)], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, fpn_feats[1], self.e2_reducer(e2)], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, fpn_feats[0], self.e1_reducer(e1)], dim=1)
        d1 = self.dec1(d1)

        return self.final_conv(d1)




# ===================================================



class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_se=True, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.residual = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.se = SEBlock(out_ch) if use_se else nn.Identity()

    def forward(self, x):
        identity = self.residual(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return self.relu(out + identity)

class TopDownFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels_list
        ])
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list
        ])

    def forward(self, features):
        features = features[::-1]
        lateral = [l(f) for l, f in zip(self.lateral_convs, features)]
        fpn_outputs = []
        x = lateral[0]
        fpn_outputs.append(self.smooth_convs[0](x))
        for i in range(1, len(lateral)):
            x = F.interpolate(x, size=lateral[i].shape[2:], mode='nearest') + lateral[i]
            fpn_outputs.append(self.smooth_convs[i](x))
        return fpn_outputs[::-1]

class FPNUNet_Lighht(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        self.enc1 = ResidualConvBlock(in_ch, features[0])
        self.enc2 = ResidualConvBlock(features[0], features[1])
        self.enc3 = ResidualConvBlock(features[1], features[2])
        self.enc4 = ResidualConvBlock(features[2], features[3])
        self.bottleneck = ResidualConvBlock(features[3], features[3]*2)

        self.fpn = TopDownFPN(
            in_channels_list=[features[3]*2, features[3], features[2], features[1], features[0]],
            out_channels=128
        )

        self.e4_reducer = nn.Conv2d(features[3], 64, 1)
        self.e3_reducer = nn.Conv2d(features[2], 64, 1)
        self.e2_reducer = nn.Conv2d(features[1], 64, 1)
        self.e1_reducer = nn.Conv2d(features[0], 64, 1)

        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 128, kernel_size=1)
        )
        self.dec4 = ResidualConvBlock(128 + 128 + 64, 128)

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 128, kernel_size=1)
        )
        self.dec3 = ResidualConvBlock(128 + 128 + 64, 128)

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 128, kernel_size=1)
        )
        self.dec2 = ResidualConvBlock(128 + 128 + 64, 128)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 128, kernel_size=1)
        )
        self.dec1 = ResidualConvBlock(128 + 128 + 64, 128)

        self.final_conv = nn.Conv2d(128, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        fpn_feats = self.fpn([e1, e2, e3, e4, b])

        d4 = self.up4(fpn_feats[4])
        d4 = torch.cat([d4, fpn_feats[3], self.e4_reducer(e4)], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, fpn_feats[2], self.e3_reducer(e3)], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, fpn_feats[1], self.e2_reducer(e2)], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, fpn_feats[0], self.e1_reducer(e1)], dim=1)
        d1 = self.dec1(d1)

        return self.final_conv(d1)


# ------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34

class TopDownFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels_list
        ])
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list
        ])

    def forward(self, features):
        features = features[::-1]
        lateral = [l(f) for l, f in zip(self.lateral_convs, features)]
        fpn_outputs = []
        x = lateral[0]
        fpn_outputs.append(self.smooth_convs[0](x))
        for i in range(1, len(lateral)):
            x = F.interpolate(x, size=lateral[i].shape[2:], mode='nearest') + lateral[i]
            fpn_outputs.append(self.smooth_convs[i](x))
        return fpn_outputs[::-1]

class FPNUNet_ResNetEncoder(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        base_model = resnet34(pretrained=False)

        if in_ch != 3:
            self.firstconv = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.firstconv = base_model.conv1

        self.encoder = nn.Sequential(
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4,
        )

        self.fpn = TopDownFPN(
            in_channels_list=[512, 256, 128, 64, 64],
            out_channels=128
        )

        self.e4_reducer = nn.Conv2d(512, 64, 1)
        self.e3_reducer = nn.Conv2d(256, 64, 1)
        self.e2_reducer = nn.Conv2d(128, 64, 1)
        self.e1_reducer = nn.Conv2d(64, 64, 1)
        self.e0_reducer = nn.Conv2d(64, 64, 1)

        self.dec4 = nn.Sequential(
            nn.Conv2d(128 + 128 + 64, 128, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(128 + 128 + 64, 128, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(128 + 128 + 64, 128, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(128 + 128 + 64, 128, 3, padding=1), nn.ReLU(inplace=True)
        )

        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 128, 1)
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 128, 1)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 128, 1)
        )
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 128, 1)
        )

        self.final_conv = nn.Conv2d(128, out_ch, 1)

    def forward(self, x):
        e0 = self.firstconv(x)              # [B, 64, H/2, W/2]
        x = self.encoder[0](e0)             # bn1
        x = self.encoder[1](x)              # relu
        x = self.encoder[2](x)              # maxpool => H/4
        e1 = self.encoder[3](x)             # layer1 => 64
        e2 = self.encoder[4](e1)            # layer2 => 128
        e3 = self.encoder[5](e2)            # layer3 => 256
        e4 = self.encoder[6](e3)            # layer4 => 512

        fpn_feats = self.fpn([e0, e1, e2, e3, e4])

        d4 = self.up4(fpn_feats[4])
        d4 = torch.cat([d4, fpn_feats[3], self.e4_reducer(F.interpolate(e4, size=d4.shape[2:], mode='bilinear', align_corners=True))], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, fpn_feats[2], self.e3_reducer(F.interpolate(e3, size=d3.shape[2:], mode='bilinear', align_corners=True))], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, fpn_feats[1], self.e2_reducer(F.interpolate(e2, size=d2.shape[2:], mode='bilinear', align_corners=True))], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, fpn_feats[0], self.e1_reducer(F.interpolate(e1, size=d1.shape[2:], mode='bilinear', align_corners=True))], dim=1)
        d1 = self.dec1(d1)

        out = self.final_conv(d1)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        return out

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


class TopDownFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels_list
        ])
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list
        ])

    def forward(self, features):
        features = features[::-1]
        lateral = [l(f) for l, f in zip(self.lateral_convs, features)]
        fpn_outputs = []
        x = lateral[0]
        fpn_outputs.append(self.smooth_convs[0](x))
        for i in range(1, len(lateral)):
            x = F.interpolate(x, size=lateral[i].shape[2:], mode='nearest') + lateral[i]
            fpn_outputs.append(self.smooth_convs[i](x))
        return fpn_outputs[::-1]

class FPNUNet_SimpleEncoderFusion(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=[64, 128, 256, 512, 1024]):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, features[0], 3, padding=1), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(nn.Conv2d(features[0], features[1], 3, padding=1), nn.ReLU(inplace=True))
        self.enc3 = nn.Sequential(nn.Conv2d(features[1], features[2], 3, padding=1), nn.ReLU(inplace=True))
        self.enc4 = nn.Sequential(nn.Conv2d(features[2], features[3], 3, padding=1), nn.ReLU(inplace=True))
        self.bottleneck = nn.Sequential(nn.Conv2d(features[3], features[4], 3, padding=1), nn.ReLU(inplace=True))

        self.fpn = TopDownFPN(
            in_channels_list=features[::-1],  # e.g. [1024, 512, 256, 128, 64]
            out_channels=128
        )

        self.e4_reducer = nn.Conv2d(features[3], 64, 1)
        self.e3_reducer = nn.Conv2d(features[2], 64, 1)
        self.e2_reducer = nn.Conv2d(features[1], 64, 1)
        self.e1_reducer = nn.Conv2d(features[0], 64, 1)

        self.dec4 = nn.Sequential(nn.Conv2d(128 + 64, 128, 3, padding=1), nn.ReLU(inplace=True))
        self.dec3 = nn.Sequential(nn.Conv2d(128 + 64, 128, 3, padding=1), nn.ReLU(inplace=True))
        self.dec2 = nn.Sequential(nn.Conv2d(128 + 64, 128, 3, padding=1), nn.ReLU(inplace=True))
        self.dec1 = nn.Sequential(nn.Conv2d(128 + 64, 128, 3, padding=1), nn.ReLU(inplace=True))

        self.up4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(128, 128, 1))
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(128, 128, 1))
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(128, 128, 1))
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(128, 128, 1))

        self.final_conv = nn.Conv2d(128, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)        # -> [B, 64, H, W]
        e2 = self.enc2(self.pool(e1))  # -> [B, 128, H/2, W/2]
        e3 = self.enc3(self.pool(e2))  # -> [B, 256, H/4, W/4]
        e4 = self.enc4(self.pool(e3))  # -> [B, 512, H/8, W/8]
        b = self.bottleneck(self.pool(e4))  # -> [B, 1024, H/16, W/16]

        fpn_feats = self.fpn([e1, e2, e3, e4, b])

        d4 = self.up4(fpn_feats[4])
        d4 = torch.cat([d4, self.e4_reducer(e4)], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, self.e3_reducer(e3)], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, self.e2_reducer(e2)], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, self.e1_reducer(e1)], dim=1)
        d1 = self.dec1(d1)

        return self.final_conv(d1)


# ========================================================================================================


class TopDownFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels_list
        ])
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list
        ])

    def forward(self, features):
        features = features[::-1]
        lateral = [l(f) for l, f in zip(self.lateral_convs, features)]
        fpn_outputs = []
        x = lateral[0]
        fpn_outputs.append(self.smooth_convs[0](x))
        for i in range(1, len(lateral)):
            x = F.interpolate(x, size=lateral[i].shape[2:], mode='nearest') + lateral[i]
            fpn_outputs.append(self.smooth_convs[i](x))
        return fpn_outputs[::-1]

class AttentionGate(nn.Module):
    def __init__(self, in_ch, gating_ch, inter_ch):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(gating_ch, inter_ch, 1, bias=False),
            nn.BatchNorm2d(inter_ch)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(in_ch, inter_ch, 1, bias=False),
            nn.BatchNorm2d(inter_ch)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, 1, bias=True),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class FPNUNet_Simple_EncoderFusion(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=[64, 128, 256, 512, 1024]):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, features[0], 3, padding=1), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(nn.Conv2d(features[0], features[1], 3, padding=1), nn.ReLU(inplace=True))
        self.enc3 = nn.Sequential(nn.Conv2d(features[1], features[2], 3, padding=1), nn.ReLU(inplace=True))
        self.enc4 = nn.Sequential(nn.Conv2d(features[2], features[3], 3, padding=1), nn.ReLU(inplace=True))
        self.bottleneck = nn.Sequential(nn.Conv2d(features[3], features[4], 3, padding=1), nn.ReLU(inplace=True))

        self.fpn = TopDownFPN(in_channels_list=features[::-1], out_channels=128)

        self.att4 = AttentionGate(512, 128, 64)
        self.att3 = AttentionGate(256, 128, 64)
        self.att2 = AttentionGate(128, 128, 64)
        self.att1 = AttentionGate(64, 128, 64)

        self.e4_reduce = nn.Conv2d(512, 64, 1)
        self.e3_reduce = nn.Conv2d(256, 64, 1)
        self.e2_reduce = nn.Conv2d(128, 64, 1)
        self.e1_reduce = nn.Conv2d(64, 64, 1)

        self.dec4 = nn.Sequential(nn.Conv2d(128 + 64, 128, 3, padding=1), nn.ReLU(inplace=True))
        self.dec3 = nn.Sequential(nn.Conv2d(128 + 64, 128, 3, padding=1), nn.ReLU(inplace=True))
        self.dec2 = nn.Sequential(nn.Conv2d(128 + 64, 128, 3, padding=1), nn.ReLU(inplace=True))
        self.dec1 = nn.Sequential(nn.Conv2d(128 + 64, 128, 3, padding=1), nn.ReLU(inplace=True))

        self.up4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(128, 128, 1))
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(128, 128, 1))
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(128, 128, 1))
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(128, 128, 1))

        self.final_conv = nn.Conv2d(128, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        fpn_feats = self.fpn([e1, e2, e3, e4, b])

        d4 = self.up4(fpn_feats[4])
        d4 = torch.cat([d4, self.e4_reduce(self.att4(e4, d4))], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, self.e3_reduce(self.att3(e3, d3))], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, self.e2_reduce(self.att2(e2, d2))], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, self.e1_reduce(self.att1(e1, d1))], dim=1)
        d1 = self.dec1(d1)

        return self.final_conv(d1)


# ------------------------------------------------------------------------------




class TopDownFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels_list
        ])
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list
        ])

    def forward(self, features):
        features = features[::-1]
        lateral = [l(f) for l, f in zip(self.lateral_convs, features)]
        fpn_outputs = []
        x = lateral[0]
        fpn_outputs.append(self.smooth_convs[0](x))
        for i in range(1, len(lateral)):
            x = F.interpolate(x, size=lateral[i].shape[2:], mode='nearest') + lateral[i]
            fpn_outputs.append(self.smooth_convs[i](x))
        return fpn_outputs[::-1]

class AttentionGate(nn.Module):
    def __init__(self, in_ch, gating_ch, inter_ch):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(gating_ch, inter_ch, 1, bias=False),
            nn.BatchNorm2d(inter_ch)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(in_ch, inter_ch, 1, bias=False),
            nn.BatchNorm2d(inter_ch)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, 1, bias=True),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class FPNUNet_Simple__EncoderFusion(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=[64, 128, 256, 512, 1024], dropout_p=0.3):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, features[0], 3, padding=1), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(nn.Conv2d(features[0], features[1], 3, padding=1), nn.ReLU(inplace=True))
        self.enc3 = nn.Sequential(nn.Conv2d(features[1], features[2], 3, padding=1), nn.ReLU(inplace=True))
        self.enc4 = nn.Sequential(nn.Conv2d(features[2], features[3], 3, padding=1), nn.ReLU(inplace=True))
        self.bottleneck = nn.Sequential(nn.Conv2d(features[3], features[4], 3, padding=1), nn.ReLU(inplace=True))

        self.fpn = TopDownFPN(in_channels_list=features[::-1], out_channels=128)

        self.att4 = AttentionGate(512, 128, 64)
        self.att3 = AttentionGate(256, 128, 64)
        self.att2 = AttentionGate(128, 128, 64)
        self.att1 = AttentionGate(64, 128, 64)

        self.e4_reduce = nn.Conv2d(512, 64, 1)
        self.e3_reduce = nn.Conv2d(256, 64, 1)
        self.e2_reduce = nn.Conv2d(128, 64, 1)
        self.e1_reduce = nn.Conv2d(64, 64, 1)

        self.dec4 = nn.Sequential(
            nn.Conv2d(128 + 64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_p)
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(128 + 64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_p)
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(128 + 64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_p)
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(128 + 64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_p)
        )

        self.up4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(128, 128, 1))
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(128, 128, 1))
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(128, 128, 1))
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(128, 128, 1))

        self.final_conv = nn.Conv2d(128, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        fpn_feats = self.fpn([e1, e2, e3, e4, b])

        d4 = self.up4(fpn_feats[4])
        d4 = torch.cat([d4, self.e4_reduce(self.att4(e4, d4))], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, self.e3_reduce(self.att3(e3, d3))], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, self.e2_reduce(self.att2(e2, d2))], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, self.e1_reduce(self.att1(e1, d1))], dim=1)
        d1 = self.dec1(d1)

        return self.final_conv(d1)





# -------------------------------------------------------------------------------




# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class ConvBlock(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_ch)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_ch)
#         self.residual = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

#     def forward(self, x):
#         identity = self.residual(x)
#         out = self.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         return self.relu(out + identity)

# class TopDownFPN(nn.Module):
#     def __init__(self, in_channels_list, out_channels):
#         super().__init__()
#         self.lateral_convs = nn.ModuleList([
#             nn.Conv2d(in_ch, out_channels, kernel_size=1) for in_ch in in_channels_list
#         ])
#         self.smooth_convs = nn.ModuleList([
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in in_channels_list
#         ])

#     def forward(self, features):
#         features = features[::-1]
#         lateral = [l(f) for l, f in zip(self.lateral_convs, features)]
#         fpn_outputs = []
#         x = lateral[0]
#         fpn_outputs.append(self.smooth_convs[0](x))
#         for i in range(1, len(lateral)):
#             x = F.interpolate(x, size=lateral[i].shape[2:], mode='nearest') + lateral[i]
#             fpn_outputs.append(self.smooth_convs[i](x))
#         return fpn_outputs[::-1]

# class FPNUNetV3_CBAM_Residual_2(nn.Module):
#     def __init__(self, in_ch=1, out_ch=1, features=[64, 128, 256, 512]):
#         super().__init__()
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.enc1 = ConvBlock(in_ch, features[0])
#         self.enc2 = ConvBlock(features[0], features[1])
#         self.enc3 = ConvBlock(features[1], features[2])
#         self.enc4 = ConvBlock(features[2], features[3])
#         self.bottleneck = ConvBlock(features[3], features[3]*2)

#         self.e1_adapter = nn.Conv2d(features[0], features[0], kernel_size=1)
#         self.e2_adapter = nn.Conv2d(features[1], features[1], kernel_size=1)
#         self.fuse3 = nn.Conv2d(features[1] + features[0], features[1], kernel_size=1)
#         self.fuse4 = nn.Conv2d(features[2] + features[1], features[2], kernel_size=1)

#         self.fpn = TopDownFPN(
#             in_channels_list=[features[3]*2, features[3], features[2], features[1], features[0]],
#             out_channels=128
#         )

#         self.up4 = nn.ConvTranspose2d(features[3]*2, 128, kernel_size=2, stride=2)
#         self.cbam4 = CBAM(128*2 + features[3])
#         self.dec4 = ConvBlock(128*2 + features[3], 128)

#         self.up3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
#         self.cbam3 = CBAM(128*2 + features[2])
#         self.dec3 = ConvBlock(128*2 + features[2], 128)

#         self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
#         self.cbam2 = CBAM(128*2 + features[1])
#         self.dec2 = ConvBlock(128*2 + features[1], 128)

#         self.up1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
#         self.cbam1 = CBAM(128*2 + features[0])
#         self.dec1 = ConvBlock(128*2 + features[0], 128)

#         self.final_conv = nn.Conv2d(128, out_ch, kernel_size=1)

#     def forward(self, x):
#         e1 = self.enc1(x)
#         e2 = self.enc2(self.pool(e1))

#         e3_input = self.pool(e2)
#         e1_resized = F.interpolate(self.e1_adapter(e1), size=e3_input.shape[2:], mode='bilinear', align_corners=True)
#         combined3 = torch.cat([e3_input, e1_resized], dim=1)
#         e3 = self.enc3(self.fuse3(combined3))

#         e4_input = self.pool(e3)
#         e2_resized = F.interpolate(self.e2_adapter(e2), size=e4_input.shape[2:], mode='bilinear', align_corners=True)
#         combined4 = torch.cat([e4_input, e2_resized], dim=1)
#         e4 = self.enc4(self.fuse4(combined4))

#         b = self.bottleneck(self.pool(e4))

#         fpn_feats = self.fpn([e1, e2, e3, e4, b])

#         d4 = self.up4(b)
#         d4 = torch.cat([d4, fpn_feats[3], e4], dim=1)
#         d4 = self.cbam4(d4)
#         d4 = self.dec4(d4)

#         d3 = self.up3(d4)
#         d3 = torch.cat([d3, fpn_feats[2], e3], dim=1)
#         d3 = self.cbam3(d3)
#         d3 = self.dec3(d3)

#         d2 = self.up2(d3)
#         d2 = torch.cat([d2, fpn_feats[1], e2], dim=1)
#         d2 = self.cbam2(d2)
#         d2 = self.dec2(d2)

#         d1 = self.up1(d2)
#         d1 = torch.cat([d1, fpn_feats[0], e1], dim=1)
#         d1 = self.cbam1(d1)
#         d1 = self.dec1(d1)

#         return self.final_conv(d1)






import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.residual = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.residual(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)

class BottomUpFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1) for in_ch in in_channels_list
        ])
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in in_channels_list
        ])

    def forward(self, features):  # features: [e1, e2, e3, e4, b]
        lateral = [l(f) for l, f in zip(self.lateral_convs, features)]
        x = lateral[0]  # 从e1开始
        fpn_outputs = [self.smooth_convs[0](x)]

        for i in range(1, len(lateral)):
            down = F.max_pool2d(x, kernel_size=2)  # 或使用 stride=2 的 Conv2d
            x = down + lateral[i]
            fpn_outputs.append(self.smooth_convs[i](x))

        return fpn_outputs  # [f1, f2, f3, f4, f5] from shallow to deep

class Del_FPN_F(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc1 = ConvBlock(in_ch, features[0])
        self.enc2 = ConvBlock(features[0], features[1])
        self.enc3 = ConvBlock(features[1], features[2])
        self.enc4 = ConvBlock(features[2], features[3])
        self.bottleneck = ConvBlock(features[3], features[3]*2)

        self.e1_adapter = nn.Conv2d(features[0], features[0], kernel_size=1)
        self.e2_adapter = nn.Conv2d(features[1], features[1], kernel_size=1)
        self.fuse3 = nn.Conv2d(features[1] + features[0], features[1], kernel_size=1)
        self.fuse4 = nn.Conv2d(features[2] + features[1], features[2], kernel_size=1)

        self.fpn = BottomUpFPN(
            in_channels_list=[features[0], features[1], features[2], features[3], features[3]*2],
            out_channels=128
        )

        self.up4 = nn.ConvTranspose2d(features[3]*2, 128, kernel_size=2, stride=2)
        self.cbam4 = CBAM(128*2 + features[3])
        self.dec4 = ConvBlock(128*2 + features[3], 128)

        self.up3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.cbam3 = CBAM(128*2 + features[2])
        self.dec3 = ConvBlock(128*2 + features[2], 128)

        self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.cbam2 = CBAM(128*2 + features[1])
        self.dec2 = ConvBlock(128*2 + features[1], 128)

        self.up1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.cbam1 = CBAM(128*2 + features[0])
        self.dec1 = ConvBlock(128*2 + features[0], 128)

        self.final_conv = nn.Conv2d(128, out_ch, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))

        e3_input = self.pool(e2)
        e1_resized = F.interpolate(self.e1_adapter(e1), size=e3_input.shape[2:], mode='bilinear', align_corners=True)
        combined3 = torch.cat([e3_input, e1_resized], dim=1)
        e3 = self.enc3(self.fuse3(combined3))

        e4_input = self.pool(e3)
        e2_resized = F.interpolate(self.e2_adapter(e2), size=e4_input.shape[2:], mode='bilinear', align_corners=True)
        combined4 = torch.cat([e4_input, e2_resized], dim=1)
        e4 = self.enc4(self.fuse4(combined4))

        b = self.bottleneck(self.pool(e4))

        fpn_feats = self.fpn([e1, e2, e3, e4, b])

        d4 = self.up4(b)
        d4 = torch.cat([d4, fpn_feats[3], e4], dim=1)
        d4 = self.cbam4(d4)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, fpn_feats[2], e3], dim=1)
        d3 = self.cbam3(d3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, fpn_feats[1], e2], dim=1)
        d2 = self.cbam2(d2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, fpn_feats[0], e1], dim=1)
        d1 = self.cbam1(d1)
        d1 = self.dec1(d1)

        return self.final_conv(d1)





import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming CBAM is defined elsewhere, as it was not provided in the original code
# If CBAM is not defined, you would need to include its implementation

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.residual = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.residual(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)

class BottomUpFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1) for in_ch in in_channels_list
        ])
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in in_channels_list
        ])

    def forward(self, features):
        # features: [b, e4, e3, e2, e1] from lowest to highest resolution
        lateral = [l(f) for l, f in zip(self.lateral_convs, features)]
        fpn_outputs = []
        x = lateral[0]  # Start with the lowest resolution feature (bottleneck)
        fpn_outputs.append(self.smooth_convs[0](x))
        for i in range(1, len(lateral)):
            x = F.interpolate(x, size=lateral[i].shape[2:], mode='nearest') + lateral[i]
            fpn_outputs.append(self.smooth_convs[i](x))
        return fpn_outputs[::-1]  # Reverse to match original output order [p1, p2, p3, p4, p5]

class FPNUNetV3_CBAM_Residual_3(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc1 = ConvBlock(in_ch, features[0])
        self.enc2 = ConvBlock(features[0], features[1])
        self.enc3 = ConvBlock(features[1], features[2])
        self.enc4 = ConvBlock(features[2], features[3])
        self.bottleneck = ConvBlock(features[3], features[3]*2)

        self.e1_adapter = nn.Conv2d(features[0], features[0], kernel_size=1)
        self.e2_adapter = nn.Conv2d(features[1], features[1], kernel_size=1)
        self.fuse3 = nn.Conv2d(features[1] + features[0], features[1], kernel_size=1)
        self.fuse4 = nn.Conv2d(features[2] + features[1], features[2], kernel_size=1)

        self.fpn = BottomUpFPN(
            in_channels_list=[features[3]*2, features[3], features[2], features[1], features[0]],
            out_channels=128
        )

        self.up4 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.cbam4 = CBAM(128*2 + features[3])
        self.dec4 = ConvBlock(128*2 + features[3], 128)

        self.up3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.cbam3 = CBAM(128*2 + features[2])
        self.dec3 = ConvBlock(128*2 + features[2], 128)

        self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.cbam2 = CBAM(128*2 + features[1])
        self.dec2 = ConvBlock(128*2 + features[1], 128)

        self.up1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.cbam1 = CBAM(128*2 + features[0])
        self.dec1 = ConvBlock(128*2 + features[0], 128)

        self.final_conv = nn.Conv2d(128, out_ch, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))

        e3_input = self.pool(e2)
        e1_resized = F.interpolate(self.e1_adapter(e1), size=e3_input.shape[2:], mode='bilinear', align_corners=True)
        combined3 = torch.cat([e3_input, e1_resized], dim=1)
        e3 = self.enc3(self.fuse3(combined3))

        e4_input = self.pool(e3)
        e2_resized = F.interpolate(self.e2_adapter(e2), size=e4_input.shape[2:], mode='bilinear', align_corners=True)
        combined4 = torch.cat([e4_input, e2_resized], dim=1)
        e4 = self.enc4(self.fuse4(combined4))

        b = self.bottleneck(self.pool(e4))

        fpn_feats = self.fpn([b, e4, e3, e2, e1])  # Pass features in order: bottleneck to e1

        d4 = self.up4(fpn_feats[4])
        d4 = torch.cat([d4, fpn_feats[3], e4], dim=1)
        d4 = self.cbam4(d4)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, fpn_feats[2], e3], dim=1)
        d3 = self.cbam3(d3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, fpn_feats[1], e2], dim=1)
        d2 = self.cbam2(d2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, fpn_feats[0], e1], dim=1)
        d1 = self.cbam1(d1)
        d1 = self.dec1(d1)

        return self.final_conv(d1)









import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.residual = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.residual(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)

class BottomUpFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1) for in_ch in in_channels_list
        ])
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in in_channels_list
        ])

    def forward(self, features):
        # features: [b, e4, e3, e2, e1] from lowest to highest resolution
        lateral = [l(f) for l, f in zip(self.lateral_convs, features)]
        fpn_outputs = []
        x = lateral[0]  # Start with the lowest resolution feature (bottleneck)
        fpn_outputs.append(self.smooth_convs[0](x))
        for i in range(1, len(lateral)):
            x = F.interpolate(x, size=lateral[i].shape[2:], mode='nearest') + lateral[i]
            fpn_outputs.append(self.smooth_convs[i](x))
        return fpn_outputs[::-1]  # Reverse to match original output order [p1, p2, p3, p4, p5]

class FPNUNetV3_Residual_4(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc1 = ConvBlock(in_ch, features[0])
        self.enc2 = ConvBlock(features[0], features[1])
        self.enc3 = ConvBlock(features[1], features[2])
        self.enc4 = ConvBlock(features[2], features[3])
        self.bottleneck = ConvBlock(features[3], features[3]*2)

        self.e1_adapter = nn.Conv2d(features[0], features[0], kernel_size=1)
        self.e2_adapter = nn.Conv2d(features[1], features[1], kernel_size=1)
        self.fuse3 = nn.Conv2d(features[1] + features[0], features[1], kernel_size=1)
        self.fuse4 = nn.Conv2d(features[2] + features[1], features[2], kernel_size=1)

        self.fpn = BottomUpFPN(
            in_channels_list=[features[3]*2, features[3], features[2], features[1], features[0]],
            out_channels=128
        )

        self.up4 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(128*2 + features[3], 128)

        self.up3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(128*2 + features[2], 128)

        self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128*2 + features[1], 128)

        self.up1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128*2 + features[0], 128)

        self.final_conv = nn.Conv2d(128, out_ch, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))

        e3_input = self.pool(e2)
        e1_resized = F.interpolate(self.e1_adapter(e1), size=e3_input.shape[2:], mode='bilinear', align_corners=True)
        combined3 = torch.cat([e3_input, e1_resized], dim=1)
        e3 = self.enc3(self.fuse3(combined3))

        e4_input = self.pool(e3)
        e2_resized = F.interpolate(self.e2_adapter(e2), size=e4_input.shape[2:], mode='bilinear', align_corners=True)
        combined4 = torch.cat([e4_input, e2_resized], dim=1)
        e4 = self.enc4(self.fuse4(combined4))

        b = self.bottleneck(self.pool(e4))

        fpn_feats = self.fpn([b, e4, e3, e2, e1])  # Pass features in order: bottleneck to e1

        d4 = self.up4(fpn_feats[4])
        d4 = torch.cat([d4, fpn_feats[3], e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, fpn_feats[2], e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, fpn_feats[1], e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, fpn_feats[0], e1], dim=1)
        d1 = self.dec1(d1)

        return self.final_conv(d1)


# ====================================================================







class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x))
        max = self.fc(self.max_pool(x))
        out = avg + max
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.residual = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.residual(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)



class TopDownFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1) for in_ch in in_channels_list
        ])
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in in_channels_list
        ])

    def forward(self, features):
        features = features[::-1]
        lateral = [l(f) for l, f in zip(self.lateral_convs, features)]
        fpn_outputs = []
        x = lateral[0]
        fpn_outputs.append(self.smooth_convs[0](x))
        for i in range(1, len(lateral)):
            x = F.interpolate(x, size=lateral[i].shape[2:], mode='nearest') + lateral[i]
            fpn_outputs.append(self.smooth_convs[i](x))
        return fpn_outputs[::-1]

class FPNUNetV3_CBAM_Residual_8(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc1 = ConvBlock(in_ch, features[0])
        self.enc2 = ConvBlock(features[0], features[1])
        self.enc3 = ConvBlock(features[1], features[2])
        self.enc4 = ConvBlock(features[2], features[3])
        self.bottleneck = ConvBlock(features[3], features[3]*2)

        self.e1_adapter = nn.Conv2d(features[0], features[0], kernel_size=1)
        self.e2_adapter = nn.Conv2d(features[1], features[1], kernel_size=1)
        self.fuse3 = nn.Conv2d(features[1] + features[0], features[1], kernel_size=1)
        self.fuse4 = nn.Conv2d(features[2] + features[1], features[2], kernel_size=1)

        self.fpn = TopDownFPN(
            in_channels_list=[features[3]*2, features[3], features[2], features[1], features[0]],
            out_channels=128
        )

        self.up4 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.cbam4 = CBAM(128*2 + features[3])
        self.dec4 = ConvBlock(128*2 + features[3], 128)

        self.up3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.cbam3 = CBAM(128*2 + features[2])
        self.dec3 = ConvBlock(128*2 + features[2], 128)

        self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.cbam2 = CBAM(128*2 + features[1])
        self.dec2 = ConvBlock(128*2 + features[1], 128)

        self.up1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.cbam1 = CBAM(128*2 + features[0])
        self.dec1 = ConvBlock(128*2 + features[0], 128)

        self.final_conv = nn.Conv2d(128, out_ch, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))

        e3_input = self.pool(e2)
        e1_resized = F.interpolate(self.e1_adapter(e1), size=e3_input.shape[2:], mode='bilinear', align_corners=True)
        combined3 = torch.cat([e3_input, e1_resized], dim=1)
        e3 = self.enc3(self.fuse3(combined3))

        e4_input = self.pool(e3)
        e2_resized = F.interpolate(self.e2_adapter(e2), size=e4_input.shape[2:], mode='bilinear', align_corners=True)
        combined4 = torch.cat([e4_input, e2_resized], dim=1)
        e4 = self.enc4(self.fuse4(combined4))

        b = self.bottleneck(self.pool(e4))

        fpn_feats = self.fpn([e1, e2, e3, e4, b])

        d4 = self.up4(fpn_feats[4])
        d4 = torch.cat([d4, fpn_feats[3], e4], dim=1)
        d4 = self.cbam4(d4)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, fpn_feats[2], e3], dim=1)
        d3 = self.cbam3(d3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, fpn_feats[1], e2], dim=1)
        d2 = self.cbam2(d2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, fpn_feats[0], e1], dim=1)
        d1 = self.cbam1(d1)
        d1 = self.dec1(d1)

        return self.final_conv(d1)
    
    


# ============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x))
        max = self.fc(self.max_pool(x))
        out = avg + max
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.residual = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.residual(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)

class TopDownFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1) for in_ch in in_channels_list
        ])
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in in_channels_list
        ])

    def forward(self, features):
        features = features[::-1]  # Reverse: [e1, e2, e3, e4, b] -> [b, e4, e3, e2, e1]
        lateral = [l(f) for l, f in zip(self.lateral_convs, features)]
        fpn_outputs = []
        x = lateral[0]
        fpn_outputs.append(self.smooth_convs[0](x))
        for i in range(1, len(lateral)):
            x = F.interpolate(x, size=lateral[i].shape[2:], mode='nearest') + lateral[i]
            fpn_outputs.append(self.smooth_convs[i](x))
        return fpn_outputs[::-1]  # Reverse back to original spatial order

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class FPNUNet_Ligh_7(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 编码器
        self.enc1 = ResidualConvBlock(in_ch, features[0])
        self.enc2 = ResidualConvBlock(features[0], features[1])
        self.e1_adapter = nn.Conv2d(features[0], features[1], kernel_size=1)
        self.e1_se = SEBlock(features[1])
        self.e3_se = SEBlock(features[1])
        self.fuse3 = nn.Conv2d(features[1], features[1], kernel_size=1)
        self.enc3 = ResidualConvBlock(features[1], features[2])
        self.e2_adapter = nn.Conv2d(features[1], features[2], kernel_size=1)
        self.e2_se = SEBlock(features[2])
        self.e4_se = SEBlock(features[2])
        self.fuse4 = nn.Conv2d(features[2], features[2], kernel_size=1)
        self.enc4 = ResidualConvBlock(features[2], features[3])
        self.bottleneck = ResidualConvBlock(features[3], features[3]*2)

        # FPN: Reverse in_channels_list to match forward pass order [b, e4, e3, e2, e1]
        self.fpn = TopDownFPN(
            in_channels_list=[features[3]*2, features[3], features[2], features[1], features[0]],  # [1024, 512, 256, 128, 64]
            out_channels=128
        )

        # 解码器
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 128, kernel_size=1)
        )
        self.d4_se = SEBlock(128)
        self.fpn3_se = SEBlock(128)
        self.e4_se_dec = SEBlock(128)
        self.e4_adjust = nn.Conv2d(features[3], 128, kernel_size=1)
        self.cbam4 = CBAM(128)
        self.dec4 = ResidualConvBlock(128, 128)

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 128, kernel_size=1)
        )
        self.d3_se = SEBlock(128)
        self.fpn2_se = SEBlock(128)
        self.e3_se_dec = SEBlock(128)
        self.e3_adjust = nn.Conv2d(features[2], 128, kernel_size=1)
        self.cbam3 = CBAM(128)
        self.dec3 = ResidualConvBlock(128, 128)

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 128, kernel_size=1)
        )
        self.d2_se = SEBlock(128)
        self.fpn1_se = SEBlock(128)
        self.e2_se_dec = SEBlock(128)
        self.cbam2 = CBAM(128)
        self.dec2 = ResidualConvBlock(128, 128)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 128, kernel_size=1)
        )
        self.d1_se = SEBlock(128)
        self.fpn0_se = SEBlock(128)
        self.e1_se_dec = SEBlock(128)
        self.e1_adjust = nn.Conv2d(features[0], 128, kernel_size=1)
        self.cbam1 = CBAM(128)
        self.dec1 = ResidualConvBlock(128, 128)

        self.final_conv = nn.Conv2d(128, out_ch, kernel_size=1)

    def forward(self, x):
        # 编码器
        e1 = self.enc1(x)  # [batch, 64, H, W]
        e2 = self.enc2(self.pool(e1))  # [batch, 128, H/2, W/2]

        e3_input = self.pool(e2)  # [batch, 128, H/4, W/4]
        e1_resized = F.interpolate(self.e1_adapter(e1), size=e3_input.shape[2:], mode='bilinear', align_corners=True)  # [batch, 128, H/4, W/4]
        e3_input_se = self.e3_se(e3_input)
        e1_resized_se = self.e1_se(e1_resized)
        combined3 = self.fuse3(e3_input_se + e1_resized_se)  # [batch, 128, H/4, W/4]
        e3 = self.enc3(combined3)  # [batch, 256, H/4, W/4]

        e4_input = self.pool(e3)  # [batch, 256, H/8, W/8]
        e2_resized = F.interpolate(self.e2_adapter(e2), size=e4_input.shape[2:], mode='bilinear', align_corners=True)  # [batch, 256, H/8, W/8]
        e4_input_se = self.e4_se(e4_input)
        e2_resized_se = self.e2_se(e2_resized)
        combined4 = self.fuse4(e4_input_se + e2_resized_se)  # [batch, 256, H/8, W/8]
        e4 = self.enc4(combined4)  # [batch, 512, H/8, W/8]

        b = self.bottleneck(self.pool(e4))  # [batch, 1024, H/16, W/16]

        # FPN
        fpn_feats = self.fpn([e1, e2, e3, e4, b])  # fpn_feats[i]: [batch, 128, H/2^i, W/2^i]

        # 解码器
        d4 = self.up4(fpn_feats[4])  # [batch, 128, H/8, W/8]
        d4_se = self.d4_se(d4)
        fpn3_se = self.fpn3_se(fpn_feats[3])
        e4_adjusted = self.e4_adjust(e4)  # [batch, 128, H/8, W/8]
        e4_se = self.e4_se_dec(e4_adjusted)
        d4 = d4_se + fpn3_se + e4_se  # [batch, 128, H/8, W/8]
        d4 = self.cbam4(d4)
        d4 = self.dec4(d4)  # [batch, 128, H/8, W/8]

        d3 = self.up3(d4)  # [batch, 128, H/4, W/4]
        d3_se = self.d3_se(d3)
        fpn2_se = self.fpn2_se(fpn_feats[2])
        e3_adjusted = self.e3_adjust(e3)  # [batch, 128, H/4, W/4]
        e3_se = self.e3_se_dec(e3_adjusted)
        d3 = d3_se + fpn2_se + e3_se  # [batch, 128, H/4, W/4]
        d3 = self.cbam3(d3)
        d3 = self.dec3(d3)  # [batch, 128, H/4, W/4]

        d2 = self.up2(d3)  # [batch, 128, H/2, W/2]
        d2_se = self.d2_se(d2)
        fpn1_se = self.fpn1_se(fpn_feats[1])
        e2_se = self.e2_se_dec(e2)  # [batch, 128, H/2, W/2]
        d2 = d2_se + fpn1_se + e2_se  # [batch, 128, H/2, W/2]
        d2 = self.cbam2(d2)
        d2 = self.dec2(d2)  # [batch, 128, H/2, W/2]

        d1 = self.up1(d2)  # [batch, 128, H, W]
        d1_se = self.d1_se(d1)
        fpn0_se = self.fpn0_se(fpn_feats[0])
        e1_adjusted = self.e1_adjust(e1)  # [batch, 128, H, W]
        e1_se = self.e1_se_dec(e1_adjusted)
        d1 = d1_se + fpn0_se + e1_se  # [batch, 128, H, W]
        d1 = self.cbam1(d1)
        d1 = self.dec1(d1)  # [batch, 128, H, W]

        return self.final_conv(d1)  # [batch, 1, H, W]

#/////////////////////////////////////////////////////////////////////////////////////////
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x))
        max = self.fc(self.max_pool(x))
        out = avg + max
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.residual = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.residual(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)

class TTopDownFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1) for in_ch in in_channels_list
        ])
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in in_channels_list
        ])

    def forward(self, features):
        features = features[::-1]
        lateral = [l(f) for l, f in zip(self.lateral_convs, features)]
        fpn_outputs = []
        x = lateral[0]
        fpn_outputs.append(self.smooth_convs[0](x))
        for i in range(1, len(lateral)):
            x = F.interpolate(x, size=lateral[i].shape[2:], mode='nearest') + lateral[i]
            fpn_outputs.append(self.smooth_convs[i](x))
        return fpn_outputs[::-1]

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class FPNUNetV3_CBAM_Residual_7(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 编码器
        self.enc1 = ConvBlock(in_ch, features[0])
        self.enc2 = ConvBlock(features[0], features[1])
        self.enc3 = ConvBlock(features[1], features[2])
        self.enc4 = ConvBlock(features[2], features[3])
        self.bottleneck = ConvBlock(features[3], features[3]*2)

        # 编码器融合适配器
        self.e1_adapter = nn.Conv2d(features[0], features[1], kernel_size=1)
        self.e2_adapter = nn.Conv2d(features[1], features[2], kernel_size=1)
        self.fuse3 = nn.Conv2d(features[1], features[1], kernel_size=1)
        self.fuse4 = nn.Conv2d(features[2], features[2], kernel_size=1)
        self.e3_se = SEBlock(features[1])
        self.e1_se = SEBlock(features[1])
        self.e4_se = SEBlock(features[2])
        self.e2_se = SEBlock(features[2])

        # FPN
        self.fpn = TopDownFPN(
            in_channels_list=[features[3]*2, features[3], features[2], features[1], features[0]],
            out_channels=128
        )

        # 解码器
        self.up4 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.d4_se = SEBlock(128)
        self.fpn3_se = SEBlock(128)
        self.e4_se_dec = SEBlock(128)  # Fixed: Match e4_adjusted channels (128)
        self.e4_adjust = nn.Conv2d(512, 128, kernel_size=1)
        self.cbam4 = CBAM(128)
        self.dec4 = ConvBlock(128, 128)

        self.up3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.d3_se = SEBlock(128)
        self.fpn2_se = SEBlock(128)
        self.e3_se_dec = SEBlock(128)  # Fixed: Match e3_adjusted channels (128)
        self.e3_adjust = nn.Conv2d(256, 128, kernel_size=1)
        self.cbam3 = CBAM(128)
        self.dec3 = ConvBlock(128, 128)

        self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.d2_se = SEBlock(128)
        self.fpn1_se = SEBlock(128)
        self.e2_se_dec = SEBlock(128)  # Correct: Matches e2 channels (128)
        self.cbam2 = CBAM(128)
        self.dec2 = ConvBlock(128, 128)

        self.up1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.d1_se = SEBlock(128)
        self.fpn0_se = SEBlock(128)
        self.e1_se_dec = SEBlock(128)  # Fixed: Match e1_adjusted channels (128)
        self.e1_adjust = nn.Conv2d(64, 128, kernel_size=1)
        self.cbam1 = CBAM(128)
        self.dec1 = ConvBlock(128, 128)

        self.final_conv = nn.Conv2d(128, out_ch, kernel_size=1)

    def forward(self, x):
        # 编码器
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))

        e3_input = self.pool(e2)
        e1_resized = F.interpolate(self.e1_adapter(e1), size=e3_input.shape[2:], mode='bilinear', align_corners=True)
        e3_input_se = self.e3_se(e3_input)
        e1_resized_se = self.e1_se(e1_resized)
        combined3 = self.fuse3(e3_input_se + e1_resized_se)
        e3 = self.enc3(combined3)

        e4_input = self.pool(e3)
        e2_resized = F.interpolate(self.e2_adapter(e2), size=e4_input.shape[2:], mode='bilinear', align_corners=True)
        e4_input_se = self.e4_se(e4_input)
        e2_resized_se = self.e2_se(e2_resized)
        combined4 = self.fuse4(e4_input_se + e2_resized_se)
        e4 = self.enc4(combined4)

        b = self.bottleneck(self.pool(e4))

        # FPN
        fpn_feats = self.fpn([e1, e2, e3, e4, b])

        # 解码器
        d4 = self.up4(fpn_feats[4])
        d4_se = self.d4_se(d4)
        fpn3_se = self.fpn3_se(fpn_feats[3])
        e4_adjusted = self.e4_adjust(e4)
        e4_se = self.e4_se_dec(e4_adjusted)
        d4 = d4_se + fpn3_se + e4_se
        d4 = self.cbam4(d4)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3_se = self.d3_se(d3)
        fpn2_se = self.fpn2_se(fpn_feats[2])
        e3_adjusted = self.e3_adjust(e3)
        e3_se = self.e3_se_dec(e3_adjusted)
        d3 = d3_se + fpn2_se + e3_se
        d3 = self.cbam3(d3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2_se = self.d2_se(d2)
        fpn1_se = self.fpn1_se(fpn_feats[1])
        e2_se = self.e2_se_dec(e2)
        d2 = d2_se + fpn1_se + e2_se
        d2 = self.cbam2(d2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1_se = self.d1_se(d1)
        fpn0_se = self.fpn0_se(fpn_feats[0])
        e1_adjusted = self.e1_adjust(e1)
        e1_se = self.e1_se_dec(e1_adjusted)
        d1 = d1_se + fpn0_se + e1_se
        d1 = self.cbam1(d1)
        d1 = self.dec1(d1)

        return self.final_conv(d1)





import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x))
        max = self.fc(self.max_pool(x))
        out = avg + max
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.residual = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.residual(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)

class TopDownFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1) for in_ch in in_channels_list
        ])
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in in_channels_list
        ])

    def forward(self, features):
        features = features[::-1]
        lateral = [l(f) for l, f in zip(self.lateral_convs, features)]
        fpn_outputs = []
        x = lateral[0]
        fpn_outputs.append(self.smooth_convs[0](x))
        for i in range(1, len(lateral)):
            x = F.interpolate(x, size=lateral[i].shape[2:], mode='nearest') + lateral[i]
            fpn_outputs.append(self.smooth_convs[i](x))
        return fpn_outputs[::-1]

class Del_FPN_F(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc1 = ConvBlock(in_ch, features[0])
        self.enc2 = ConvBlock(features[0], features[1])
        self.enc3 = ConvBlock(features[1], features[2])
        self.enc4 = ConvBlock(features[2], features[3])
        self.bottleneck = ConvBlock(features[3], features[3]*2)

        self.e1_adapter = nn.Conv2d(features[0], features[0], kernel_size=1)
        self.e2_adapter = nn.Conv2d(features[1], features[1], kernel_size=1)
        self.fuse3 = nn.Conv2d(features[1] + features[0], features[1], kernel_size=1)
        self.fuse4 = nn.Conv2d(features[2] + features[1], features[2], kernel_size=1)

        self.fpn = TopDownFPN(
            in_channels_list=[features[3]*2, features[3], features[2], features[1], features[0]],
            out_channels=128
        )

        self.up4 = nn.ConvTranspose2d(features[3]*2, 128, kernel_size=2, stride=2)
        self.cbam4 = CBAM(128*2 + features[3])  # = 128*2 + 512 = 768
        self.dec4 = ConvBlock(128*2 + features[3], 128)

        self.up3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.cbam3 = CBAM(128*2 + features[2])
        self.dec3 = ConvBlock(128*2 + features[2], 128)

        self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.cbam2 = CBAM(128*2 + features[1])
        self.dec2 = ConvBlock(128*2 + features[1], 128)

        self.up1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.cbam1 = CBAM(128*2 + features[0])
        self.dec1 = ConvBlock(128*2 + features[0], 128)

        self.final_conv = nn.Conv2d(128, out_ch, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))

        e3_input = self.pool(e2)
        e1_resized = F.interpolate(self.e1_adapter(e1), size=e3_input.shape[2:], mode='bilinear', align_corners=True)
        combined3 = torch.cat([e3_input, e1_resized], dim=1)
        e3 = self.enc3(self.fuse3(combined3))

        e4_input = self.pool(e3)
        e2_resized = F.interpolate(self.e2_adapter(e2), size=e4_input.shape[2:], mode='bilinear', align_corners=True)
        combined4 = torch.cat([e4_input, e2_resized], dim=1)
        e4 = self.enc4(self.fuse4(combined4))

        b = self.bottleneck(self.pool(e4))

        fpn_feats = self.fpn([e1, e2, e3, e4, b])

        d4 = self.up4(b)
        d4 = torch.cat([d4, fpn_feats[3], e4], dim=1)
        d4 = self.cbam4(d4)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, fpn_feats[2], e3], dim=1)
        d3 = self.cbam3(d3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, fpn_feats[1], e2], dim=1)
        d2 = self.cbam2(d2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, fpn_feats[0], e1], dim=1)
        d1 = self.cbam1(d1)
        d1 = self.dec1(d1)

        return self.final_conv(d1)



import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.residual = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.residual(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)

class TopDownFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1) for in_ch in in_channels_list
        ])
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in in_channels_list
        ])

    def forward(self, features):
        features = features[::-1]
        lateral = [l(f) for l, f in zip(self.lateral_convs, features)]
        fpn_outputs = []
        x = lateral[0]
        fpn_outputs.append(self.smooth_convs[0](x))
        for i in range(1, len(lateral)):
            x = F.interpolate(x, size=lateral[i].shape[2:], mode='nearest') + lateral[i]
            fpn_outputs.append(self.smooth_convs[i](x))
        return fpn_outputs[::-1]

class Del_CBAM(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc1 = ConvBlock(in_ch, features[0])
        self.enc2 = ConvBlock(features[0], features[1])
        self.enc3 = ConvBlock(features[1], features[2])
        self.enc4 = ConvBlock(features[2], features[3])
        self.bottleneck = ConvBlock(features[3], features[3]*2)

        self.e1_adapter = nn.Conv2d(features[0], features[0], kernel_size=1)
        self.e2_adapter = nn.Conv2d(features[1], features[1], kernel_size=1)
        self.fuse3 = nn.Conv2d(features[1] + features[0], features[1], kernel_size=1)
        self.fuse4 = nn.Conv2d(features[2] + features[1], features[2], kernel_size=1)

        self.fpn = TopDownFPN(
            in_channels_list=[features[3]*2, features[3], features[2], features[1], features[0]],
            out_channels=128
        )

        self.up4 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(128*2 + features[3], 128)

        self.up3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(128*2 + features[2], 128)

        self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128*2 + features[1], 128)

        self.up1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128*2 + features[0], 128)

        self.final_conv = nn.Conv2d(128, out_ch, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))

        e3_input = self.pool(e2)
        e1_resized = F.interpolate(self.e1_adapter(e1), size=e3_input.shape[2:], mode='bilinear', align_corners=True)
        combined3 = torch.cat([e3_input, e1_resized], dim=1)
        e3 = self.enc3(self.fuse3(combined3))

        e4_input = self.pool(e3)
        e2_resized = F.interpolate(self.e2_adapter(e2), size=e4_input.shape[2:], mode='bilinear', align_corners=True)
        combined4 = torch.cat([e4_input, e2_resized], dim=1)
        e4 = self.enc4(self.fuse4(combined4))

        b = self.bottleneck(self.pool(e4))

        fpn_feats = self.fpn([e1, e2, e3, e4, b])

        d4 = self.up4(fpn_feats[4])
        d4 = torch.cat([d4, fpn_feats[3], e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, fpn_feats[2], e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, fpn_feats[1], e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, fpn_feats[0], e1], dim=1)
        d1 = self.dec1(d1)

        return self.final_conv(d1)

# ==============================================================================




import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x))
        max = self.fc(self.max_pool(x))
        out = avg + max
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.residual = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.residual(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)

class TopDownFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1) for in_ch in in_channels_list
        ])
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in in_channels_list
        ])

    def forward(self, features):
        features = features[::-1]
        lateral = [l(f) for l, f in zip(self.lateral_convs, features)]
        fpn_outputs = []
        x = lateral[0]
        fpn_outputs.append(self.smooth_convs[0](x))
        for i in range(1, len(lateral)):
            x = F.interpolate(x, size=lateral[i].shape[2:], mode='nearest') + lateral[i]
            fpn_outputs.append(self.smooth_convs[i](x))
        return fpn_outputs[::-1]

class FPNUNetV3_CBAM_Residual_SUM(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 编码器
        self.enc1 = ConvBlock(in_ch, features[0])     # 64
        self.enc2 = ConvBlock(features[0], features[1])  # 128
        self.enc3 = ConvBlock(features[1], features[2])  # 256
        self.enc4 = ConvBlock(features[2], features[3])  # 512
        self.bottleneck = ConvBlock(features[3], features[3]*2)  # 1024

        # encoder阶段特征融合（skip跳接）adapter（128通道统一）
        self.e1_encoder_adapter = nn.Conv2d(features[0], features[1], kernel_size=1)
        self.e2_encoder_adapter = nn.Conv2d(features[1], features[2], kernel_size=1)

        # decoder跳接adapter：将 e1 ~ e4 全部适配为 128 通道
        self.e1_adapter = nn.Conv2d(features[0], 128, kernel_size=1)
        self.e2_adapter = nn.Conv2d(features[1], 128, kernel_size=1)
        self.e3_adapter = nn.Conv2d(features[2], 128, kernel_size=1)
        self.e4_adapter = nn.Conv2d(features[3], 128, kernel_size=1)

        self.fpn = TopDownFPN(
            in_channels_list=[features[3]*2, features[3], features[2], features[1], features[0]],
            out_channels=128
        )

        # 解码器
        self.up4 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.cbam4 = CBAM(128)
        self.dec4 = ConvBlock(128, 128)

        self.up3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.cbam3 = CBAM(128)
        self.dec3 = ConvBlock(128, 128)

        self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.cbam2 = CBAM(128)
        self.dec2 = ConvBlock(128, 128)

        self.up1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.cbam1 = CBAM(128)
        self.dec1 = ConvBlock(128, 128)

        
        
        # --- 主输出 ---

        self.final_conv = nn.Conv2d(128, out_ch, kernel_size=1)

        
        # --- 辅助输出 (deep supervision) ---
        self.aux4 = nn.Conv2d(128, out_ch, kernel_size=1)
        self.aux3 = nn.Conv2d(128, out_ch, kernel_size=1)
        self.aux2 = nn.Conv2d(128, out_ch, kernel_size=1)
              
        
    def forward(self, x):
        # 编码器
        e1 = self.enc1(x)                          # [B, 64, H, W]
        e2 = self.enc2(self.pool(e1))              # [B, 128, H/2, W/2]

        e3_input = self.pool(e2)                   # [B, 128, H/4, W/4]
        e1_resized = F.interpolate(
            self.e1_encoder_adapter(e1), 
            size=e3_input.shape[2:], 
            mode='bilinear', align_corners=True
        )
        e3 = self.enc3(e3_input + e1_resized)      # [B, 256, H/4, W/4]

        e4_input = self.pool(e3)
        e2_resized = F.interpolate(
            self.e2_encoder_adapter(e2), 
            size=e4_input.shape[2:], 
            mode='bilinear', align_corners=True
        )
        e4 = self.enc4(e4_input + e2_resized)      # [B, 512, H/8, W/8]

        b = self.bottleneck(self.pool(e4))         # [B, 1024, H/16, W/16]

        # FPN 输出（5层全为 [B, 128, ...]）
        fpn_feats = self.fpn([e1, e2, e3, e4, b])

        # 解码阶段，全部加法融合，adapter 压缩 eX → 128 通道
        d4 = self.up4(fpn_feats[4]) + fpn_feats[3] + self.e4_adapter(e4)
        d4 = self.cbam4(d4)
        d4 = self.dec4(d4)

        d3 = self.up3(d4) + fpn_feats[2] + self.e3_adapter(e3)
        d3 = self.cbam3(d3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3) + fpn_feats[1] + self.e2_adapter(e2)
        d2 = self.cbam2(d2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2) + fpn_feats[0] + self.e1_adapter(e1)
        d1 = self.cbam1(d1)
        d1 = self.dec1(d1)


         # --- 主输出 ---
        out_main = self.final_conv(d1)

        # --- 辅助输出 (上采样到原图大小) ---
        # aux_out2 = F.interpolate(self.aux2(d2), size=x.shape[2:], mode='bilinear', align_corners=True)
        # aux_out3 = F.interpolate(self.aux3(d3), size=x.shape[2:], mode='bilinear', align_corners=True)
        # aux_out4 = F.interpolate(self.aux4(d4), size=x.shape[2:], mode='bilinear', align_corners=True)

        # return out_main, aux_out2, aux_out3, aux_out4
        return out_main






# ==================================================================





class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x))
        max = self.fc(self.max_pool(x))
        out = avg + max
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x



class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.residual = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.residual(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)



class TopDownFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1) for in_ch in in_channels_list
        ])
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in in_channels_list
        ])

    def forward(self, features):
        features = features[::-1]
        lateral = [l(f) for l, f in zip(self.lateral_convs, features)]
        fpn_outputs = []
        x = lateral[0]
        fpn_outputs.append(self.smooth_convs[0](x))
        for i in range(1, len(lateral)):
            x = F.interpolate(x, size=lateral[i].shape[2:], mode='nearest') + lateral[i]
            fpn_outputs.append(self.smooth_convs[i](x))
        return fpn_outputs[::-1]

class FPNUNetV3_CBAM_Residual(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc1 = ConvBlock(in_ch, features[0])
        self.enc2 = ConvBlock(features[0], features[1])
        self.enc3 = ConvBlock(features[1], features[2])
        self.enc4 = ConvBlock(features[2], features[3])
        self.bottleneck = ConvBlock(features[3], features[3]*2)

        self.e1_adapter = nn.Conv2d(features[0], features[0], kernel_size=1)
        self.e2_adapter = nn.Conv2d(features[1], features[1], kernel_size=1)
        self.fuse3 = nn.Conv2d(features[1] + features[0], features[1], kernel_size=1)
        self.fuse4 = nn.Conv2d(features[2] + features[1], features[2], kernel_size=1)

        self.fpn = TopDownFPN(
            in_channels_list=[features[3]*2, features[3], features[2], features[1], features[0]],
            out_channels=128
        )

        self.up4 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.cbam4 = CBAM(128*2 + features[3])
        self.dec4 = ConvBlock(128*2 + features[3], 128)

        self.up3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.cbam3 = CBAM(128*2 + features[2])
        self.dec3 = ConvBlock(128*2 + features[2], 128)

        self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.cbam2 = CBAM(128*2 + features[1])
        self.dec2 = ConvBlock(128*2 + features[1], 128)

        self.up1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.cbam1 = CBAM(128*2 + features[0])
        self.dec1 = ConvBlock(128*2 + features[0], 128)

        self.final_conv = nn.Conv2d(128, out_ch, kernel_size=1)
        
        
        
        
                
        # --- 辅助输出 (deep supervision) ---
        self.aux4 = nn.Conv2d(128, out_ch, kernel_size=1)
        self.aux3 = nn.Conv2d(128, out_ch, kernel_size=1)
        self.aux2 = nn.Conv2d(128, out_ch, kernel_size=1)
              
        

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))

        e3_input = self.pool(e2)
        e1_resized = F.interpolate(self.e1_adapter(e1), size=e3_input.shape[2:], mode='bilinear', align_corners=True)
        combined3 = torch.cat([e3_input, e1_resized], dim=1)
        e3 = self.enc3(self.fuse3(combined3))

        e4_input = self.pool(e3)
        e2_resized = F.interpolate(self.e2_adapter(e2), size=e4_input.shape[2:], mode='bilinear', align_corners=True)
        combined4 = torch.cat([e4_input, e2_resized], dim=1)
        e4 = self.enc4(self.fuse4(combined4))

        b = self.bottleneck(self.pool(e4))

        fpn_feats = self.fpn([e1, e2, e3, e4, b])

        d4 = self.up4(fpn_feats[4])
        d4 = torch.cat([d4, fpn_feats[3], e4], dim=1)
        d4 = self.cbam4(d4)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, fpn_feats[2], e3], dim=1)
        d3 = self.cbam3(d3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, fpn_feats[1], e2], dim=1)
        d2 = self.cbam2(d2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, fpn_feats[0], e1], dim=1)
        d1 = self.cbam1(d1)
        d1 = self.dec1(d1)
        
        out_main = self.final_conv(d1)
        
                # --- 辅助输出 (上采样到原图大小) ---
        aux_out2 = F.interpolate(self.aux2(d2), size=x.shape[2:], mode='bilinear', align_corners=True)
        aux_out3 = F.interpolate(self.aux3(d3), size=x.shape[2:], mode='bilinear', align_corners=True)
        aux_out4 = F.interpolate(self.aux4(d4), size=x.shape[2:], mode='bilinear', align_corners=True)
        
        

        
        return out_main, aux_out2, aux_out3, aux_out4 





# model = FPNUNetV3_CBAM_Residual_2(in_ch=1, out_ch=1)
# x = torch.randn(1, 1, 256, 256)
# # out,x1,x2,x3 = model(x)
# out = model(x)
# print(out)  # torch.Size([1, 1, 256, 256])
# print(out.shape)  # torch.Size([1, 1, 256, 256])





