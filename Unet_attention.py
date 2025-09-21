import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: gating signal (decoder), x: skip connection (encoder)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# class ConvBlock(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#         )
#     def forward(self, x):
#         return self.conv(x)

class AttentionUNet(nn.Module):
    """
    Attention U-Net for medical image segmentation.

    Args:
        in_ch: number of input channels (e.g., 1 for grayscale)
        out_ch: number of segmentation classes (e.g., 1 for binary)
        features: list of channel sizes for encoder, e.g. [64,128,256,512]
    """
    def __init__(self, in_ch=1, out_ch=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        self.enc1 = ConvBlock(in_ch, features[0])
        self.enc2 = ConvBlock(features[0], features[1])
        self.enc3 = ConvBlock(features[1], features[2])
        self.enc4 = ConvBlock(features[2], features[3])

        # Bottleneck
        self.bottleneck = ConvBlock(features[3], features[3]*2)

        # Decoder
        self.up4 = nn.ConvTranspose2d(features[3]*2, features[3], kernel_size=2, stride=2)
        self.att4 = AttentionGate(F_g=features[3], F_l=features[3], F_int=features[3]//2)
        self.dec4 = ConvBlock(features[3]*2, features[3])

        self.up3 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.att3 = AttentionGate(F_g=features[2], F_l=features[2], F_int=features[2]//2)
        self.dec3 = ConvBlock(features[2]*2, features[2])

        self.up2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.att2 = AttentionGate(F_g=features[1], F_l=features[1], F_int=features[1]//2)
        self.dec2 = ConvBlock(features[1]*2, features[1])

        self.up1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.att1 = AttentionGate(F_g=features[0], F_l=features[0], F_int=features[0]//2)
        self.dec1 = ConvBlock(features[0]*2, features[0])

        # Final 1x1 conv
        self.final_conv = nn.Conv2d(features[0], out_ch, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        e2 = self.enc2(p1)
        p2 = self.pool(e2)
        e3 = self.enc3(p2)
        p3 = self.pool(e3)
        e4 = self.enc4(p3)
        p4 = self.pool(e4)

        # Bottleneck
        b = self.bottleneck(p4)

        # Decoder + attention
        d4 = self.up4(b)
        e4_att = self.att4(g=d4, x=e4)
        d4 = torch.cat([d4, e4_att], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        e3_att = self.att3(g=d3, x=e3)
        d3 = torch.cat([d3, e3_att], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        e2_att = self.att2(g=d2, x=e2)
        d2 = torch.cat([d2, e2_att], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        e1_att = self.att1(g=d1, x=e1)
        d1 = torch.cat([d1, e1_att], dim=1)
        d1 = self.dec1(d1)

        # Final conv
        return self.final_conv(d1)





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






######################################################################################




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



##################################################################




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

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

        # 1x1 conv for matching channels if needed
        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)



# class ConvBlock(nn.Module):
#     def __init__(self, in_ch, out_ch, dropout=0.2):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, 3, padding=1),
#             nn.GroupNorm(8, out_ch),
#             nn.ReLU(inplace=True),
#             nn.Dropout2d(dropout),
#             nn.Conv2d(out_ch, out_ch, 3, padding=1),
#             nn.GroupNorm(8, out_ch),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x):
#         return self.block(x)

class ResBlock_D(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_rate=0.15):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout2d(p=dropout_rate)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.dropout2 = nn.Dropout2d(p=dropout_rate)

        # Shortcut connection for channel matching
        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout2(out)

        out += identity
        return self.relu(out)



class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # [B,C,1,1]
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = kernel_size // 2

        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel-wise max & avg -> concat
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B,1,H,W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B,1,H,W]
        x_cat = torch.cat([avg_out, max_out], dim=1)  # [B,2,H,W]
        out = self.conv(x_cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x):
        # Channel Attention
        x_ca = self.channel_attention(x) * x
        # Spatial Attention
        x_sa = self.spatial_attention(x_ca) * x_ca
        return x_sa


class FPNUNetV3(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ---------- Encoder ----------
        self.enc1 = ResBlock_D(in_ch, features[0])
        self.enc2 = ResBlock_D(features[0], features[1])
        self.enc3 = ResBlock_D(features[1], features[2])
        self.enc4 = ResBlock_D(features[2], features[3])
        self.bottleneck = ResBlock_D(features[3], features[3] * 2)

        # ---------- FPN ----------
        self.fpn = TopDownFPN(
            in_channels_list=[
                features[3]*2,  # bottleneck
                features[3],    # e4
                features[2],    # e3
                features[1],    # e2
                features[0],    # e1
            ],
            out_channels=128
        )

        # ---------- CBAM with skip-connection ----------
        self.cbam4 = CBAM(128*2 + features[3])  # 128*2 + 512
        self.cbam3 = CBAM(128*2 + features[2])  # 128*2 + 256
        self.cbam2 = CBAM(128*2 + features[1])  # 128*2 + 128
        self.cbam1 = CBAM(128*2 + features[0])  # 128*2 + 64

        # ---------- Decoder ----------
        self.up4 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec4 = ResBlock_D(128*2 + features[3], 128)

        self.up3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec3 = ResBlock_D(128*2 + features[2], 128)

        self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec2 = ResBlock_D(128*2 + features[1], 128)

        self.up1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec1 = ResBlock_D(128*2 + features[0], 128)

        self.final_conv = nn.Conv2d(128, out_ch, kernel_size=1)

    def forward(self, x):
        # ----------- Encoder -----------
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        # ----------- FPN Features -----------
        fpn_feats = self.fpn([e1, e2, e3, e4, b])  # [P1, P2, P3, P4, P5]

        # ----------- Decoder with FPN + Encoder skip + CBAM -----------
        d4 = self.up4(fpn_feats[4])
        fusion4 = torch.cat([d4, fpn_feats[3], e4], dim=1)
        fusion4 = self.cbam4(fusion4)
        d4 = self.dec4(fusion4)

        d3 = self.up3(d4)
        fusion3 = torch.cat([d3, fpn_feats[2], e3], dim=1)
        fusion3 = self.cbam3(fusion3)
        d3 = self.dec3(fusion3)

        d2 = self.up2(d3)
        fusion2 = torch.cat([d2, fpn_feats[1], e2], dim=1)
        fusion2 = self.cbam2(fusion2)
        d2 = self.dec2(fusion2)

        d1 = self.up1(d2)
        fusion1 = torch.cat([d1, fpn_feats[0], e1], dim=1)
        fusion1 = self.cbam1(fusion1)
        d1 = self.dec1(fusion1)

        return self.final_conv(d1)

# ======================================================================

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
        max_out,_ = torch.max(x, dim=1, keepdim=True)
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


# class ConvBlock(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#         )
#     def forward(self, x):
#         return self.conv(x)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

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
    

    
class FPNUNetV3_CBAM(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        self.enc1 = ConvBlock(in_ch, features[0])
        self.enc2 = ConvBlock(features[0], features[1])
        self.enc3 = ConvBlock(features[1], features[2])
        self.enc4 = ConvBlock(features[2], features[3])
        self.bottleneck = ConvBlock(features[3], features[3]*2)

        # FPN
        self.fpn = TopDownFPN(
            in_channels_list=[features[3]*2, features[3], features[2], features[1], features[0]],
            out_channels=128
        )

        # Decoder + CBAM
        self.up4 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.cbam4 = CBAM(128 * 2)
        self.dec4 = ConvBlock(128 * 2, 128)

        self.up3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.cbam3 = CBAM(128 * 2)
        self.dec3 = ConvBlock(128 * 2, 128)

        self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.cbam2 = CBAM(128 * 2)
        self.dec2 = ConvBlock(128 * 2, 128)

        self.up1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.cbam1 = CBAM(128 * 2)
        self.dec1 = ConvBlock(128 * 2, 128)

        self.final_conv = nn.Conv2d(128, out_ch, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        # FPN
        fpn_feats = self.fpn([e1, e2, e3, e4, b])

        # Decoder + CBAM
        d4 = self.up4(fpn_feats[4])
        d4 = torch.cat([d4, fpn_feats[3]], dim=1)
        d4 = self.cbam4(d4)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, fpn_feats[2]], dim=1)
        d3 = self.cbam3(d3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, fpn_feats[1]], dim=1)
        d2 = self.cbam2(d2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, fpn_feats[0]], dim=1)
        d1 = self.cbam1(d1)
        d1 = self.dec1(d1)

        return self.final_conv(d1)



# =======================================================================



# ------------------- CBAM -------------------
class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, 1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_attention(x)
        x = x * ca
        sa = self.spatial_attention(
            torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], dim=1)
        )
        return x * sa

# ------------------- ConvBlock（优化） -------------------
# class ConvBlock(nn.Module):
#     def __init__(self, in_ch, out_ch, dropout=0.2):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, 3, padding=1),
#             nn.GroupNorm(8, out_ch),
#             nn.ReLU(inplace=True),
#             nn.Dropout2d(dropout),
#             nn.Conv2d(out_ch, out_ch, 3, padding=1),
#             nn.GroupNorm(8, out_ch),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x):
#         return self.block(x)

# ------------------- FPN -------------------
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

# ------------------- 主模型 -------------------
class FPNUNetV3_CBAM_pro(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        # 自动构造 encoder（简洁写法）
        self.encoders = nn.ModuleList([
            ConvBlock(in_ch, features[0]),
            ConvBlock(features[0], features[1]),
            ConvBlock(features[1], features[2]),
            ConvBlock(features[2], features[3])
        ])
        self.bottleneck = ConvBlock(features[3], features[3]*2)

        # FPN
        self.fpn = TopDownFPN(
            in_channels_list=[features[3]*2, features[3], features[2], features[1], features[0]],
            out_channels=128
        )

        # Decoder + CBAM
        self.up4 = nn.ConvTranspose2d(128, 128, 2, 2)
        self.cbam4 = CBAM(128 * 2)
        self.dec4 = ConvBlock(128 * 2, 128)

        self.up3 = nn.ConvTranspose2d(128, 128, 2, 2)
        self.cbam3 = CBAM(128 * 2)
        self.dec3 = ConvBlock(128 * 2, 128)

        self.up2 = nn.ConvTranspose2d(128, 128, 2, 2)
        self.cbam2 = CBAM(128 * 2)
        self.dec2 = ConvBlock(128 * 2, 128)

        self.up1 = nn.ConvTranspose2d(128, 128, 2, 2)
        self.cbam1 = CBAM(128 * 2)
        self.dec1 = ConvBlock(128 * 2, 128)

        self.final_conv = nn.Conv2d(128, out_ch, kernel_size=1)

    def forward(self, x):
        feats = []
        for enc in self.encoders:
            x = enc(x)
            feats.append(x)
            x = self.pool(x)
        b = self.bottleneck(x)

        # FPN 融合
        fpn_feats = self.fpn(feats + [b])

        d4 = self.dec4(self.cbam4(torch.cat([self.up4(fpn_feats[4]), fpn_feats[3]], dim=1)))
        d3 = self.dec3(self.cbam3(torch.cat([self.up3(d4), fpn_feats[2]], dim=1)))
        d2 = self.dec2(self.cbam2(torch.cat([self.up2(d3), fpn_feats[1]], dim=1)))
        d1 = self.dec1(self.cbam1(torch.cat([self.up1(d2), fpn_feats[0]], dim=1)))

        return self.final_conv(d1)
    
    # =========================================================================
    
    
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------- Channel Attention -----------------
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

# ----------------- Spatial Attention -----------------
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

# ----------------- CBAM -----------------
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

# ----------------- Conv Block -----------------
# class ConvBlock(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x):
#         return self.conv(x)

# ----------------- FPN Module -----------------
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
        features = features[::-1]  # from high level to low level
        lateral = [l(f) for l, f in zip(self.lateral_convs, features)]
        fpn_outputs = []
        x = lateral[0]
        fpn_outputs.append(self.smooth_convs[0](x))
        for i in range(1, len(lateral)):
            x = F.interpolate(x, size=lateral[i].shape[2:], mode='nearest') + lateral[i]
            fpn_outputs.append(self.smooth_convs[i](x))
        return fpn_outputs[::-1]  # reverse back to low to high

# ----------------- FPN-UNet with CBAM and Skip Connections -----------------
class FPNUNetV3_CCBAM(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        self.enc1 = ConvBlock(in_ch, features[0])
        self.enc2 = ConvBlock(features[0], features[1])
        self.enc3 = ConvBlock(features[1], features[2])
        self.enc4 = ConvBlock(features[2], features[3])
        self.bottleneck = ConvBlock(features[3], features[3]*2)

        # FPN
        self.fpn = TopDownFPN(
            in_channels_list=[features[3]*2, features[3], features[2], features[1], features[0]],
            out_channels=128
        )

        # Decoder + CBAM + Skip Fusion
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
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        # FPN
        fpn_feats = self.fpn([e1, e2, e3, e4, b])  # [P1, P2, P3, P4, P5]

        # Decoder with skip + CBAM
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








# 将之前的 FPNUNetV3_CBAM 模型整合残差结构后写入 .py 文件

# 修改后的 fpn_cbam_unet_with_residual.py 模块，添加通道映射后打包保存


# ----------------- Channel Attention -----------------
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

# ----------------- Spatial Attention -----------------
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

# ----------------- CBAM -----------------
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

# ----------------- Conv Block -----------------
# class ConvBlock(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x):
#         return self.conv(x)

# ----------------- FPN Module -----------------
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
        features = features[::-1]  # from high level to low level
        lateral = [l(f) for l, f in zip(self.lateral_convs, features)]
        fpn_outputs = []
        x = lateral[0]
        fpn_outputs.append(self.smooth_convs[0](x))
        for i in range(1, len(lateral)):
            x = F.interpolate(x, size=lateral[i].shape[2:], mode='nearest') + lateral[i]
            fpn_outputs.append(self.smooth_convs[i](x))
        return fpn_outputs[::-1]  # reverse back to low to high

# ----------------- FPN-UNet with CBAM and Residual Skip -----------------
class FPNUNetV3_CBAMM(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        self.enc1 = ConvBlock(in_ch, features[0])
        self.enc2 = ConvBlock(features[0], features[1])
        self.enc3 = ConvBlock(features[1], features[2])
        self.enc4 = ConvBlock(features[2], features[3])
        self.bottleneck = ConvBlock(features[3], features[3]*2)

        # Residual projection for skip
        self.e1_proj = nn.Conv2d(features[0], features[2], kernel_size=1)
        self.e2_proj = nn.Conv2d(features[1], features[3], kernel_size=1)

        # FPN
        self.fpn = TopDownFPN(
            in_channels_list=[features[3]*2, features[3], features[2], features[1], features[0]],
            out_channels=128
        )

        # Decoder + CBAM + Skip Fusion
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
        # Encoder with residual long-skip connections
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))

        e3_input = self.pool(e2)
        e1_resized = F.interpolate(e1, size=e3_input.shape[2:], mode='bilinear', align_corners=False)
        e1_resized = self.e1_proj(e1_resized)
        e3 = self.enc3(e3_input + e1_resized)

        e4_input = self.pool(e3)
        e2_resized = F.interpolate(e2, size=e4_input.shape[2:], mode='bilinear', align_corners=False)
        e2_resized = self.e2_proj(e2_resized)
        e4 = self.enc4(e4_input + e2_resized)

        b = self.bottleneck(self.pool(e4))

        # FPN
        fpn_feats = self.fpn([e1, e2, e3, e4, b])

        # Decoder with skip + CBAM
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



# ================================================================






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



# ======================================================





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



# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++




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









# ==========================================================




import torch
import torch.nn as nn
import torch.nn.functional as F

# class DepthwiseSeparableConv(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.depthwise = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
#         self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
#     def forward(self, x):
#         return self.pointwise(self.depthwise(x))

# class ResidualConvBlock(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.conv1 = DepthwiseSeparableConv(in_ch, out_ch)
#         self.bn1 = nn.BatchNorm2d(out_ch)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = DepthwiseSeparableConv(out_ch, out_ch)
#         self.bn2 = nn.BatchNorm2d(out_ch)
#         self.se = SEBlock(out_ch)
#         self.residual = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, 1, bias=False),
#             nn.BatchNorm2d(out_ch)
#         ) if in_ch != out_ch else nn.Identity()

#     def forward(self, x):
#         identity = self.residual(x)
#         out = self.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out = self.se(out)
#         return self.relu(out + identity)




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




# =================================================================================================



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

class AttentionFuse(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels * 3, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, d, fpn, skip):
        concat = torch.cat([d, fpn, skip], dim=1)
        weights = self.fc(self.global_pool(concat)).view(concat.size(0), 3, -1, 1, 1)
        d_w, fpn_w, skip_w = weights[:, 0], weights[:, 1], weights[:, 2]
        return d * d_w + fpn * fpn_w + skip * skip_w

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

class FPNUNet_A_Lightt(nn.Module):
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

        self.skip4 = nn.Conv2d(features[3], 128, 1)
        self.skip3 = nn.Conv2d(features[2], 128, 1)
        self.skip2 = nn.Conv2d(features[1], 128, 1)
        self.skip1 = nn.Conv2d(features[0], 128, 1)

        self.att4 = AttentionFuse(128)
        self.att3 = AttentionFuse(128)
        self.att2 = AttentionFuse(128)
        self.att1 = AttentionFuse(128)

        self.dec4 = ResidualConvBlock(128, 128)
        self.dec3 = ResidualConvBlock(128, 128)
        self.dec2 = ResidualConvBlock(128, 128)
        self.dec1 = ResidualConvBlock(128, 128)

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
        d4 = self.att4(d4, fpn_feats[3], self.skip4(e4))
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = self.att3(d3, fpn_feats[2], self.skip3(e3))
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self.att2(d2, fpn_feats[1], self.skip2(e2))
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self.att1(d1, fpn_feats[0], self.skip1(e1))
        d1 = self.dec1(d1)

        return self.final_conv(d1)
















    
#---------------------------------------------------------------------------------------------


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

class SPIK(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // 4, in_ch, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.global_avg_pool(x)
        w = self.fc(w)
        return x * w

class FPNUNet_Lighttt(nn.Module):
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

        self.spik4 = SPIK(features[3])
        self.spik3 = SPIK(features[2])
        self.spik2 = SPIK(features[1])
        self.spik1 = SPIK(features[0])

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
        e1 = self.spik1(self.enc1(x))
        e2 = self.spik2(self.enc2(self.pool(e1)))
        e3 = self.spik3(self.enc3(self.pool(e2)))
        e4 = self.spik4(self.enc4(self.pool(e3)))
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
 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# ===============================================================================






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




# =================================================================





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

# ---------------------------------------------------------------------------------------------



from torchvision.models import resnet18
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
from torchvision.models import resnet18

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
        base_model = resnet18(pretrained=False)

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



# ====================================================================================================



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



# -------------------------------------------------------------------------------------------------




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



# ----------------------------------------------------------------------------------



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

class SPIK(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // 4, in_ch, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.global_avg_pool(x)
        w = self.fc(w)
        return x * w

class Del_Res(nn.Module):
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

        self.spik4 = SPIK(features[3])
        self.spik3 = SPIK(features[2])
        self.spik2 = SPIK(features[1])
        self.spik1 = SPIK(features[0])

        self.dec4 = ResidualConvBlock(384, 128)
        self.dec3 = ResidualConvBlock(384, 128)
        self.dec2 = ResidualConvBlock(384, 128)
        self.dec1 = ResidualConvBlock(384, 128)

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
        e1 = self.spik1(self.enc1(x))
        e2 = self.spik2(self.enc2(self.pool(e1)))
        e3 = self.spik3(self.enc3(self.pool(e2)))
        e4 = self.spik4(self.enc4(self.pool(e3)))
        b = self.bottleneck(self.pool(e4))

        fpn_feats = self.fpn([e1, e2, e3, e4, b])

        d4 = self.up4(fpn_feats[4])
        d4 = torch.cat([d4, fpn_feats[3], fpn_feats[3]], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, fpn_feats[2], fpn_feats[2]], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, fpn_feats[1], fpn_feats[1]], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, fpn_feats[0], fpn_feats[0]], dim=1)
        d1 = self.dec1(d1)

        return self.final_conv(d1)
