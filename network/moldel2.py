# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# # -------------------- DownFPN --------------------
# class DownFPN(nn.Module):
#     def __init__(self, in_channels_list, out_channels):
#         super().__init__()
#         self.lateral_convs = nn.ModuleList([
#             nn.Conv2d(in_ch, out_channels, kernel_size=1) for in_ch in in_channels_list
#         ])
#         self.smooth_convs = nn.ModuleList([
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in in_channels_list
#         ])

#     def forward(self, features):
#         lateral = [l(f) for l, f in zip(self.lateral_convs, features)]
#         fpn_outputs = []
#         x = lateral[0]
#         fpn_outputs.append(self.smooth_convs[0](x))
#         for i in range(1, len(lateral)):
#             if x.shape[2:] != lateral[i].shape[2:]:
#                 x = F.interpolate(x, size=lateral[i].shape[2:], mode='nearest')
#             x = x + lateral[i]
#             fpn_outputs.append(self.smooth_convs[i](x))
#         return fpn_outputs


# # -------------------- CBAM --------------------
# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
#         )
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg = self.fc(self.avg_pool(x))
#         return self.sigmoid(avg)


# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super().__init__()
#         self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x_cat = torch.cat([avg_out, max_out], dim=1)
#         return self.sigmoid(self.conv(x_cat))


# class CBAM(nn.Module):
#     def __init__(self, channels, reduction=16, spatial_kernel=7):
#         super().__init__()
#         self.ca = ChannelAttention(channels, reduction)
#         self.sa = SpatialAttention(spatial_kernel)

#     def forward(self, x):
#         x = x * self.ca(x)
#         x = x * self.sa(x)
#         return x


# # -------------------- ConvBlock --------------------
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


# # -------------------- FPN + U-Net + CBAM --------------------
# class FPNUNet_CBAMResidual(nn.Module):
#     def __init__(self, in_ch=1, out_ch=1, features=[64, 128, 256, 512]):
#         super().__init__()
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
          
#         # ---- Encoder ----
#         self.enc1 = ConvBlock(in_ch, features[0])
#         self.enc2 = ConvBlock(features[0], features[1])
#         self.enc3 = ConvBlock(features[1], features[2])
#         self.enc4 = ConvBlock(features[2], features[3])
#         self.bottleneck = ConvBlock(features[3], features[3] * 2)

#         # ---- Cross-layer fusion ----
#         self.e1_adapter = nn.Conv2d(features[0], features[0], kernel_size=1)
#         self.e2_adapter = nn.Conv2d(features[1], features[1], kernel_size=1)
#         self.fuse3 = nn.Conv2d(features[1] + features[0], features[1], kernel_size=1)
#         self.fuse4 = nn.Conv2d(features[2] + features[1], features[2], kernel_size=1)

#         # ---- DownFPN ----
#         self.fpn = DownFPN(
#             in_channels_list=[features[0], features[1], features[2], features[3], features[3] * 2],
#             out_channels=128
#         )

#         # ---- b adapter ----
#         self.b_adapter = nn.Conv2d(features[3] * 2, 128, kernel_size=1)

#         # ---- Decoder (修复通道数) ----
#         # d4: 128 + 128 + 512 + 128 = 896
#         self.up4 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
#         self.cbam4 = CBAM(896)
#         self.dec4 = ConvBlock(896, 128)

#         # d3: 128 + 128 + 256 + 128 = 640
#         self.up3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
#         self.cbam3 = CBAM(640)
#         self.dec3 = ConvBlock(640, 128)

#         # d2: 128 + 128 + 128 + 128 = 512
#         self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
#         self.cbam2 = CBAM(512)
#         self.dec2 = ConvBlock(512, 128)

#         # d1: 128 + 128 + 64 + 128 = 448
#         self.up1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
#         self.cbam1 = CBAM(448)
#         self.dec1 = ConvBlock(448, 128)

#         self.final_conv = nn.Conv2d(128, out_ch, kernel_size=1)

#     def forward(self, x):
#         # ---- Encoder ----
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

#         # ---- FPN ----
#         fpn_feats = self.fpn([e1, e2, e3, e4, b])
#         b_aligned = self.b_adapter(b)

#         # ---- Decoder ----
#         # d4
#         d4 = self.up4(fpn_feats[4])
#         b_resized = F.interpolate(b_aligned, size=d4.shape[2:], mode="bilinear", align_corners=True)
#         # d4 = torch.cat([d4, fpn_feats[3], e4, b_resized], dim=1)
#         d4 = torch.cat([d4, fpn_feats[3], e4])#, dim=1)
#         d4 = self.cbam4(d4)
#         d4 = self.dec4(d4)

#         # d3
#         d3 = self.up3(d4)
#         b_resized = F.interpolate(b_aligned, size=d3.shape[2:], mode="bilinear", align_corners=True)
#         # d3 = torch.cat([d3, fpn_feats[2], e3, b_resized], dim=1)
#         d3 = torch.cat([d3, fpn_feats[2], e3])#, dim=1)
#         d3 = self.cbam3(d3)
#         d3 = self.dec3(d3)

#         # d2
#         d2 = self.up2(d3)
#         b_resized = F.interpolate(b_aligned, size=d2.shape[2:], mode="bilinear", align_corners=True)
#         d2 = torch.cat([d2, fpn_feats[1], e2, b_resized], dim=1)
#         d2 = self.cbam2(d2)
#         d2 = self.dec2(d2)

#         # d1
#         d1 = self.up1(d2)
#         b_resized = F.interpolate(b_aligned, size=d1.shape[2:], mode="bilinear", align_corners=True)
#         d1 = torch.cat([d1, fpn_feats[0], e1, b_resized], dim=1)
#         d1 = self.cbam1(d1)
#         d1 = self.dec1(d1)

#         return self.final_conv(d1)
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------- MaxFuseFPN：全部上采样到最大尺寸后相加，再回采样输出 --------------------
class MaxFuseFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels, up_mode="bilinear"):
        super().__init__()
        assert up_mode in ("bilinear", "nearest")
        self.up_mode = up_mode

        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, out_channels, kernel_size=1, bias=False)
            for c in in_channels_list
        ])
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
            for _ in in_channels_list
        ])

    def _upsample(self, x, size):
        if x.shape[2:] == size:
            return x
        if self.up_mode == "bilinear":
            return F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        else:
            return F.interpolate(x, size=size, mode="nearest")

    def forward(self, features):
        # 1) 1x1 映射到统一通道
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]
        # 2) 计算最大空间大小
        Hmax = max(x.shape[2] for x in laterals)
        Wmax = max(x.shape[3] for x in laterals)
        target = (Hmax, Wmax)
        # 3) 统一上采样并相加
        fused_max = None
        for x in laterals:
            x_up = self._upsample(x, target)
            fused_max = x_up if fused_max is None else fused_max + x_up
        # 4) 回采样到各层原始尺寸，并3x3平滑
        outs = []
        for i, ref in enumerate(features):
            y = self._upsample(fused_max, ref.shape[2:])
            outs.append(self.smooth_convs[i](y))
        return outs


# -------------------- CBAM --------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg = self.fc(self.avg_pool(x))
        return self.sigmoid(avg)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
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


# -------------------- ConvBlock --------------------
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


# # -------------------- FPN + U-Net + CBAM (使用 MaxFuseFPN) --------------------
# class FPNUNet_CBAMResidual(nn.Module):
#     def __init__(self, in_ch=1, out_ch=1, features=[64, 128, 256, 512]):
#         super().__init__()
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

#         # ---- Encoder ----
#         self.enc1 = ConvBlock(in_ch, features[0])           # H
#         self.enc2 = ConvBlock(features[0], features[1])     # H/2
#         self.enc3 = ConvBlock(features[1], features[2])     # H/4
#         self.enc4 = ConvBlock(features[2], features[3])     # H/8
#         self.bottleneck = ConvBlock(features[3], features[3] * 2)  # H/16

#         # ---- Cross-layer fusion ----
#         self.e1_adapter = nn.Conv2d(features[0], features[0], kernel_size=1)
#         self.e2_adapter = nn.Conv2d(features[1], features[1], kernel_size=1)
#         self.fuse3 = nn.Conv2d(features[1] + features[0], features[1], kernel_size=1)
#         self.fuse4 = nn.Conv2d(features[2] + features[1], features[2], kernel_size=1)

#         # ---- MaxFuseFPN：只用 e1..e4 四层，全部上采样到最大尺寸相加 ----
#         self.fpn = MaxFuseFPN(
#             in_channels_list=[features[0], features[1], features[2], features[3]],
#             out_channels=128,
#             up_mode="bilinear"
#         )

#         # ---- b adapter ----
#         self.b_adapter = nn.Conv2d(features[3] * 2, 128, kernel_size=1)

#         # ---- Decoder ----
#         # 约定：fpn 输出 4 层，对应 e1(H), e2(H/2), e3(H/4), e4(H/8)
#         # d4：从 b(H/16) 起步上采样到 H/8，与 fpn(e4) 与 e4 融合 => 128 + 128 + 512 = 768
#         self.up4 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
#         self.cbam4 = CBAM(768)
#         self.dec4 = ConvBlock(768, 128)

#         # d3：H/8 -> H/4，与 fpn(e3), e3 融合 => 128 + 128 + 256 = 512
#         self.up3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
#         self.cbam3 = CBAM(512)
#         self.dec3 = ConvBlock(512, 128)

#         # d2：H/4 -> H/2，与 fpn(e2), e2 融合 => 128 + 128 + 128 = 384
#         self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
#         self.cbam2 = CBAM(384)
#         self.dec2 = ConvBlock(384, 128)

#         # d1：H/2 -> H，与 fpn(e1), e1 融合 => 128 + 128 + 64 = 320
#         self.up1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
#         self.cbam1 = CBAM(320)
#         self.dec1 = ConvBlock(320, 128)

#         self.final_conv = nn.Conv2d(128, out_ch, kernel_size=1)

#     def forward(self, x):
#         # ---- Encoder ----
#         e1 = self.enc1(x)                    # H
#         e2 = self.enc2(self.pool(e1))        # H/2

#         e3_in = self.pool(e2)                # H/4
#         e1_resized = F.interpolate(self.e1_adapter(e1), size=e3_in.shape[2:], mode='bilinear', align_corners=False)
#         e3 = self.enc3(self.fuse3(torch.cat([e3_in, e1_resized], dim=1)))  # H/4

#         e4_in = self.pool(e3)                # H/8
#         e2_resized = F.interpolate(self.e2_adapter(e2), size=e4_in.shape[2:], mode='bilinear', align_corners=False)
#         e4 = self.enc4(self.fuse4(torch.cat([e4_in, e2_resized], dim=1)))  # H/8

#         b  = self.bottleneck(self.pool(e4))  # H/16

#         # ---- FPN（最大尺寸相加 -> 回采样多尺度）----
#         # 输出 4 层：与 e1,e2,e3,e4 空间一致，通道=128
#         fpn_feats = self.fpn([e1, e2, e3, e4])

#         # ---- b 对齐 ----
#         b_aligned = self.b_adapter(b)        # (B,128,H/16,W/16)

#         # ---- Decoder ----
#         # d4: H/16 -> H/8（从 b 起步），与 fpn(e4), e4 融合
#         d4 = self.up4(b_aligned)                                            # (B,128,H/8,W/8)
#         d4 = torch.cat([d4, fpn_feats[3], e4], dim=1)                       # 128 + 128 + 512 = 768
#         d4 = self.cbam4(d4)
#         d4 = self.dec4(d4)                                                  # -> (B,128,H/8,W/8)

#         # d3: H/8 -> H/4，与 fpn(e3), e3 融合
#         d3 = self.up3(d4)                                                   # (B,128,H/4,W/4)
#         d3 = torch.cat([d3, fpn_feats[2], e3], dim=1)                       # 128 + 128 + 256 = 512
#         d3 = self.cbam3(d3)
#         d3 = self.dec3(d3)                                                  # -> (B,128,H/4,W/4)

#         # d2: H/4 -> H/2，与 fpn(e2), e2 融合
#         d2 = self.up2(d3)                                                   # (B,128,H/2,W/2)
#         d2 = torch.cat([d2, fpn_feats[1], e2], dim=1)                       # 128 + 128 + 128 = 384
#         d2 = self.cbam2(d2)
#         d2 = self.dec2(d2)                                                  # -> (B,128,H/2,W/2)

#         # d1: H/2 -> H，与 fpn(e1), e1 融合
#         d1 = self.up1(d2)                                                   # (B,128,H,W)
#         d1 = torch.cat([d1, fpn_feats[0], e1], dim=1)                       # 128 + 128 + 64 = 320
#         d1 = self.cbam1(d1)
#         d1 = self.dec1(d1)                                                  # -> (B,128,H,W)

#         return self.final_conv(d1)                                          # (B,out_ch,H,W)

import torch
import torch.nn as nn
import torch.nn.functional as F

class FPNUNet_CBAMResidual(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=[64, 128, 256, 512],
                 fpn_out=128, drop=0.10):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ---- Encoder ----
        self.enc1 = ConvBlock(in_ch, features[0])           # H
        self.enc2 = ConvBlock(features[0], features[1])     # H/2
        self.enc3 = ConvBlock(features[1], features[2])     # H/4
        self.enc4 = ConvBlock(features[2], features[3])     # H/8
        self.bottleneck = ConvBlock(features[3], features[3] * 2)  # H/16

        # ---- Cross-layer fusion ----
        self.e1_adapter = nn.Conv2d(features[0], features[0], kernel_size=1)
        self.e2_adapter = nn.Conv2d(features[1], features[1], kernel_size=1)
        self.fuse3 = nn.Conv2d(features[1] + features[0], features[1], kernel_size=1)
        self.fuse4 = nn.Conv2d(features[2] + features[1], features[2], kernel_size=1)

        # ---- MaxFuseFPN（把 e1..e4 全部上采到最大分辨率相加，再回采样多尺度）----
        self.fpn = MaxFuseFPN(
            in_channels_list=[features[0], features[1], features[2], features[3]],
            out_channels=fpn_out,
            up_mode="bilinear"
        )

        # ---- b adapter ----
        self.b_adapter = nn.Conv2d(features[3] * 2, fpn_out, kernel_size=1)

        # ---- Learnable gates（可学习门控，范围 0~1）----
        def make_gate(init=0.0):
            # 参数化为标量，sigmoid 后作为权重
            p = nn.Parameter(torch.tensor([init], dtype=torch.float32))
            return p
        # d4 处三路：b / fpn(e4) / e4
        self.g4_b   = make_gate(0.0)
        self.g4_fpn = make_gate(0.0)
        self.g4_e   = make_gate(0.0)
        # d3、d2、d1 处两路：fpn / e
        self.g3_fpn = make_gate(0.0); self.g3_e = make_gate(0.0)
        self.g2_fpn = make_gate(0.0); self.g2_e = make_gate(0.0)
        self.g1_fpn = make_gate(0.0); self.g1_e = make_gate(0.0)

        # ---- Decoder ----
        # d4：128 + 128 + 512 = 768
        self.up4 = nn.ConvTranspose2d(fpn_out, fpn_out, kernel_size=2, stride=2)
        self.cbam4 = CBAM(768)
        self.dec4 = ConvBlock(768, fpn_out)
        self.drop4 = nn.Dropout2d(drop)

        # d3：128 + 128 + 256 = 512
        self.up3 = nn.ConvTranspose2d(fpn_out, fpn_out, kernel_size=2, stride=2)
        self.cbam3 = CBAM(512)
        self.dec3 = ConvBlock(512, fpn_out)
        self.drop3 = nn.Dropout2d(drop)

        # d2：128 + 128 + 128 = 384
        self.up2 = nn.ConvTranspose2d(fpn_out, fpn_out, kernel_size=2, stride=2)
        self.cbam2 = CBAM(384)
        self.dec2 = ConvBlock(384, fpn_out)
        self.drop2 = nn.Dropout2d(drop)

        # d1：128 + 128 + 64 = 320
        self.up1 = nn.ConvTranspose2d(fpn_out, fpn_out, kernel_size=2, stride=2)
        self.cbam1 = CBAM(320)
        self.dec1 = ConvBlock(320, fpn_out)
        self.drop1 = nn.Dropout2d(drop)

        # ---- Heads（主输出 + Deep supervision）----
        self.final_conv = nn.Conv2d(fpn_out, out_ch, kernel_size=1)
        self.aux2_head  = nn.Conv2d(fpn_out, out_ch, kernel_size=1)  # from d2
        self.aux3_head  = nn.Conv2d(fpn_out, out_ch, kernel_size=1)  # from d3

    def forward(self, x):
        B, _, H, W = x.shape

        # ---- Encoder ----
        e1 = self.enc1(x)                    # H
        e2 = self.enc2(self.pool(e1))        # H/2

        e3_in = self.pool(e2)                # H/4
        e1_resized = F.interpolate(self.e1_adapter(e1), size=e3_in.shape[2:], mode='bilinear', align_corners=False)
        e3 = self.enc3(self.fuse3(torch.cat([e3_in, e1_resized], dim=1)))  # H/4

        e4_in = self.pool(e3)                # H/8
        e2_resized = F.interpolate(self.e2_adapter(e2), size=e4_in.shape[2:], mode='bilinear', align_corners=False)
        e4 = self.enc4(self.fuse4(torch.cat([e4_in, e2_resized], dim=1)))  # H/8

        b  = self.bottleneck(self.pool(e4))  # H/16

        # ---- FPN 输出（与 e1..e4 尺寸一致；通道=fpn_out）----
        fpn_feats = self.fpn([e1, e2, e3, e4])   # [H, H/2, H/4, H/8]
        b_aligned = self.b_adapter(b)            # (B,fpn_out,H/16,W/16)

        # ---- Decoder + gates ----
        sig = torch.sigmoid  # 省事

        # d4: H/16 -> H/8
        d4 = self.up4(b_aligned)  # (B,fpn_out,H/8,W/8)
        # 加门控缩放：让网络学每路的贡献
        d4_in = torch.cat([
            d4 * sig(self.g4_b),
            fpn_feats[3] * sig(self.g4_fpn),
            e4 * sig(self.g4_e)
        ], dim=1)
        d4 = self.cbam4(d4_in)
        d4 = self.dec4(d4)
        d4 = self.drop4(d4)

        # d3: H/8 -> H/4
        d3 = self.up3(d4)
        d3_in = torch.cat([
            d3,
            fpn_feats[2] * sig(self.g3_fpn),
            e3 * sig(self.g3_e)
        ], dim=1)
        d3 = self.cbam3(d3_in)
        d3 = self.dec3(d3)
        d3 = self.drop3(d3)

        # d2: H/4 -> H/2
        d2 = self.up2(d3)
        d2_in = torch.cat([
            d2,
            fpn_feats[1] * sig(self.g2_fpn),
            e2 * sig(self.g2_e)
        ], dim=1)
        d2 = self.cbam2(d2_in)
        d2 = self.dec2(d2)
        d2 = self.drop2(d2)

        # d1: H/2 -> H
        d1 = self.up1(d2)
        d1_in = torch.cat([
            d1,
            fpn_feats[0] * sig(self.g1_fpn),
            e1 * sig(self.g1_e)
        ], dim=1)
        d1 = self.cbam1(d1_in)
        d1 = self.dec1(d1)
        d1 = self.drop1(d1)

        # ---- heads ----
        out  = self.final_conv(d1)                # (B,out_ch,H,W)
        # aux2 = self.aux2_head(d2)                 # (B,out_ch,H/2,W/2)
        # aux3 = self.aux3_head(d3)                 # (B,out_ch,H/4,W/4)
        # aux2 = F.interpolate(aux2, size=(H, W), mode="bilinear", align_corners=False)
        # aux3 = F.interpolate(aux3, size=(H, W), mode="bilinear", align_corners=False)

        return out #, aux2, aux3
