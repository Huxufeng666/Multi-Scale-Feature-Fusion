import torch
import torch.nn as nn
import torch.nn.functional as F



# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
#         )
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg = self.fc(self.avg_pool(x))
#         max = self.fc(self.max_pool(x))
#         out = avg + max
#         return self.sigmoid(out)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveMaxPool2d(1)
        # 去掉 max_pool
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x))
        out = avg   # 只用平均池化
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

class FPNUNet_CBAM_Residual(nn.Module):
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

        # return self.final_conv(d1)
    
    
        out_main = self.final_conv(d1)
        
                # --- 辅助输出 (上采样到原图大小) ---
        aux_out2 = F.interpolate(self.aux2(d2), size=x.shape[2:], mode='bilinear', align_corners=True)
        aux_out3 = F.interpolate(self.aux3(d3), size=x.shape[2:], mode='bilinear', align_corners=True)
        aux_out4 = F.interpolate(self.aux4(d4), size=x.shape[2:], mode='bilinear', align_corners=True)
        
        

        
        return out_main, aux_out2, aux_out3, aux_out4 

