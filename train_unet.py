
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.utils as vutils
from PIL import Image
import csv
from  Data  import  get_data,BUS_UCLM_Dataset
from torch.utils.data import  DataLoader

from tools import append_loss_log,dice_loss,dice_loss_per_sample,plot_loss_curve,dice_loss
from Unet_attention import AttentionUNet,UNet,FPNUNet,  FPNUNetV2,FPNUNetV3,\
    FPNUNetV3_CBAM,FPNUNetV3_CBAM_pro,FPNUNetV3_CCBAM,FPNUNetV3_CCBAM,\
        FPNUNetV3_CBAM_Residual_2,FPNUNetV3_CBAM_AttnGate,FPNUNet_Light,FPNUNet_LLight,FPNUNet_Lightt\
            ,FPNUNet_Liight,FPNUNet_Lighht,FPNUNet_ResNetEncoder,FPNUNet_SimpleEncoderFusion\
            ,FPNUNet_Simple_EncoderFusion,FPNUNet_Lighttt,FPNUNet_A_Lightt
#Loading dataset 
from tqdm  import tqdm

from network.model import FPNUNet_CBAM_Residual

from A import Del_FPN_F,Del_CBAM,FPNUNetV3_CBAM_Residual_SUM, FPNUNetV3_CBAM_Residual
import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from network.moldel2 import FPNUNet_CBAMResidual

import os, random
import numpy as np
import torch

def set_seed(seed: int = 42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass

set_seed(2025)
train_data = get_data(image_dir='/home/user/HUXUFENG/UI/Diffusin-net_GAN/BUSI/train/images',
                      mask_dir='/home/user/HUXUFENG/UI/Diffusin-net_GAN/BUSI/train/masks')
val_data = get_data(image_dir='/home/user/HUXUFENG/UI/Diffusin-net_GAN/BUSI/val/images',
                    mask_dir='/home/user/HUXUFENG/UI/Diffusin-net_GAN/BUSI/val/masks')

# # ------------------------
# # 1. 设备设置
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------
# 2. 数据加载
# ------------------------
train_loader = DataLoader(train_data, batch_size=32, drop_last=True, shuffle=False)
val_loader = DataLoader(val_data, batch_size=32, drop_last=True, shuffle=False)

# ------------------------
# 3. 模型、损失、优化器
# ------------------------
model = FPNUNet_CBAMResidual()
model = model.to(device) 
model_name = model.__class__.__name__
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%d_%M")
log_dir = f"results/{model_name}_{timestamp}"
os.makedirs(log_dir, exist_ok=True)


class BCEWithLogitsLossWithSmoothing(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss(reduction='none')  # 不先平均，自己再平均

    def forward(self, logits, targets):
        # smooth 标签：1 -> 1 - ε, 0 -> ε
        targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        loss = self.bce(logits, targets)
        return loss.mean()
    

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        """
        pred: [B, C] raw logits
        target: [B] class indices (int)
        """
        num_classes = pred.size(1)
        confidence = 1.0 - self.smoothing

        # 将 target 转为 one-hot，并加平滑
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (num_classes - 1))
        true_dist.scatter_(1, target.unsqueeze(1), confidence)

        log_probs = F.log_softmax(pred, dim=1)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))


# 或
# bce_loss = LabelSmoothingCrossEntropy(0.1)
bce_loss = BCEWithLogitsLossWithSmoothing(smoothing=0.1)

def dice_loss_per_sample(logits, masks, smooth=1e-6):
    probs = torch.sigmoid(logits)
    B = probs.shape[0]
    probs_flat = probs.view(B, -1)
    masks_flat = masks.view(B, -1)
    inter = (probs_flat * masks_flat).sum(dim=1)
    union = probs_flat.sum(dim=1) + masks_flat.sum(dim=1)
    dice = (2 * inter + smooth) / (union + smooth)
    return 1 - dice

model = nn.DataParallel(model)
optimizer = optim.AdamW(model.parameters(), lr=1e-4,weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8, verbose=True)


log_csv = os.path.join(log_dir, "log_{}_{}.csv".format(model_name, timestamp))
model_path = os.path.join(log_dir, "best_{}_{}.pth".format(model_name, timestamp))
loss_plot_path = os.path.join(log_dir, "loss_plot_{}_{}.png".format(model_name, timestamp))
image_save_template = os.path.join(log_dir, "epoch{}_{}.png".format("{:03d}", model_name))

# 写入日志文件头（自动获取参数）
with open(log_csv, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["# Hyperparameters"])
    writer.writerow(["Model name", model_name])
    writer.writerow(["Time", timestamp])
    writer.writerow(["Batch size", train_loader.batch_size])
    writer.writerow(["Learning rate", optimizer.param_groups[0]['lr']])
    writer.writerow(["Loss function", str(bce_loss.__class__.__name__) + " + " + str(dice_loss_per_sample.__name__)])
    writer.writerow(["Optimizer", type(optimizer).__name__])
    # 如果你使用 scheduler
    writer.writerow(["Scheduler", type(scheduler).__name__])
    writer.writerow([])
    writer.writerow(["epoch", "train_loss", "val_loss", "learning_rate"])
# ------------------------
# 5. 训练循环
# ------------------------
num_epochs = 100
best_val_loss = float("inf")

top_k = 3  # 最多保留 top-k 个模型
saved_models = []  # 存储 (val_loss, path) 对

    
for epoch in range(1, num_epochs + 1):
    model.train()
    total_train_loss = 0.0
    for imgs, masks in tqdm(train_loader, desc=f"[Train] Epoch {epoch}"):
        imgs, masks = imgs.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)

        # 如果模型返回多个输出 (final + auxiliaries)
        if isinstance(outputs, (list, tuple)):
            final_out, *aux_outs = outputs
        else:
            final_out, aux_outs = outputs, []

        # 计算主输出的 loss
        loss_bce = bce_loss(final_out, masks)
        loss_dice = dice_loss_per_sample(final_out, masks).mean()
        main_loss = 0.5 * loss_bce + 0.5 * loss_dice

        # 计算辅助输出的 loss
        aux_loss = 0.0
        for aux in aux_outs:
            aux_bce = bce_loss(aux, masks)
            aux_dice = dice_loss_per_sample(aux, masks).mean()
            aux_loss += 0.5 * aux_bce + 0.5 * aux_dice
        if len(aux_outs) > 0:
            aux_loss /= len(aux_outs)

        # 总损失 = 主损失 + 辅助损失(加权)
        loss = main_loss + 0.4 * aux_loss   # α=0.4 可调

        loss.backward()
        optimizer.step()
        total_train_loss += loss.item() * imgs.size(0)

    avg_train = total_train_loss / len(train_loader.dataset)

    # ---------- 验证 ----------
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc=f"[ Val ] Epoch {epoch}"):
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)

            if isinstance(outputs, (list, tuple)):
                final_out, *aux_outs = outputs
            else:
                final_out, aux_outs = outputs, []

            loss_bce = bce_loss(final_out, masks)
            loss_dice = dice_loss_per_sample(final_out, masks).mean()
            main_loss = 0.5 * loss_bce + 0.5 * loss_dice

            aux_loss = 0.0
            for aux in aux_outs:
                aux_bce = bce_loss(aux, masks)
                aux_dice = dice_loss_per_sample(aux, masks).mean()
                aux_loss += 0.5 * aux_bce + 0.5 * aux_dice
            if len(aux_outs) > 0:
                aux_loss /= len(aux_outs)

            loss = main_loss + 0.4 * aux_loss
            total_val_loss += loss.item() * imgs.size(0)

    avg_val = total_val_loss / len(val_loader.dataset)

 
    print(f"Epoch {epoch:02d} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

    # ---------- 写日志 ----------
    with open(log_csv, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, f"{avg_train:.6f}", f"{avg_val:.6f}", optimizer.param_groups[0]['lr']])

    # ---------- 保存最优权重 ----------
    model_file = os.path.join(log_dir, f"model_epoch{epoch}_{avg_val:.4f}.pth")
    torch.save(model.state_dict(), model_file)
    saved_models.append((avg_val, model_file))
    saved_models.sort(key=lambda x: x[0])  # 按 val loss 升序排序

    # 删除多余的旧模型（只保留 top-k）
    if len(saved_models) > top_k:
        _, to_delete = saved_models.pop(-1)
        if os.path.exists(to_delete):
            os.remove(to_delete)
            print(f"🗑️ Deleted old model: {to_delete}")

    # 更新 best_val_loss（可选）
    if avg_val < best_val_loss:
        best_val_loss = avg_val

    # ---------- 每10轮保存图 ----------
    if epoch % 10 == 0:
        sample_imgs, sample_masks = next(iter(val_loader))
        sample_imgs = sample_imgs.to(device)
        with torch.no_grad():
            sample_logits = model(sample_imgs)
            
            if isinstance(outputs, (list, tuple)):
                final_out, *aux_outs = sample_logits
            else:
                final_out, aux_outs = sample_logits, []

            sample_probs = torch.sigmoid(final_out)
            sample_preds = (sample_probs > 0.5).float()

        sample_masks = sample_masks.to(device)
        composites = []
        for i in range(4):
            img, msk, pred = sample_imgs[i], sample_masks[i], sample_preds[i]
            comp = torch.cat([img, msk, pred], dim=2)
            composites.append(comp)

        grid = torch.stack(composites, dim=0)
        vutils.save_image(grid, image_save_template.format(epoch), nrow=2, normalize=True, scale_each=True)

    plot_loss_curve(log_csv, output_path=loss_plot_path, show_head=False)
    last_model_path = os.path.join(log_dir, "model_last.pth")
    torch.save(model.state_dict(), last_model_path)
    print(f"💾 Saved last model to {last_model_path}")
print(f"✅ Training complete! Best val loss: {best_val_loss:.4f}")