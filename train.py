
# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# import torchvision.utils as vutils
# from PIL import Image
# import csv
# from  Data  import  get_data,BUS_UCLM_Dataset
# from torch.utils.data import  DataLoader

# from tools import append_loss_log,dice_loss,dice_loss_per_sample,plot_loss_curve,dice_loss

# from tqdm  import tqdm

# from network.model import FPNUNet_CBAM_Residual
# import datetime
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# import torch.nn.functional as F

# import os, random
# import numpy as np
# import torch
# CUDA_VISIBLE_DEVICES=0

# # ------------------------
# # 1. 固定随机种子函数
# # ------------------------
# def set_seed(seed: int = 2025):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     # torch.use_deterministic_algorithms(True, warn_only=True)

#     def seed_worker(worker_id):
#         worker_seed = seed + worker_id
#         np.random.seed(worker_seed)
#         random.seed(worker_seed)

#     return seed_worker, torch.Generator().manual_seed(seed)

# # ------------------------
# # 2. 设置随机种子（全局）
# # ------------------------
# seed_worker, g = set_seed(2025)

# # ------------------------
# # 3. 数据集 & DataLoader
# # ------------------------
# train_data = BUS_UCLM_Dataset(
#     image_dir='/home/user/HUXUFENG/UI/Diffusin-net_GAN/BUS-UCLM/train/images',
#     mask_dir='/home/user/HUXUFENG/UI/Diffusin-net_GAN/BUS-UCLM/train/masks'
# )
# val_data = BUS_UCLM_Dataset(
#     image_dir='/home/user/HUXUFENG/UI/Diffusin-net_GAN/BUS-UCLM/val/images',
#     mask_dir='/home/user/HUXUFENG/UI/Diffusin-net_GAN/BUS-UCLM/val/masks'
# )



# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# train_loader = DataLoader(
#     train_data, batch_size=16, drop_last=True, shuffle=True,
#     num_workers=2, worker_init_fn=seed_worker, generator=g
# )
# val_loader = DataLoader(
#     val_data, batch_size=16, drop_last=True, shuffle=False,
#     num_workers=2, worker_init_fn=seed_worker, generator=g
# )

# # ------------------------
# # 4. 模型、损失、优化器
# # ------------------------
# model = FPNUNet_CBAM_Residual()
# model = model.to(device)
# model = nn.DataParallel(model)

# model_name = model.__class__.__name__
# timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M_%S")
# log_dir = f"results/{model_name}_{timestamp}"
# os.makedirs(log_dir, exist_ok=True)

# class BCEWithLogitsLossWithSmoothing(nn.Module):
#     def __init__(self, smoothing=0.1):
#         super().__init__()
#         self.smoothing = smoothing
#         self.bce = nn.BCEWithLogitsLoss(reduction='none')

#     def forward(self, logits, targets):
#         targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
#         loss = self.bce(logits, targets)
#         return loss.mean()

# def dice_loss_per_sample(logits, masks, smooth=1e-6):
#     probs = torch.sigmoid(logits)
#     B = probs.shape[0]
#     probs_flat = probs.view(B, -1)
#     masks_flat = masks.view(B, -1)
#     inter = (probs_flat * masks_flat).sum(dim=1)
#     union = probs_flat.sum(dim=1) + masks_flat.sum(dim=1)
#     dice = (2 * inter + smooth) / (union + smooth)
#     return 1 - dice

# bce_loss = BCEWithLogitsLossWithSmoothing(smoothing=0.1)
# optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8, verbose=True)

# # ------------------------
# # 5. 日志文件
# # ------------------------
# log_csv = os.path.join(log_dir, f"log_{model_name}_{timestamp}.csv")
# model_path = os.path.join(log_dir, f"best_{model_name}_{timestamp}.pth")
# loss_plot_path = os.path.join(log_dir, f"loss_plot_{model_name}_{timestamp}.png")
# image_save_template = os.path.join(log_dir, "epoch{}_{}.png".format("{:03d}", model_name))

# with open(log_csv, mode="w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["# Hyperparameters"])
#     writer.writerow(["Model name", model_name])
#     writer.writerow(["Time", timestamp])
#     writer.writerow(["Batch size", train_loader.batch_size])
#     writer.writerow(["Learning rate", optimizer.param_groups[0]['lr']])
#     writer.writerow(["Loss function", "BCEWithLogitsLossWithSmoothing + dice_loss_per_sample"])
#     writer.writerow(["Optimizer", type(optimizer).__name__])
#     writer.writerow(["Scheduler", type(scheduler).__name__])
#     writer.writerow([])
#     writer.writerow(["epoch", "train_loss", "val_loss", "learning_rate"])

# # ------------------------
# # 6. 训练循环
# # ------------------------
# num_epochs = 100
# best_val_loss = float("inf")
# top_k = 3
# saved_models = []

# for epoch in range(1, num_epochs + 1):
#     model.train()
#     total_train_loss = 0.0
#     for imgs, masks in tqdm(train_loader, desc=f"[Train] Epoch {epoch}"):
#         imgs, masks = imgs.to(device), masks.to(device)
#         optimizer.zero_grad()
#         outputs = model(imgs)

#         if isinstance(outputs, (list, tuple)):
#             final_out, *aux_outs = outputs
#         else:
#             final_out, aux_outs = outputs, []

#         loss_bce = bce_loss(final_out, masks)
#         loss_dice = dice_loss_per_sample(final_out, masks).mean()
#         main_loss = 0.5 * loss_bce + 0.5 * loss_dice

#         aux_loss = 0.0
#         for aux in aux_outs:
#             aux_bce = bce_loss(aux, masks)
#             aux_dice = dice_loss_per_sample(aux, masks).mean()
#             aux_loss += 0.5 * aux_bce + 0.5 * aux_dice
#         if len(aux_outs) > 0:
#             aux_loss /= len(aux_outs)

#         loss = main_loss + 0.4 * aux_loss
#         loss.backward()
#         optimizer.step()
#         total_train_loss += loss.item()* imgs.size(0)
        
        
#     avg_train = total_train_loss / len(train_loader.dataset)
#     # ---------- 验证 ----------
#     model.eval()
#     total_val_loss = 0.0
#     with torch.no_grad():
#         for imgs, masks in tqdm(val_loader, desc=f"[ Val ] Epoch {epoch}"):
#             imgs, masks = imgs.to(device), masks.to(device)
#             outputs = model(imgs)

#             if isinstance(outputs, (list, tuple)):
#                 final_out, *aux_outs = outputs
#             else:
#                 final_out, aux_outs = outputs, []

#             loss_bce = bce_loss(final_out, masks)
#             loss_dice = dice_loss_per_sample(final_out, masks).mean()
#             main_loss = 0.5 * loss_bce + 0.5 * loss_dice

#             aux_loss = 0.0
#             for aux in aux_outs:
#                 aux_bce = bce_loss(aux, masks)
#                 aux_dice = dice_loss_per_sample(aux, masks).mean()
#                 aux_loss += 0.5 * aux_bce + 0.5 * aux_dice
#             if len(aux_outs) > 0:
#                 aux_loss /= len(aux_outs)

#             loss = main_loss + 0.4 * aux_loss
#             total_val_loss += loss.item() * imgs.size(0)

#     avg_val = total_val_loss / len(val_loader.dataset)
#     print(f"Epoch {epoch:02d} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

#     # ---------- 写日志 ----------
#     with open(log_csv, mode="a", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow([epoch, f"{avg_train:.6f}", f"{avg_val:.6f}", optimizer.param_groups[0]['lr']])

#     # ---------- 保存模型 ----------
#     model_file = os.path.join(log_dir, f"model_epoch{epoch}_{avg_val:.4f}.pth")
#     torch.save(model.state_dict(), model_file)
#     saved_models.append((avg_val, model_file))
#     saved_models.sort(key=lambda x: x[0])

#     if len(saved_models) > top_k:
#         _, to_delete = saved_models.pop(-1)
#         if os.path.exists(to_delete):
#             os.remove(to_delete)
#             print(f"🗑️ Deleted old model: {to_delete}")

#     if avg_val < best_val_loss:
#         best_val_loss = avg_val

#     # ---------- 可视化 ----------
#     if epoch % 10 == 0:
#         sample_imgs, sample_masks = next(iter(val_loader))
#         sample_imgs = sample_imgs.to(device)
#         with torch.no_grad():
#             sample_logits = model(sample_imgs)
#             if isinstance(sample_logits, (list, tuple)):
#                 final_out, *aux_outs = sample_logits
#             else:
#                 final_out, aux_outs = sample_logits, []
#             sample_probs = torch.sigmoid(final_out)
#             sample_preds = (sample_probs > 0.5).float()

#         sample_masks = sample_masks.to(device)
#         composites = []
#         for i in range(min(4, sample_imgs.size(0))):
#             img, msk, pred = sample_imgs[i], sample_masks[i], sample_preds[i]
#             comp = torch.cat([img, msk, pred], dim=2)
#             composites.append(comp)

#         grid = torch.stack(composites, dim=0)
#         vutils.save_image(grid, image_save_template.format(epoch), nrow=2, normalize=True, scale_each=True)

#     last_model_path = os.path.join(log_dir, "model_last.pth")
#     torch.save(model.state_dict(), last_model_path)
#     print(f"💾 Saved last model to {last_model_path}")

# print(f"✅ Training complete! Best val loss: {best_val_loss:.4f}")






# import os
# import random
# import datetime
# import csv
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# from tqdm import tqdm
# import torchvision.utils as vutils
# from statistics import mean, stdev

# # ========================
# # 1. 半可复现随机种子
# # ========================
# def set_seed(seed: int = 2025):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

#     # 允许非确定性算子（性能更好）
#     torch.backends.cudnn.benchmark = True
#     torch.backends.cudnn.deterministic = False
#     torch.use_deterministic_algorithms(False)

#     def seed_worker(worker_id):
#         worker_seed = seed + worker_id
#         np.random.seed(worker_seed)
#         random.seed(worker_seed)

#     return seed_worker, torch.Generator().manual_seed(seed)


# # ========================
# # 2. Loss 函数
# # ========================
# class BCEWithLogitsLossWithSmoothing(nn.Module):
#     def __init__(self, smoothing=0.1):
#         super().__init__()
#         self.smoothing = smoothing
#         self.bce = nn.BCEWithLogitsLoss(reduction='none')

#     def forward(self, logits, targets):
#         targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
#         loss = self.bce(logits, targets)
#         return loss.mean()


# def dice_loss_per_sample(logits, masks, smooth=1e-6):
#     probs = torch.sigmoid(logits)
#     B = probs.shape[0]
#     probs_flat = probs.view(B, -1)
#     masks_flat = masks.view(B, -1)
#     inter = (probs_flat * masks_flat).sum(dim=1)
#     union = probs_flat.sum(dim=1) + masks_flat.sum(dim=1)
#     dice = (2 * inter + smooth) / (union + smooth)
#     return 1 - dice


# # ========================
# # 3. 训练函数（单个 seed）
# # ========================
# def train_one_seed(seed, num_epochs=100, batch_size=32):
#     print(f"\n=== Running seed {seed} ===")
#     seed_worker, g = set_seed(seed)

#     # 数据集
#     train_data = BUS_UCLM_Dataset(
#         image_dir='/home/user/HUXUFENG/UI/Diffusin-net_GAN/BUS-UCLM/train/images',
#         mask_dir='/home/user/HUXUFENG/UI/Diffusin-net_GAN/BUS-UCLM/train/masks'
#     )
#     val_data = BUS_UCLM_Dataset(
#         image_dir='/home/user/HUXUFENG/UI/Diffusin-net_GAN/BUS-UCLM/val/images',
#         mask_dir='/home/user/HUXUFENG/UI/Diffusin-net_GAN/BUS-UCLM/val/masks'
#     )

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     train_loader = DataLoader(
#         train_data, batch_size=batch_size, drop_last=True, shuffle=True,
#         num_workers=2, worker_init_fn=seed_worker, generator=g
#     )
#     val_loader = DataLoader(
#         val_data, batch_size=batch_size, drop_last=True, shuffle=False,
#         num_workers=2, worker_init_fn=seed_worker, generator=g
#     )

#     # 模型
#     model = FPNUNet_CBAM_Residual().to(device)
#     model = nn.DataParallel(model)
#     model_name = model.__class__.__name__
#     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M_%S")
#     log_dir = f"results/{model_name}_{timestamp}_seed{seed}"
#     os.makedirs(log_dir, exist_ok=True)

#     # 优化器 & loss
#     bce_loss = BCEWithLogitsLossWithSmoothing(smoothing=0.1)
#     optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
#     scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8, verbose=True)

#     # 日志文件
#     log_csv = os.path.join(log_dir, f"log_{model_name}_{timestamp}.csv")
#     with open(log_csv, mode="w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["# Hyperparameters"])
#         writer.writerow(["Model name", model_name])
#         writer.writerow(["Time", timestamp])
#         writer.writerow(["Seed", seed])
#         writer.writerow(["Batch size", train_loader.batch_size])
#         writer.writerow(["Learning rate", optimizer.param_groups[0]['lr']])
#         writer.writerow(["Loss function", "BCEWithLogitsLossWithSmoothing + dice_loss_per_sample"])
#         writer.writerow(["Optimizer", type(optimizer).__name__])
#         writer.writerow(["Scheduler", type(scheduler).__name__])
#         writer.writerow([])
#         writer.writerow(["epoch", "train_loss", "val_loss", "learning_rate"])

#     # ------------------------
#     # 训练循环
#     # ------------------------
#     best_val_loss = float("inf")
#     top_k = 3
#     saved_models = []

#     for epoch in range(1, num_epochs + 1):
#         model.train()
#         total_train_loss = 0.0
#         for imgs, masks in tqdm(train_loader, desc=f"[Train] Epoch {epoch}"):
#             imgs, masks = imgs.to(device), masks.to(device)
#             optimizer.zero_grad()
#             outputs = model(imgs)

#             if isinstance(outputs, (list, tuple)):
#                 final_out, *aux_outs = outputs
#             else:
#                 final_out, aux_outs = outputs, []

#             loss_bce = bce_loss(final_out, masks)
#             loss_dice = dice_loss_per_sample(final_out, masks).mean()
#             main_loss = 0.5 * loss_bce + 0.5 * loss_dice

#             aux_loss = 0.0
#             for aux in aux_outs:
#                 aux_bce = bce_loss(aux, masks)
#                 aux_dice = dice_loss_per_sample(aux, masks).mean()
#                 aux_loss += 0.5 * aux_bce + 0.5 * aux_dice
#             if len(aux_outs) > 0:
#                 aux_loss /= len(aux_outs)

#             loss = main_loss + 0.4 * aux_loss
#             loss.backward()
#             optimizer.step()
#             total_train_loss += loss.item() * imgs.size(0)

#         avg_train = total_train_loss / len(train_loader.dataset)

#         # 验证
#         model.eval()
#         total_val_loss = 0.0
#         with torch.no_grad():
#             for imgs, masks in tqdm(val_loader, desc=f"[ Val ] Epoch {epoch}"):
#                 imgs, masks = imgs.to(device), masks.to(device)
#                 outputs = model(imgs)

#                 if isinstance(outputs, (list, tuple)):
#                     final_out, *aux_outs = outputs
#                 else:
#                     final_out, aux_outs = outputs, []

#                 loss_bce = bce_loss(final_out, masks)
#                 loss_dice = dice_loss_per_sample(final_out, masks).mean()
#                 main_loss = 0.5 * loss_bce + 0.5 * loss_dice

#                 aux_loss = 0.0
#                 for aux in aux_outs:
#                     aux_bce = bce_loss(aux, masks)
#                     aux_dice = dice_loss_per_sample(aux, masks).mean()
#                     aux_loss += 0.5 * aux_bce + 0.5 * aux_dice
#                 if len(aux_outs) > 0:
#                     aux_loss /= len(aux_outs)

#                 loss = main_loss + 0.4 * aux_loss
#                 total_val_loss += loss.item() * imgs.size(0)

#         avg_val = total_val_loss / len(val_loader.dataset)
#         print(f"Epoch {epoch:02d} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

#         with open(log_csv, mode="a", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow([epoch, f"{avg_train:.6f}", f"{avg_val:.6f}", optimizer.param_groups[0]['lr']])

#         # 保存模型
#         model_file = os.path.join(log_dir, f"model_epoch{epoch}_{avg_val:.4f}.pth")
#         torch.save(model.state_dict(), model_file)
#         saved_models.append((avg_val, model_file))
#         saved_models.sort(key=lambda x: x[0])

#         if len(saved_models) > top_k:
#             _, to_delete = saved_models.pop(-1)
#             if os.path.exists(to_delete):
#                 os.remove(to_delete)
#                 print(f"🗑️ Deleted old model: {to_delete}")

#         if avg_val < best_val_loss:
#             best_val_loss = avg_val

#         last_model_path = os.path.join(log_dir, "model_last.pth")
#         torch.save(model.state_dict(), last_model_path)

#     print(f"✅ Training complete! Best val loss (seed {seed}): {best_val_loss:.4f}")
#     return best_val_loss


# # ========================
# # 4. 多 seed 运行
# # ========================
# def run_multiple_seeds(seeds=[2025, 2026, 2027], epochs=100):
#     results = []
#     for s in seeds:
#         best_val = train_one_seed(seed=s, num_epochs=epochs)
#         results.append(best_val)

#     mean_val = mean(results)
#     std_val = stdev(results) if len(results) > 1 else 0.0
#     print("\n=== Summary ===")
#     for s, r in zip(seeds, results):
#         print(f"Seed {s}: {r:.4f}")
#     print(f"Ensemble Mean ± Std: {mean_val:.4f} ± {std_val:.4f}")


# if __name__ == "__main__":
#     run_multiple_seeds(seeds=[2025, 2026, 2027], epochs=100)

import os
import random
import numpy as np
import datetime
import csv
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 自己的模块
from Data import BUS_UCLM_Dataset,get_data
from tools import plot_loss_curve
from network.model import FPNUNet_CBAM_Residual

# ==================================================
# 1. 固定随机种子（完全可复现）
# ==================================================
def set_seed(seed: int = 2025):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 确定性设置
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True,warn_only=True)

    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return seed_worker, torch.Generator().manual_seed(seed)


# ==================================================
# 2. 数据集 & DataLoader
# ==================================================
set_seed(2025)

train_data = get_data(
    image_dir='/home/user/HUXUFENG/UI/Diffusin-net_GAN/BUSI/train/images',
    mask_dir='/home/user/HUXUFENG/UI/Diffusin-net_GAN/BUSI/train/masks'
)
val_data = get_data(
    image_dir='/home/user/HUXUFENG/UI/Diffusin-net_GAN/BUSI/val/images',
    mask_dir='/home/user/HUXUFENG/UI/Diffusin-net_GAN/BUSI/val/masks'
)

seed_worker, g = set_seed(2025)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_loader = DataLoader(
    train_data, batch_size=32, drop_last=True, shuffle=True,
    num_workers=2, worker_init_fn=seed_worker, generator=g
)

val_loader = DataLoader(
    val_data, batch_size=32, drop_last=False, shuffle=False,
    num_workers=2, worker_init_fn=seed_worker, generator=g
)


# ==================================================
# 3. 模型、损失、优化器
# ==================================================
model = FPNUNet_CBAM_Residual().to(device)
model = nn.DataParallel(model)

model_name = model.__class__.__name__
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"results/{model_name}_{timestamp}_BUSI"
os.makedirs(log_dir, exist_ok=True)


class BCEWithLogitsLossWithSmoothing(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(logits, targets).mean()


def dice_loss_per_sample(logits, masks, smooth=1e-6):
    probs = torch.sigmoid(logits)
    B = probs.shape[0]
    probs_flat = probs.view(B, -1)
    masks_flat = masks.view(B, -1)
    inter = (probs_flat * masks_flat).sum(dim=1)
    union = probs_flat.sum(dim=1) + masks_flat.sum(dim=1)
    dice = (2 * inter + smooth) / (union + smooth)
    return 1 - dice


bce_loss = BCEWithLogitsLossWithSmoothing(smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, verbose=True)


# ==================================================
# 4. 日志文件
# ==================================================
log_csv = os.path.join(log_dir, f"log_{model_name}_{timestamp}.csv")
model_path = os.path.join(log_dir, f"best_{model_name}_{timestamp}.pth")
loss_plot_path = os.path.join(log_dir, f"loss_plot_{model_name}_{timestamp}.png")
image_save_template = os.path.join(log_dir, "epoch{}_{}.png".format("{:03d}", model_name))

with open(log_csv, mode="w", newline="") as f:
    
    writer = csv.writer(f)
    writer.writerow(["# Hyperparameters"])
    writer.writerow(["Model name", model_name])
    writer.writerow(["Time", timestamp])
    writer.writerow(["Batch size", train_loader.batch_size])
    writer.writerow(["Learning rate", optimizer.param_groups[0]['lr']])
    writer.writerow(["Loss function", "BCEWithLogitsLossWithSmoothing + dice_loss_per_sample"])
    writer.writerow(["Optimizer", type(optimizer).__name__])
    writer.writerow(["Scheduler", type(scheduler).__name__])
    writer.writerow([])
    writer.writerow(["epoch", "train_loss", "val_loss", "learning_rate"])


# ==================================================
# 5. 训练循环
# ==================================================
num_epochs = 100
best_val_loss = float("inf")

top_k = 3
saved_models = []

for epoch in range(1, num_epochs + 1):
    # ----------- 训练 -----------
    model.train()
    total_train_loss = 0.0
    for imgs, masks in tqdm(train_loader, desc=f"[Train] Epoch {epoch}"):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
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
   
        # if len(aux_outs) > 0:
        #     aux_losses = []
        #     for aux in aux_outs:
        #         aux_bce = bce_loss(aux, masks)
        #         aux_dice = dice_loss_per_sample(aux, masks).mean()
        #         aux_losses.append(0.5 * aux_bce + 0.5 * aux_dice)
        #     aux_loss = torch.stack(aux_losses).max()  # 用最难的分支
        # else:
        #     aux_loss = 0.0
                 
        # alpha = min(0.4, epoch / 50 * 0.4) 
        alpha = min(0.4, max(0.0, (epoch - 20) / 50 * 0.4))
        if epoch < 40:
            loss = main_loss
        else:
            loss = main_loss + alpha * aux_loss
        
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item() * imgs.size(0)

    avg_train = total_train_loss / len(train_loader.dataset)

    # ----------- 验证 -----------
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

    # 写日志
    with open(log_csv, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, f"{avg_train:.6f}", f"{avg_val:.6f}", optimizer.param_groups[0]['lr']])

    # 保存最优模型
    model_file = os.path.join(log_dir, f"model_epoch{epoch}_{avg_val:.4f}.pth")
    torch.save(model.state_dict(), model_file)
    saved_models.append((avg_val, model_file))
    saved_models.sort(key=lambda x: x[0])

    if len(saved_models) > top_k:
        _, to_delete = saved_models.pop(-1)
        if os.path.exists(to_delete):
            os.remove(to_delete)
            print(f"🗑️ Deleted old model: {to_delete}")

    if avg_val < best_val_loss:
        best_val_loss = avg_val

    # 每10轮保存可视化图
    if epoch % 10 == 0:
        sample_imgs, sample_masks = next(iter(val_loader))
        sample_imgs = sample_imgs.to(device)
        with torch.no_grad():
            sample_logits = model(sample_imgs)
            if isinstance(sample_logits, (list, tuple)):
                final_out, *aux_outs = sample_logits
            else:
                final_out, aux_outs = sample_logits, []
            sample_probs = torch.sigmoid(final_out)
            sample_preds = (sample_probs > 0.5).float()

        sample_masks = sample_masks.to(device)
        composites = []
        for i in range(min(4, sample_imgs.size(0))):
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
