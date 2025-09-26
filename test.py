

# import os
# import random
# import numpy as np
# import datetime
# import csv
# from tqdm import tqdm
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import torchvision.utils as vutils
# from torch.utils.data import DataLoader
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# from thop import profile   # FLOPs 计算
# from network.moldel2 import FPNUNet_CBAMResidual   # 你的模型
# from Data import get_data                          # 数据加载
# from tools import plot_loss_curve


# # ==================================================
# # 1. 固定随机种子（完全可复现）
# # ==================================================
# def set_seed(seed: int = 2025):
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     torch.use_deterministic_algorithms(True, warn_only=True)

#     def seed_worker(worker_id):
#         worker_seed = seed + worker_id
#         np.random.seed(worker_seed)
#         random.seed(worker_seed)

#     return seed_worker, torch.Generator().manual_seed(seed)


# # ==================================================
# # 2. 参数统计函数
# # ==================================================
# def num_of_param(model):
#     print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     total_params = sum(p.numel() for p in model.parameters())
#     notice_param = f"Trainable: {trainable_params:,} / Total: {total_params:,}"
#     print(notice_param)
#     return notice_param


# # ==================================================
# # 3. 数据集 & DataLoader
# # ==================================================
# set_seed(2025)

# train_data = get_data(
#     image_dir='/home/user/HUXUFENG/UI/Diffusin-net_GAN/BUSI/train/images',
#     mask_dir='/home/user/HUXUFENG/UI/Diffusin-net_GAN/BUSI/train/masks'
# )
# val_data = get_data(
#     image_dir='/home/user/HUXUFENG/UI/Diffusin-net_GAN/BUSI/val/images',
#     mask_dir='/home/user/HUXUFENG/UI/Diffusin-net_GAN/BUSI/val/masks'
# )

# seed_worker, g = set_seed(2025)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# train_loader = DataLoader(
#     train_data, batch_size=32, drop_last=True, shuffle=True,
#     num_workers=2, worker_init_fn=seed_worker, generator=g
# )
# val_loader = DataLoader(
#     val_data, batch_size=32, drop_last=False, shuffle=False,
#     num_workers=2, worker_init_fn=seed_worker, generator=g
# )


# # ==================================================
# # 4. 模型、profile、DataParallel
# # ==================================================
# # Step 1: 先建立单卡模型
# single_model = FPNUNet_CBAMResidual().to(device)

# # Step 2: profile FLOPs / Params / 推理速度
# dummy_input = torch.randn(1, 1, 256, 256).to(device)   # BUSI 数据是 1 通道
# flops, params = profile(single_model, inputs=(dummy_input,))
# single_model = nn.DataParallel(single_model)

# print(f"GFLOPS: {flops/1e9:.3f}")
# print(f"Params: {params:,}")
# num_of_param(single_model)

# starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
# repeats = 50
# starter.record()
# with torch.no_grad():
#     for _ in range(repeats):
#         _ = single_model(dummy_input)
# ender.record()
# torch.cuda.synchronize()
# elapsed_ms = starter.elapsed_time(ender) / repeats
# print(f"데이터당 추론 시간: {elapsed_ms:.3f} ms")

# # Step 3: 切换成 DataParallel (多卡训练)
# model = nn.DataParallel(single_model)


# # ==================================================
# # 5. Loss 函数
# # ==================================================
# class BCEWithLogitsLossWithSmoothing(nn.Module):
#     def __init__(self, smoothing=0.1):
#         super().__init__()
#         self.smoothing = smoothing
#         self.bce = nn.BCEWithLogitsLoss(reduction='none')

#     def forward(self, logits, targets):
#         targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
#         return self.bce(logits, targets).mean()


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
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, verbose=True)


# # ==================================================
# # 6. 日志文件
# # ==================================================
# model_name = model.__class__.__name__
# timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# log_dir = f"results/{model_name}_{timestamp}_BUSI"
# os.makedirs(log_dir, exist_ok=True)

# log_csv = os.path.join(log_dir, f"log_{model_name}_{timestamp}.csv")
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
#     writer.writerow(["GFLOPS", f"{flops/1e9:.3f}"])
#     writer.writerow(["Params", f"{params:,}"])
#     writer.writerow(["Inference Time(ms)", f"{elapsed_ms:.3f}"])
#     writer.writerow([])
#     writer.writerow(["epoch", "train_loss", "val_loss", "learning_rate"])


# # ==================================================
# # 7. 训练循环
# # ==================================================
# num_epochs = 100
# best_val_loss = float("inf")
# top_k = 3
# saved_models = []

# for epoch in range(1, num_epochs + 1):
#     # ----------- 训练 -----------
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

#         alpha = min(0.4, max(0.0, (epoch - 20) / 50 * 0.4))
#         if epoch < 40:
#             loss = main_loss
#         else:
#             loss = main_loss + alpha * aux_loss

#         loss.backward()
#         optimizer.step()
#         total_train_loss += loss.item() * imgs.size(0)

#     avg_train = total_train_loss / len(train_loader.dataset)

#     # ----------- 验证 -----------
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

#     with open(log_csv, mode="a", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow([epoch, f"{avg_train:.6f}", f"{avg_val:.6f}", optimizer.param_groups[0]['lr']])

#     # 保存最优模型
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

#     # 每10轮保存可视化图
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
#         vutils.save_image(grid, os.path.join(log_dir, f"epoch{epoch:03d}_{model_name}.png"),
#                           nrow=2, normalize=True, scale_each=True)

#     plot_loss_curve(log_csv, output_path=os.path.join(log_dir, f"loss_plot_{model_name}.png"), show_head=False)
#     last_model_path = os.path.join(log_dir, "model_last.pth")
#     torch.save(model.state_dict(), last_model_path)
#     print(f"💾 Saved last model to {last_model_path}")

# print(f"✅ Training complete! Best val loss: {best_val_loss:.4f}")



import os
import time
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

# thop（GFLOPs/Params）
try:
    from thop import profile
    _HAS_THOP = True
except Exception:
    profile = None
    _HAS_THOP = False

# 你的工程内模块
from network.moldel2 import FPNUNet_CBAMResidual   # 模型
from Data import GetData                          # 数据加载
from tools import plot_loss_curve                  # 画 loss 曲线


# ================================
# 1) 固定随机种子（可复现）
# ================================
def set_seed(seed: int = 2025):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return seed_worker, torch.Generator().manual_seed(seed)


# ================================
# 2) 参数统计
# ================================
def num_of_param(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total:,}")
    print(f"Trainable: {trainable:,} / Total: {total:,}")
    return total, trainable


# ================================
# 3) 推理时间测试（单样本）
# ================================
def measure_infer_latency(model: nn.Module, input_tensor: torch.Tensor, repeats=30, warmup=5):
    """返回单样本平均推理毫秒(ms)。input_tensor 已在正确 device 上。"""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        # 预热
        for _ in range(warmup):
            _ = model(input_tensor)
        # 计时
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()
            for _ in range(repeats):
                _ = model(input_tensor)
            ender.record()
            torch.cuda.synchronize()
            elapsed_ms = starter.elapsed_time(ender) / repeats
        else:
            t0 = time.perf_counter()
            for _ in range(repeats):
                _ = model(input_tensor)
            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000.0 / repeats
    if was_training:
        model.train()
    return elapsed_ms


# ================================
# 4) 设备一致性工具（定位 & 兜底搬家）
# ================================
def dump_off_device(module: nn.Module, device: torch.device):
    bad = False
    for n, p in module.named_parameters():
        if p.device != device:
            print(f"[Param on {p.device}] {n}")
            bad = True
    for n, b in module.named_buffers():
        if b.device != device:
            print(f"[Buffer on {b.device}] {n}")
            bad = True
    if not bad:
        print("✓ All params/buffers are on", device)
    return bad

def force_move_to_device(module: nn.Module, device: torch.device):
    for sub in module.modules():
        # 参数
        for name, p in list(sub._parameters.items()):
            if p is not None and p.device != device:
                sub._parameters[name] = torch.nn.Parameter(p.data.to(device), requires_grad=p.requires_grad)
        # 缓冲（如 BN 的 running_mean/var）
        for name, b in list(sub._buffers.items()):
            if b is not None and b.device != device:
                sub._buffers[name] = b.to(device)


# ================================
# 5) Loss
# ================================
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


# ================================
# 6) 主流程
# ================================
if __name__ == "__main__":
    # —— 环境 & 数据 ——
    seed_worker, g = set_seed(2025)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_data = GetData(
        image_dir='/home/user/HUXUFENG/UI/Diffusin-net_GAN/BUSI/train/images',
        mask_dir='/home/user/HUXUFENG/UI/Diffusin-net_GAN/BUSI/train/masks'
    )
    val_data = GetData(
        image_dir='/home/user/HUXUFENG/UI/Diffusin-net_GAN/BUSI/val/images',
        mask_dir='/home/user/HUXUFENG/UI/Diffusin-net_GAN/BUSI/val/masks'
    )

    train_loader = DataLoader(
        train_data, batch_size=32, drop_last=True, shuffle=True,
        num_workers=2, worker_init_fn=seed_worker, generator=g
    )
    val_loader = DataLoader(
        val_data, batch_size=32, drop_last=False, shuffle=False,
        num_workers=2, worker_init_fn=seed_worker, generator=g
    )

    # —— 单卡实例化（用于 profile） ——
    single_model = FPNUNet_CBAMResidual().to(device)

    print("\n[Check BEFORE DP]")
    dump_off_device(single_model, device)
    force_move_to_device(single_model, device)
    dump_off_device(single_model, device)

    # —— 计算 GFLOPs / Params / 单样本推理时间（baseline） ——
    dummy_input = torch.randn(1, 1, 256, 256, device=device)  # 按你的数据通道 & 分辨率
    if _HAS_THOP:
        try:
            flops, params = profile(single_model, inputs=(dummy_input,))
        except Exception as e:
            print(f"[WARN] thop.profile 失败：{e}")
            flops, params = None, None
    else:
        flops, params = None, None

    total_params, trainable_params = num_of_param(single_model)
    base_infer_ms = measure_infer_latency(single_model, dummy_input, repeats=50, warmup=10)

    # —— 打印三行 —— 
    print(f"GFLOPS: {flops/1e9:.3f}" if flops is not None else "GFLOPS: N/A")
    print(f"Params: {params:,}" if params is not None else f"Params: {total_params:,}")
    print(f"데이터당 추론 시간: {base_infer_ms:.3f} ms")

    # —— 切换多卡训练（锁定主卡为 cuda:0） ——
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        device_ids = list(range(n_gpus)) if n_gpus > 1 else [0]
        model = nn.DataParallel(single_model, device_ids=device_ids, output_device=device_ids[0]).to(device)
    else:
        model = single_model  # CPU fallback

    print("\n[Check AFTER DP] (model.module if DP)")
    mcheck = model.module if isinstance(model, nn.DataParallel) else model
    dump_off_device(mcheck, device)
    force_move_to_device(mcheck, device)
    dump_off_device(mcheck, device)

    # —— 优化器 / 调度器 ——
    bce_loss = BCEWithLogitsLossWithSmoothing(smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, verbose=True)

    # —— 日志 & 输出目录 ——
    model_name = mcheck.__class__.__name__   # 用真实模型名，不用 DataParallel
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"results/{model_name}_{timestamp}_BUSI456"
    os.makedirs(log_dir, exist_ok=True)

    log_csv = os.path.join(log_dir, f"log_{model_name}_{timestamp}.csv")
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
        writer.writerow(["GFLOPS", f"{flops/1e9:.3f}" if flops is not None else "N/A"])
        writer.writerow(["Params", f"{params:,}" if params is not None else f"{total_params:,}"])
        writer.writerow(["Inference Time(ms)", f"{base_infer_ms:.3f}"])
        writer.writerow(["Trainable Params", f"{trainable_params:,}"])
        writer.writerow([])
        # 每个 epoch 的列（每 5 轮测一次推理；其余行留空）
        writer.writerow(["epoch", "train_loss", "val_loss", "learning_rate", "infer_ms(single)", "throughput(img/s)"])

    # ================================
    # 7) 训练循环
    # ================================
    num_epochs = 100
    best_val_loss = float("inf")
    top_k = 3
    saved_models = []

    for epoch in range(1, num_epochs + 1):
        # ---- Train ----
        model.train()
        total_train_loss = 0.0
        for imgs, masks in tqdm(train_loader, desc=f"[Train] Epoch {epoch}"):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)  # 千万不要再 .to(device)

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

            alpha = min(0.4, max(0.0, (epoch - 20) / 50 * 0.4))
            loss = main_loss if epoch < 40 else (main_loss + alpha * aux_loss)

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * imgs.size(0)

        avg_train = total_train_loss / len(train_loader.dataset)

        # ---- Val ----
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
        scheduler.step(avg_val)

        print(f"Epoch {epoch:02d} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

        # —— 每 5 轮测单样本推理时间（吞吐=1000/ms），写入本行；其他行留空 —— 
        if epoch % 5 == 0:
            sample_imgs, _ = next(iter(val_loader))
            single = sample_imgs[:1].to(device)
            infer_ms_epoch = measure_infer_latency(model, single, repeats=30, warmup=5)
            thr = 1000.0 / infer_ms_epoch
            infer_ms_str = f"{infer_ms_epoch:.3f}"
            thr_str = f"{thr:.1f}"
        else:
            infer_ms_str = ""
            thr_str = ""

        with open(log_csv, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{avg_train:.6f}",
                f"{avg_val:.6f}",
                optimizer.param_groups[0]['lr'],
                infer_ms_str,
                thr_str
            ])

        # —— 保存最优 top-k & last —— 
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

        # —— 可视化（每10轮） ——
        if epoch % 10 == 0:
            sample_imgs, sample_masks = next(iter(val_loader))
            sample_imgs = sample_imgs.to(device)
            with torch.no_grad():
                sample_logits = model(sample_imgs)
                if isinstance(sample_logits, (list, tuple)):
                    final_out, *_ = sample_logits
                else:
                    final_out = sample_logits
                sample_probs = torch.sigmoid(final_out)
                sample_preds = (sample_probs > 0.5).float()

            sample_masks = sample_masks.to(device)
            composites = []
            for i in range(min(4, sample_imgs.size(0))):
                img, msk, pred = sample_imgs[i], sample_masks[i], sample_preds[i]
                comp = torch.cat([img, msk, pred], dim=2)
                composites.append(comp)

            grid = torch.stack(composites, dim=0)
            vutils.save_image(
                grid,
                os.path.join(log_dir, f"epoch{epoch:03d}_{model_name}.png"),
                nrow=2, normalize=True, scale_each=True
            )

        plot_loss_curve(log_csv, output_path=os.path.join(log_dir, f"loss_plot_{model_name}.png"), show_head=False)
        last_model_path = os.path.join(log_dir, "model_last.pth")
        torch.save(model.state_dict(), last_model_path)
        print(f"💾 Saved last model to {last_model_path}")

    print(f"✅ Training complete! Best val loss: {best_val_loss:.4f}")
