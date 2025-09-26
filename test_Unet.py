import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import numpy as np
from Data import get_data,BUS_UCLM_Dataset
from Unet_attention import AttentionUNet,UNet,FPNUNet, FPNUNet_LLight ,FPNUNetV2,FPNUNetV3\
    ,UNet,FPNUNetV3_CBAM,FPNUNetV3_CBAM_pro,FPNUNetV3_CCBAM,FPNUNetV3_CBAM_Residual_2\
        ,FPNUNetV3_CBAM_AttnGate,FPNUNet_Light,FPNUNet_Lightt,FPNUNet_Liight,FPNUNet_Lighht\
            ,FPNUNet_ResNetEncoder,FPNUNet_SimpleEncoderFusion,Del_Res,FPNUNet_A_Lightt
            
from glob import glob        

# from A import FPNUNetV3_CBAM_Residual_7,FPNUNet_Ligh_7,Del_FPN_F,\
#     Del_CBAM,FPNUNetV3_CBAM_Residual_SUM,FPNUNetV3_CBAM_Residual


from medpy.metric.binary import hd95 # # pip install medpy
# import datetime
# from network.moldel2 import FPNUNet_CBAMResidual  


    
    
# import os
# import torch
# import torch.nn as nn
# import numpy as np
# import csv
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# from glob import glob
# from network.moldel2 import FPNUNet_CBAMResidual   
# # --- 配置 ---
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # 指定包含所有权重文件的目录
# weight_dir = "results/FPNUNet_CBAMResidual_20250924_004042_BUSI456"

# # --- 加载数据 ---
# test_data = get_data(
#     image_dir='/home/user/HUXUFENG/UI/Diffusin-net_GAN/BUSI/test/images',
#     mask_dir='/home/user/HUXUFENG/UI/Diffusin-net_GAN/BUSI/test/masks'
# )
# test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

# # --- 评估指标函数（保持不变）---
# def dice_coef(pred, target, eps=1e-6):
#     pred = pred.flatten()
#     target = target.flatten()
#     intersection = (pred * target).sum()
#     return (2. * intersection + eps) / (pred.sum() + target.sum() + eps)

# def iou_score(pred, target, eps=1e-6):
#     pred = pred.flatten()
#     target = target.flatten()
#     intersection = (pred * target).sum()
#     union = pred.sum() + target.sum() - intersection
#     return (intersection + eps) / (union + eps)

# def precision_score(pred, target, eps=1e-6):
#     pred = pred.flatten()
#     target = target.flatten()
#     tp = (pred * target).sum()
#     fp = (pred * (1 - target)).sum()
#     return (tp + eps) / (tp + fp + eps)

# def recall_score(pred, target, eps=1e-6):
#     pred = pred.flatten()
#     target = target.flatten()
#     tp = (pred * target).sum()
#     fn = ((1 - pred) * target).sum()
#     return (tp + eps) / (tp + fn + eps)

# def specificity_score(pred, target, eps=1e-6):
#     pred = pred.flatten()
#     target = target.flatten()
#     tn = ((1 - pred) * (1 - target)).sum()
#     fp = (pred * (1 - target)).sum()
#     return (tn + eps) / (tn + fp + eps)

# def pixel_accuracy(pred, target):
#     return (pred == target).sum() / np.prod(target.shape)
    
# # --- 存储所有模型的指标结果 ---
# all_metrics = {
#     'dice': [], 'iou': [], 'prec': [], 'recall': [],
#     'spe': [], 'acc': [], 'hd95': []
# }

# # 遍历目录下的所有 .pth 或 .pt 文件
# weight_files = glob(os.path.join(weight_dir, "*.pth")) + glob(os.path.join(weight_dir, "*.pt"))

# if not weight_files:
#     print(f"在目录 {weight_dir} 中未找到 .pth 或 .pt 文件。")
# else:
#     # 定义 CSV 文件名
#     csv_filename = "ensemble_metrics.csv"
#     csv_file_path = os.path.join(weight_dir, csv_filename)
    
#     # 写入 CSV 文件
#     with open(csv_file_path, 'w', newline='') as csvfile:
#         fieldnames = ['Model Name', 'Dice', 'IoU', 'Precision', 'Recall', 'Specificity', 'Accuracy', 'Hd95']
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
#         # 写入表头
#         writer.writeheader()
        
#         # 循环遍历每个权重文件
#         for weight_path in weight_files:
#             print(f"正在评估模型: {weight_path}")
            
#             # --- 加载模型 ---
#             model = FPNUNet_CBAMResidual()
#             model = nn.DataParallel(model)
#             model.load_state_dict(torch.load(weight_path, map_location=device))
#             model = model.to(device)
#             model.eval()

#             # --- 测试与统计单个模型的平均指标 ---
#             dice_list, iou_list, prec_list, recall_list, acc_list, spe_list, hd95_list = [], [], [], [], [], [], []
#             with torch.no_grad():
#                 for images, masks in tqdm(test_loader, desc=f"Evaluating {os.path.basename(weight_path)}"):
#                     images = images.to(device)
#                     masks = masks.to(device)
                    
#                     outputs, *aux_outs = model(images)
#                     preds = torch.sigmoid(outputs)
#                     preds = (preds > 0.5).float()
                    
#                     dice_list.append(dice_coef(preds, masks).item())
#                     iou_list.append(iou_score(preds, masks).item())
#                     prec_list.append(precision_score(preds, masks).item())
#                     recall_list.append(recall_score(preds, masks).item())
#                     spe_list.append(specificity_score(preds, masks).item())
                    
#                     pred_np = preds.cpu().numpy().astype(np.uint8)[0, 0]
#                     mask_np = masks.cpu().numpy().astype(np.uint8)[0, 0]
#                     acc_list.append(pixel_accuracy(pred_np, mask_np))
                    
#                     try:
#                         if pred_np.any() and mask_np.any():
#                             hd95_val = hd95(pred_np, mask_np)
#                         else:
#                             hd95_val = np.nan
#                     except:
#                         hd95_val = np.nan
#                     hd95_list.append(hd95_val)
            
#             # 计算当前模型的平均指标
#             avg_dice = np.nanmean(dice_list)
#             avg_iou = np.nanmean(iou_list)
#             avg_prec = np.nanmean(prec_list)
#             avg_recall = np.nanmean(recall_list)
#             avg_spe = np.nanmean(spe_list)
#             avg_acc = np.nanmean(acc_list)
#             avg_hd95 = np.nanmean([v for v in hd95_list if not np.isnan(v)])

#             # 将当前模型的平均指标添加到总列表中
#             all_metrics['dice'].append(avg_dice)
#             all_metrics['iou'].append(avg_iou)
#             all_metrics['prec'].append(avg_prec)
#             all_metrics['recall'].append(avg_recall)
#             all_metrics['spe'].append(avg_spe)
#             all_metrics['acc'].append(avg_acc)
#             all_metrics['hd95'].append(avg_hd95)
            
#             # 写入当前模型的结果到 CSV
#             writer.writerow({
#                 'Model Name': os.path.basename(weight_path),
#                 'Dice': f"{avg_dice:.4f}",
#                 'IoU': f"{avg_iou:.4f}",
#                 'Precision': f"{avg_prec:.4f}",
#                 'Recall': f"{avg_recall:.4f}",
#                 'Specificity': f"{avg_spe:.4f}",
#                 'Accuracy': f"{avg_acc:.4f}",
#                 'Hd95': f"{avg_hd95:.4f}" if not np.isnan(avg_hd95) else "N/A"
#             })
            
#         # --- 统计所有模型的平均值和标准差 ---
#         final_results = {}
#         for metric, values in all_metrics.items():
#             if values: # 确保列表不为空
#                 avg = np.nanmean(values)
#                 std = np.nanstd(values)
#                 final_results[metric] = {'avg': avg, 'std': std}
#             else:
#                 final_results[metric] = {'avg': np.nan, 'std': np.nan}
        
#         # 将最终的平均值和标准差写入 CSV
#         writer.writerow({
#             'Model Name': f"Ensemble Mean ± Std ({len(weight_files)} models)",
#             'Dice': f"{final_results['dice']['avg']:.4f} ± {final_results['dice']['std']:.4f}",
#             'IoU': f"{final_results['iou']['avg']:.4f} ± {final_results['iou']['std']:.4f}",
#             'Precision': f"{final_results['prec']['avg']:.4f} ± {final_results['prec']['std']:.4f}",
#             'Recall': f"{final_results['recall']['avg']:.4f} ± {final_results['recall']['std']:.4f}",
#             'Specificity': f"{final_results['spe']['avg']:.4f} ± {final_results['spe']['std']:.4f}",
#             'Accuracy': f"{final_results['acc']['avg']:.4f} ± {final_results['acc']['std']:.4f}",
#             'Hd95': f"{final_results['hd95']['avg']:.4f} ± {final_results['hd95']['std']:.4f}"
#         })
        
#     print(f"\n✅ 所有评估结果已保存到 {csv_file_path}")



import os
import csv
from glob import glob
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ==== 你的工程内依赖 ====
# 模型
from network.moldel2 import FPNUNet_CBAMResidual
# 数据
from Data import get_data

# ==== 可选的 hd95 ====
try:
    from medpy.metric.binary import hd95 as hd95_fn
    _HAS_HAUSDORFF = True
except Exception:
    hd95_fn = None
    _HAS_HAUSDORFF = False

# ---------------- Metrics ----------------
def dice_coef(pred, target, eps=1e-6):
    # pred/target: torch.Tensor (B,1,H,W) , 已经是 {0,1}
    pred = pred.flatten()
    target = target.flatten()
    inter = (pred * target).sum()
    return (2. * inter + eps) / (pred.sum() + target.sum() + eps)

def iou_score(pred, target, eps=1e-6):
    pred = pred.flatten()
    target = target.flatten()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return (inter + eps) / (union + eps)

def precision_score(pred, target, eps=1e-6):
    pred = pred.flatten()
    target = target.flatten()
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    return (tp + eps) / (tp + fp + eps)

def recall_score(pred, target, eps=1e-6):
    pred = pred.flatten()
    target = target.flatten()
    tp = (pred * target).sum()
    fn = ((1 - pred) * target).sum()
    return (tp + eps) / (tp + fn + eps)

def specificity_score(pred, target, eps=1e-6):
    pred = pred.flatten()
    target = target.flatten()
    tn = ((1 - pred) * (1 - target)).sum()
    fp = (pred * (1 - target)).sum()
    return (tn + eps) / (tn + fp + eps)

def pixel_accuracy(pred_np, target_np):
    # pred_np/target_np: np.uint8 的二维 0/1 图
    return (pred_np == target_np).mean()

# ---------------- Loader helper ----------------
_DROP_KEYS = ("total_ops", "total_params")

def load_clean_state(model: nn.Module, ckpt_path: str, device: torch.device):
    """过滤 thop 注入键 & 处理 DataParallel 前缀，宽松加载。"""
    raw = torch.load(ckpt_path, map_location=device)
    state = raw.get('state_dict', raw) if isinstance(raw, dict) else raw
    # 1) drop thop keys
    state = {k: v for k, v in state.items() if all(dk not in k for dk in _DROP_KEYS)}
    # 2) 直接试 strict=False
    try:
        model.load_state_dict(state, strict=False)
        return
    except Exception:
        pass
    # 3) 处理 module. 前缀
    has_module = any(k.startswith("module.") for k in state.keys())
    if has_module:
        # 去前缀
        state2 = {k.replace("module.", "", 1): v for k, v in state.items()}
    else:
        # 加前缀
        state2 = {f"module.{k}": v for k, v in state.items()}
    model.load_state_dict(state2, strict=False)

# ---------------- Main ----------------
if __name__ == "__main__":
    # --- 配置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True  # 评测阶段也可开

    # 指定包含所有权重文件的目录
    weight_dir = "results/FPNUNet_CBAMResidual_20250924_004042_BUSI456"

    # --- 加载数据 ---
    test_data = get_data(
        image_dir='/home/user/HUXUFENG/UI/Diffusin-net_GAN/BUSI/test/images',
        mask_dir='/home/user/HUXUFENG/UI/Diffusin-net_GAN/BUSI/test/masks'
    )
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=2)

    # --- 收集权重文件 ---
    weight_files = glob(os.path.join(weight_dir, "*.pth")) + glob(os.path.join(weight_dir, "*.pt"))
    if not weight_files:
        raise FileNotFoundError(f"在目录 {weight_dir} 中未找到 .pth 或 .pt 文件。")

    # --- CSV 输出 ---
    csv_filename = "ensemble_metrics.csv"
    csv_file_path = os.path.join(weight_dir, csv_filename)

    # --- 累计各模型指标 ---
    all_metrics = {'dice': [], 'iou': [], 'prec': [], 'recall': [], 'spe': [], 'acc': [], 'hd95': []}

    with open(csv_file_path, 'w', newline='') as csvfile:
        fieldnames = ['Model Name', 'Dice', 'IoU', 'Precision', 'Recall', 'Specificity', 'Accuracy', 'Hd95']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for weight_path in weight_files:
            print(f"\n==> 正在评估模型: {weight_path}")

            # --- 构建模型（先 base，再按需包 DP）---
            base = FPNUNet_CBAMResidual().to(device)
            model = nn.DataParallel(base) if (torch.cuda.is_available() and torch.cuda.device_count() > 1) else base

            # --- 加载权重（清理 thop/前缀）---
            load_clean_state(model, weight_path, device)
            model.eval()

            # --- 测试该模型 ---
            dice_list, iou_list, prec_list, recall_list, acc_list, spe_list, hd95_list = [], [], [], [], [], [], []
            with torch.no_grad():
                for images, masks in tqdm(test_loader, desc=f"Evaluating {os.path.basename(weight_path)}"):
                    images = images.to(device)
                    masks = masks.to(device)

                    out = model(images)
                    # 兼容：模型可能返回张量或 (主输出, 辅助输出…)
                    if isinstance(out, (list, tuple)):
                        logits = out[0]
                    else:
                        logits = out

                    probs = torch.sigmoid(logits)
                    preds = (probs > 0.5).float()

                    # torch 上算核心指标
                    dice_list.append(dice_coef(preds, masks).item())
                    iou_list.append(iou_score(preds, masks).item())
                    prec_list.append(precision_score(preds, masks).item())
                    recall_list.append(recall_score(preds, masks).item())
                    spe_list.append(specificity_score(preds, masks).item())

                    # numpy 上算像素精度 & hd95
                    pred_np = preds.detach().cpu().numpy().astype(np.uint8)[0, 0]
                    mask_np = masks.detach().cpu().numpy().astype(np.uint8)[0, 0]
                    acc_list.append(pixel_accuracy(pred_np, mask_np))
                    if _HAS_HAUSDORFF and pred_np.any() and mask_np.any():
                        try:
                            hd95_val = float(hd95_fn(pred_np, mask_np))
                        except Exception:
                            hd95_val = np.nan
                    else:
                        hd95_val = np.nan
                    hd95_list.append(hd95_val)

            # --- 汇总该模型 ---
            avg_dice = float(np.nanmean(dice_list))
            avg_iou = float(np.nanmean(iou_list))
            avg_prec = float(np.nanmean(prec_list))
            avg_recall = float(np.nanmean(recall_list))
            avg_spe = float(np.nanmean(spe_list))
            avg_acc = float(np.nanmean(acc_list))
            valid_hd = [v for v in hd95_list if not np.isnan(v)]
            avg_hd95 = float(np.nanmean(valid_hd)) if len(valid_hd) > 0 else float("nan")

            # 进全局
            all_metrics['dice'].append(avg_dice)
            all_metrics['iou'].append(avg_iou)
            all_metrics['prec'].append(avg_prec)
            all_metrics['recall'].append(avg_recall)
            all_metrics['spe'].append(avg_spe)
            all_metrics['acc'].append(avg_acc)
            all_metrics['hd95'].append(avg_hd95)

            # 写 CSV
            writer.writerow({
                'Model Name': os.path.basename(weight_path),
                'Dice': f"{avg_dice:.4f}",
                'IoU': f"{avg_iou:.4f}",
                'Precision': f"{avg_prec:.4f}",
                'Recall': f"{avg_recall:.4f}",
                'Specificity': f"{avg_spe:.4f}",
                'Accuracy': f"{avg_acc:.4f}",
                'Hd95': f"{avg_hd95:.4f}" if not np.isnan(avg_hd95) else "N/A"
            })

        # --- 全模型统计 ---
        final_results = {}
        for metric, values in all_metrics.items():
            if len(values) > 0:
                avg = float(np.nanmean(values))
                std = float(np.nanstd(values))
            else:
                avg, std = float("nan"), float("nan")
            final_results[metric] = {'avg': avg, 'std': std}

        writer.writerow({
            'Model Name': f"Ensemble Mean ± Std ({len(weight_files)} models)",
            'Dice': f"{final_results['dice']['avg']:.4f} ± {final_results['dice']['std']:.4f}",
            'IoU': f"{final_results['iou']['avg']:.4f} ± {final_results['iou']['std']:.4f}",
            'Precision': f"{final_results['prec']['avg']:.4f} ± {final_results['prec']['std']:.4f}",
            'Recall': f"{final_results['recall']['avg']:.4f} ± {final_results['recall']['std']:.4f}",
            'Specificity': f"{final_results['spe']['avg']:.4f} ± {final_results['spe']['std']:.4f}",
            'Accuracy': f"{final_results['acc']['avg']:.4f} ± {final_results['acc']['std']:.4f}",
            'Hd95': f"{final_results['hd95']['avg']:.4f} ± {final_results['hd95']['std']:.4f}" if not np.isnan(final_results['hd95']['avg']) else "N/A"
        })

    print(f"\n✅ 所有评估结果已保存到 {csv_file_path}")



















# # --- 配置 ---
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # 更改：现在指定一个包含所有权重文件的目录
# weight_dir = "results/FPNUNet_CBAM_Residual_20250916_1416_37/"

# # --- 加载数据 ---
# test_data = BUS_UCLM_Dataset(
#     image_dir='/home/user/HUXUFENG/UI/Diffusin-net_GAN/BUS-UCLM/test/images',
#     mask_dir='/home/user/HUXUFENG/UI/Diffusin-net_GAN/BUS-UCLM/test/masks'
# )
# test_loader = DataLoader(test_data, batch_size=1, shuffle=False)



# # ----------------- 评估指标函数 --------------------
# def dice_coef(pred, target, eps=1e-6):
#     pred = pred.flatten()
#     target = target.flatten()
#     intersection = (pred * target).sum()
#     return (2. * intersection + eps) / (pred.sum() + target.sum() + eps)

# def iou_score(pred, target, eps=1e-6):
#     pred = pred.flatten()
#     target = target.flatten()
#     intersection = (pred * target).sum()
#     union = pred.sum() + target.sum() - intersection
#     return (intersection + eps) / (union + eps)

# def precision_score(pred, target, eps=1e-6):
#     pred = pred.flatten()
#     target = target.flatten()
#     tp = (pred * target).sum()
#     fp = (pred * (1 - target)).sum()
#     return (tp + eps) / (tp + fp + eps)

# def recall_score(pred, target, eps=1e-6):
#     pred = pred.flatten()
#     target = target.flatten()
#     tp = (pred * target).sum()
#     fn = ((1 - pred) * target).sum()
#     return (tp + eps) / (tp + fn + eps)

# def specificity_score(pred, target, eps=1e-6):
#     pred = pred.flatten()
#     target = target.flatten()
#     tn = ((1 - pred) * (1 - target)).sum()
#     fp = (pred * (1 - target)).sum()
#     return (tn + eps) / (tn + fp + eps)

# def pixel_accuracy(pred, target):
#     return (pred == target).sum() / np.prod(target.shape)

# # --- 统计多个模型的平均指标和标准差 ---
# # 创建一个字典来存储所有模型的指标结果
# all_metrics = {
#     'dice': [], 'iou': [], 'prec': [], 'recall': [],
#     'spe': [], 'acc': [], 'hd95': []
# }

# # 遍历目录下的所有 .pth 或 .pt 文件
# weight_files = glob(os.path.join(weight_dir, "*.pth")) + glob(os.path.join(weight_dir, "*.pt"))

# # 检查是否找到权重文件
# if not weight_files:
#     print(f"在目录 {weight_dir} 中未找到 .pth 或 .pt 文件。")
# else:
#     # 循环遍历每个权重文件
#     for weight_path in weight_files:
#         print(f"正在评估模型: {weight_path}")

#         # --- 加载模型 ---
#         model = FPNUNet_CBAM_Residual()
#         model = nn.DataParallel(model)
#         model.load_state_dict(torch.load(weight_path, map_location=device))
#         model = model.to(device)
#         model.eval()

#         # --- 测试与统计单个模型的平均指标 ---
#         dice_list, iou_list, prec_list, recall_list, acc_list, spe_list, hd95_list = [], [], [], [], [], [], []

#         with torch.no_grad():
#             for images, masks in tqdm(test_loader, desc=f"Evaluating {os.path.basename(weight_path)}"):
#                 images = images.to(device)
#                 masks = masks.to(device)
                
#                 outputs, *aux_outs = model(images)
#                 preds = torch.sigmoid(outputs)
#                 preds = (preds > 0.5).float()
                
#                 # ... (计算指标的逻辑，与原代码相同)
#                 dice_list.append(dice_coef(preds, masks).item())
#                 iou_list.append(iou_score(preds, masks).item())
#                 prec_list.append(precision_score(preds, masks).item())
#                 recall_list.append(recall_score(preds, masks).item())
#                 spe_list.append(specificity_score(preds, masks).item())
                
#                 pred_np = preds.cpu().numpy().astype(np.uint8)[0, 0]
#                 mask_np = masks.cpu().numpy().astype(np.uint8)[0, 0]
#                 acc_list.append(pixel_accuracy(pred_np, mask_np))
                
#                 try:
#                     if pred_np.any() and mask_np.any():
#                         hd95_val = hd95(pred_np, mask_np)
#                     else:
#                         hd95_val = np.nan
#                 except:
#                     hd95_val = np.nan
#                 hd95_list.append(hd95_val)
                          
#         # 计算当前模型的平均指标
#         avg_dice = np.nanmean(dice_list)
#         avg_iou = np.nanmean(iou_list)
#         avg_prec = np.nanmean(prec_list)
#         avg_recall = np.nanmean(recall_list)
#         avg_spe = np.nanmean(spe_list)
#         avg_acc = np.nanmean(acc_list)
#         avg_hd95 = np.nanmean([v for v in hd95_list if not np.isnan(v)])

#         # 将当前模型的平均指标添加到总列表中
#         all_metrics['dice'].append(np.nanmean(dice_list))
#         all_metrics['iou'].append(np.nanmean(iou_list))
#         all_metrics['prec'].append(np.nanmean(prec_list))
#         all_metrics['recall'].append(np.nanmean(recall_list))
#         all_metrics['spe'].append(np.nanmean(spe_list))
#         all_metrics['acc'].append(np.nanmean(acc_list))
#         all_metrics['hd95'].append(np.nanmean([v for v in hd95_list if not np.isnan(v)]))
        
        
#         weight_name = os.path.splitext(os.path.basename(weight_path))[0]
#         metrics_txt_single = os.path.join(weight_dir, f"{weight_name}_metrics.txt")
        
#         with open(metrics_txt_single, "w") as f:
#             f.write(f"==== Evaluation Metrics for {weight_name} ====\n")
#             f.write(f"Dice:           {avg_dice:.4f}\n")
#             f.write(f"IoU:            {avg_iou:.4f}\n")
#             f.write(f"Precision:      {avg_prec:.4f}\n")
#             f.write(f"Recall:         {avg_recall:.4f}\n")
#             f.write(f"Specificity:    {avg_spe:.4f}\n")
#             f.write(f"Accuracy:       {avg_acc:.4f}\n")
#             f.write(f"Hd95:           {avg_hd95:.4f}\n")
        
#         print(f"✅ Metrics saved to {metrics_txt_single}")
        
        
        
#     # --- 统计所有模型的平均值和标准差 ---
#     final_results = {}
#     for metric, values in all_metrics.items():
#         if values: # 确保列表不为空
#             avg = np.nanmean(values)
#             std = np.nanstd(values)
#             final_results[metric] = {'avg': avg, 'std': std}
#         else:
#             final_results[metric] = {'avg': np.nan, 'std': np.nan}


#     # --- 保存最终结果 ---
#     metrics_txt = os.path.join(weight_dir, "ensemble_metrics.txt")
#     with open(metrics_txt, "w") as f:
#         f.write("==== Ensemble Evaluation Metrics on Test Set ====\n")
#         f.write(f"Evaluated {len(weight_files)} models.\n\n")
        
#         for metric, stats in final_results.items():
#             f.write(f"{metric.capitalize()}: {stats['avg']:.4f} ± {stats['std']:.4f}\n")
        
#     print(f"\n✅ Ensemble metrics saved to {metrics_txt}")
    




# # 配置
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# weight_path = "results/FPNUNet_CBAM_Residual_Auxiliary _Classifier/model_last.pth"

# # 加载数据
# test_data = BUS_UCLM_Dataset(
#     image_dir='/home/user/HUXUFENG/UI/Diffusin-net_GAN/BUS-UCLM/test/images',
#     mask_dir='/home/user/HUXUFENG/UI/Diffusin-net_GAN/BUS-UCLM/test/masks'
# )
# test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

# # 加载模型
# model = FPNUNet_CBAM_Residual()
# model = nn.DataParallel(model)
# model.load_state_dict(torch.load(weight_path, map_location=device))
# model = model.to(device)
# model.eval()


# def count_parameters(model):
#     total = sum(p.numel() for p in model.parameters())
#     trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     return total, trainable

# total_params, trainable_params = count_parameters(model)
# model_name = model.module.__class__.__name__

# # ----------------- 评估指标函数 --------------------
# def dice_coef(pred, target, eps=1e-6):
#     pred = pred.flatten()
#     target = target.flatten()
#     intersection = (pred * target).sum()
#     return (2. * intersection + eps) / (pred.sum() + target.sum() + eps)

# def iou_score(pred, target, eps=1e-6):
#     pred = pred.flatten()
#     target = target.flatten()
#     intersection = (pred * target).sum()
#     union = pred.sum() + target.sum() - intersection
#     return (intersection + eps) / (union + eps)

# def precision_score(pred, target, eps=1e-6):
#     pred = pred.flatten()
#     target = target.flatten()
#     tp = (pred * target).sum()
#     fp = (pred * (1 - target)).sum()
#     return (tp + eps) / (tp + fp + eps)

# def recall_score(pred, target, eps=1e-6):
#     pred = pred.flatten()
#     target = target.flatten()
#     tp = (pred * target).sum()
#     fn = ((1 - pred) * target).sum()
#     return (tp + eps) / (tp + fn + eps)

# def specificity_score(pred, target, eps=1e-6):
#     pred = pred.flatten()
#     target = target.flatten()
#     tn = ((1 - pred) * (1 - target)).sum()
#     fp = (pred * (1 - target)).sum()
#     return (tn + eps) / (tn + fp + eps)

# def pixel_accuracy(pred, target):
#     return (pred == target).sum() / np.prod(target.shape)

# # ----------------- 测试与统计平均指标 --------------------
# dice_list, iou_list, prec_list, recall_list, acc_list, spe_list, hd95_list = [], [], [], [], [], [], []

# with torch.no_grad():
#     for images, masks in tqdm(test_loader, desc="Evaluating"):
#         images = images.to(device)
#         masks = masks.to(device)

#         outputs, *aux_outs = model(images)
#         preds = torch.sigmoid(outputs)
#         preds = (preds > 0.5).float()

#         # flatten tensors
#         pred_np = preds.cpu().numpy().astype(np.uint8)[0, 0]
#         mask_np = masks.cpu().numpy().astype(np.uint8)[0, 0]

#         # 计算指标
#         dice_list.append(dice_coef(preds, masks).item())
#         iou_list.append(iou_score(preds, masks).item())
#         prec_list.append(precision_score(preds, masks).item())
#         recall_list.append(recall_score(preds, masks).item())
#         spe_list.append(specificity_score(preds, masks).item())
#         acc_list.append(pixel_accuracy(pred_np, mask_np))
        
#         # Hd95 需要预测与标签中至少有一个前景
#         try:
#             if pred_np.any() and mask_np.any():
#                 hd95_val = hd95(pred_np, mask_np)
#             else:
#                 hd95_val = np.nan
#         except:
#             hd95_val = np.nan
#         hd95_list.append(hd95_val)

# # ----------------- 平均指标结果 --------------------
# avg_dice = np.nanmean(dice_list)
# avg_iou = np.nanmean(iou_list)
# avg_prec = np.nanmean(prec_list)
# avg_recall = np.nanmean(recall_list)
# avg_spe = np.nanmean(spe_list)
# avg_acc = np.nanmean(acc_list)
# avg_hd95 = np.nanmean([v for v in hd95_list if not np.isnan(v)])


# save_dir = os.path.dirname(weight_path)



# os.makedirs(save_dir, exist_ok=True)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # 提取权重文件名（去掉后缀 .pth/.pt）
# weight_name = os.path.splitext(os.path.basename(weight_path))[0]

# # 生成 metrics.txt 文件名
# metrics_txt = os.path.join(save_dir, f"{weight_name}_metrics.txt")


# results = []

# with torch.no_grad():
#     for idx, (img, mask) in enumerate(tqdm(test_loader)):
#         img = img.to(device)
#         mask = mask.to(device)

#         output = model(img)
            
#         if isinstance(outputs, (list, tuple)):
#             final_out, *aux_outs = outputs
#         else:
#             final_out, aux_outs = outputs, []


#         prob = torch.sigmoid(final_out)
#         pred = (prob > 0.5).float()

#         # 不再计算 metrics，仅记录 sample 名称
#         sample_name = f"sample_{idx+1}"
#         results.append({"Sample": sample_name})

#         # 拼图保存
#         comp = torch.cat([img[0], mask[0], pred[0]], dim=2)
#         # save_image(comp, os.path.join(save_dir, f"{sample_name}_comp.png"), normalize=True)


# print(f"权重路径: {weight_path}")
# print(f"保存目录: {save_dir}")
# print(f"指标文件路径: {metrics_txt}")
# print(f"设备: {device}")

# with open(metrics_txt, "w") as f:
#     f.write("==== Evaluation Metrics on Test Set ====\n")
#     f.write(f"Dice:         {avg_dice:.4f}\n")
#     f.write(f"IoU:          {avg_iou:.4f}\n")
#     f.write(f"Precision:    {avg_prec:.4f}\n")
#     f.write(f"Recall:       {avg_recall:.4f}\n")
#     f.write(f"Specificity:  {avg_spe:.4f}\n")
#     f.write(f"Accuracy:     {avg_acc:.4f}\n")
#     f.write(f"Hd95:         {avg_hd95:.4f}\n")
#     f.write("\n==== Model Info ====\n")
#     f.write(f"Model Structure:  {model_name}\n")
#     f.write(f"Total Params:     {total_params:,}\n")
#     f.write(f"Trainable Params: {trainable_params:,}\n")
#     f.write("\n==== Inference Hyperparameters ====\n")
#     f.write(f"batch_size:       {test_loader.batch_size}\n")
#     f.write(f"loss_fn:          0.5 * BCEWithLogitsLoss + 0.5 * DiceLoss\n")
#     f.write(f"input_size:       {str(next(iter(test_loader))[0].shape)}\n")

# print(f"\n✅ Metrics saved to {metrics_txt}")





  