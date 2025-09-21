import torch
import torchvision.transforms as T
import torchvision.utils as vutils
from PIL import Image
from Unet_attention import AttentionUNet,UNet,FPNUNet,  FPNUNetV2,FPNUNetV3
from PIL import ImageDraw, ImageFont
from tools import visualize_with_labels

# 工具：去掉 module. 前缀
# -----------------------------------
def strip_module_prefix(state_dict):
    new_state = {}
    for k, v in state_dict.items():
        key = k
        if k.startswith("module."):
            key = k[len("module."):]
        new_state[key] = v
    return new_state

# -----------------------------------
# 加载模型
# -----------------------------------
def load_model(checkpoint_path, device):
    # 1) 构建网络（参数要和训练时一致）
    model = FPNUNetV3().to(device)

    # 2) 载入 checkpoint
    raw_state = torch.load(checkpoint_path, map_location=device)
    # 如果你保存的是 {'model': state_dict, ...} 形式，请用 raw_state['model']
    if "state_dict" in raw_state:
        raw_state = raw_state["state_dict"]

    # 3) 去掉可能的 "module." 前缀
    state = strip_module_prefix(raw_state)

    # 4) 加载权重，strict=False 跳过不匹配的键
    msg = model.load_state_dict(state, strict=False)
    if msg.missing_keys:
        print("Warning: missing keys in checkpoint:", msg.missing_keys)
    if msg.unexpected_keys:
        print("Warning: unexpected keys in checkpoint:", msg.unexpected_keys)

    model.eval()
    return model

# -----------------------------------
# 预处理
# -----------------------------------
def preprocess_image(img_path, device, size=(256,256)):
    tf = T.Compose([
        T.Grayscale(num_output_channels=1),
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])
    img = Image.open(img_path)
    return tf(img).unsqueeze(0).to(device)

# -----------------------------------
# 推理 & 保存可视化

def infer_and_save(model, img_tensor, save_path, thresh=0.5):
    with torch.no_grad():
        logits = model(img_tensor)
        probs  = torch.sigmoid(logits)
        mask   = (probs > thresh).float()

        # 原图要反 Normalize 回 [0,1]
        orig = img_tensor * 0.5 + 0.5

    visualize_with_labels(orig, probs, mask, save_path, n=min(4, orig.size(0)))
    print(f"Saved labeled result to {save_path}")
    
    
    

# -----------------------------------
# 脚本入口
# -----------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--img",  type=str,
                        default="image/benign_benign(193).png",
                        help="输入影像路径")
    parser.add_argument("--ckpt", type=str,
                        default="FPNUNetV3_result/FPNUNetV3_result_result_unet.pth",
                        help="权重文件")
    parser.add_argument("--out",  type=str,
                        default="output/result.png",
                        help="保存结果")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(args.ckpt, device)
    img    = preprocess_image(args.img, device)
    infer_and_save(model, img, args.out)