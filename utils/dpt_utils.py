# utils/dpt_utils.py
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from pathlib import Path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'DPT'))

DPT_PATH = Path("weights/dpt_hybrid-midas-501f0c75.pt")

def load_dpt_model(device="cuda"):
    from dpt.models import DPTDepthModel
    model = DPTDepthModel(
        path=str(DPT_PATH),
        backbone="vitb_rn50_384",
        non_negative=True,
        enable_attention_hooks=False,
    )
    model.to(device)
    model.eval()
    return model

def predict_dpt_depth(model, rgb_image, original_size, device="cuda"):
    """
    rgb_image: numpy array [H, W, 3], values in [0,1]
    original_size: (H, W)
    Returns: depth map [H, W], numpy, inverse depth (larger = closer)
    """
    H_orig, W_orig = original_size

    # DPT-Hybrid (vitb_rn50_384) requires input size divisible by 32
    # We'll resize to the nearest multiple of 32, but keep aspect ratio via padding
    from torchvision.transforms import Compose, Resize, ToTensor, Normalize, Pad

    # Calculate new size (multiple of 32)
    def _make_multiple(x, m=32):
        return ((x + m - 1) // m) * m

    H_new = _make_multiple(H_orig)
    W_new = _make_multiple(W_orig)

    # Resize + pad to exact multiple of 32
    transform = Compose([
        ToTensor(),
        Resize((H_new, W_new), antialias=True),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    img_tensor = transform(rgb_image).unsqueeze(0).to(device)  # [1,3,H_new,W_new]

    with torch.no_grad():
        depth = model(img_tensor)

    # Resize depth back to original size
    depth = F.interpolate(
        depth.unsqueeze(1),
        size=(H_orig, W_orig),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1).squeeze(0)  # [H, W]

    return depth.cpu().numpy()