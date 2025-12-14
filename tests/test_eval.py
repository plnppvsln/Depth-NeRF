import torch
import lpips
import numpy as np
from pathlib import Path

from skimage.io import imread
from skimage import img_as_float
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# пути к изображениям
GT_DIR = Path("data/nerf_llff_data/table/images_64")
PR_DIR = Path("logs/table_test_hashXYZ_sphereVIEW_fine1024_log2T19_lr0.01_decay10_RAdam_TV1e-06/testset_001000")

IMG_GT = "IMG_20251211_214144.png"
IMG_PR = "000.png"


def load_numpy(path):
    """[H, W, 3], float32, [0,1]"""
    return img_as_float(imread(path)).astype(np.float32)


def load_torch(path):
    """[1, 3, H, W], float32, [-1,1]"""
    img = img_as_float(imread(path)).astype(np.float32)
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    img = img * 2.0 - 1.0
    return img


def compute_metrics():
    # ---------- NumPy (PSNR, SSIM) ----------
    img_gt_np = load_numpy(GT_DIR / IMG_GT)
    img_pr_np = load_numpy(PR_DIR / IMG_PR)

    assert img_gt_np.shape == img_pr_np.shape

    psnr = peak_signal_noise_ratio(
        img_gt_np,
        img_pr_np,
        data_range=1.0
    )

    ssim = structural_similarity(
        img_gt_np,
        img_pr_np,
        channel_axis=-1,
        data_range=1.0
    )

    # ---------- PyTorch (LPIPS) ----------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img_gt_t = load_torch(GT_DIR / IMG_GT).to(device)
    img_pr_t = load_torch(PR_DIR / IMG_PR).to(device)

    lpips_fn = lpips.LPIPS(net="alex").to(device)

    with torch.no_grad():
        lpips_val = lpips_fn(img_gt_t, img_pr_t).item()

    # ---------- Результат ----------
    print(f"PSNR : {psnr:.3f} dB   ↑")
    print(f"SSIM : {ssim:.4f}     ↑")
    print(f"LPIPS: {lpips_val:.6f} ↓")


if __name__ == "__main__":
    compute_metrics()
