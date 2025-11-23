import os
import torch
import numpy as np
import imageio 
import json
import cv2

from utils import get_bbox3d_for_nerf_style, pose_spherical, compute_near_far_from_poses

def load_nerf_style(basedir, half_res=False, testskip=8, device='cuda:0'):
    # ---- Load transforms.json ----
    with open(os.path.join(basedir, "transforms.json"), "r") as f:
        meta = json.load(f)

    H = int(meta["h"])
    W = int(meta["w"])

    # Get focal lengths (InstantNGP format supports separate fl_x and fl_y)
    fl_x = meta.get("fl_x", None)
    fl_y = meta.get("fl_y", None)
    
    if fl_x is not None:
        focal_x = fl_x
    else:
        # derive from camera_angle_x
        camera_angle_x = meta.get("camera_angle_x")
        if camera_angle_x is not None:
            focal_x = 0.5 * W / np.tan(0.5 * camera_angle_x)
        else:
            raise ValueError("Either fl_x or camera_angle_x must be provided in transforms.json")
    
    if fl_y is not None:
        focal_y = fl_y
    else:
        # derive from camera_angle_y if available, otherwise use focal_x
        camera_angle_y = meta.get("camera_angle_y")
        if camera_angle_y is not None:
            focal_y = 0.5 * H / np.tan(0.5 * camera_angle_y)
        else:
            focal_y = focal_x  # Assume square pixels if not specified
    
    # Use focal_x as the primary focal length (for compatibility with existing code)
    focal = focal_x

    cx = meta.get("cx", W / 2)
    cy = meta.get("cy", H / 2)

    # ---- Load images & pose matrices ----
    imgs = []
    poses = []
    test_indices = []

    for i, frame in enumerate(meta["frames"]):
        fname = os.path.join(basedir, frame["file_path"])
        if not os.path.exists(fname):
            print(f"[WARN] missing image: {fname}")
            continue

        img = imageio.imread(fname)
        imgs.append(img)
        poses.append(np.array(frame["transform_matrix"], dtype=np.float32))

        if (i + 1) % testskip == 0:
            test_indices.append(len(imgs) - 1)

    imgs = (np.array(imgs) / 255.).astype(np.float32)
    poses = np.array(poses)

    # ---- Train/val/test split ----
    all_ids = np.arange(len(imgs))
    test_ids = np.array(test_indices)
    train_ids = np.array([i for i in all_ids if i not in test_ids])
    val_ids = test_ids.copy()
    i_split = [train_ids, val_ids, test_ids]

    # ---- Optional half resolution ----
    if half_res:
        H //= 2
        W //= 2
        focal /= 2
        focal_x /= 2
        focal_y /= 2
        cx /= 2
        cy /= 2
        imgs_half = np.zeros((imgs.shape[0], H, W, imgs.shape[-1]), dtype=np.float32)
        for i, im in enumerate(imgs):
            imgs_half[i] = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half

    # ---- Camera intrinsics ----
    # Use separate focal lengths for x and y if available (InstantNGP format)
    K = np.array([
        [focal_x, 0, cx],
        [0, focal_y, cy],
        [0,       0,  1]
    ], dtype=np.float32)

    # ---- Render poses (like Blender/LLFF) ----
    render_poses = torch.stack([
        pose_spherical(angle, -30.0, 4.0)
        for angle in np.linspace(-180, 180, 120 + 1)[:-1]
    ], 0)

    # ---- Compute near/far from camera distribution ----
    near, far = compute_near_far_from_poses(poses)

    # ---- Compute bounding box ----
    bounding_box = get_bbox3d_for_nerf_style(meta, H, W, near, far, device=device)

    return imgs, poses, render_poses, [H, W, focal], K, i_split, bounding_box, near, far