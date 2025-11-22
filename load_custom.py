import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2

from utils import get_bbox3d_for_blenderobj, pose_spherical

def load_custom_data(basedir, half_res=False, testskip=8):
    with open(os.path.join(basedir, "transforms.json"), 'r') as f:
        meta = json.load(f)

    # Извлекаем параметры камеры
    H = int(meta['h'])
    W = int(meta['w'])
    fl_x = meta.get('fl_x', None)
    fl_y = meta.get('fl_y', None)
    cx = meta.get('cx', W/2)
    cy = meta.get('cy', H/2)
    focal = fl_x if fl_x is not None else .5 * W / np.tan(.5 * meta['camera_angle_x'])

    # Собираем изображения и позы
    imgs = []
    poses = []
    test_indices = []
    for i, frame in enumerate(meta['frames']):
        fname = os.path.join(basedir, frame['file_path'])
        if not os.path.exists(fname):
            print(f"Warning: file {fname} not found, skipping.")
            continue
        imgs.append(imageio.imread(fname))
        poses.append(np.array(frame['transform_matrix'], dtype=np.float32))
        if (i + 1) % testskip == 0:
            test_indices.append(len(imgs)-1)  # индекс в imgs/poses

    imgs = (np.array(imgs) / 255.).astype(np.float32)
    poses = np.array(poses)

    # Формируем индексы для train/val/test
    all_indices = np.arange(len(imgs))
    test_indices = np.array(test_indices)
    train_indices = np.array([i for i in all_indices if i not in test_indices])
    val_indices = test_indices.copy()

    i_split = [train_indices, val_indices, test_indices]

    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2.
        imgs_half_res = np.zeros((imgs.shape[0], H, W, imgs.shape[-1]))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res

    K = np.array([
        [focal, 0, cx],
        [0, focal, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,120+1)[:-1]], 0)

    bounding_box = None  # если нужно

    return imgs, poses, render_poses, [H, W, focal], K, i_split, bounding_box