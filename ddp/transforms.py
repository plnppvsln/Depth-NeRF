import numpy as np
import torch
from torchvision import transforms


def convert_depth_completion_scaling_to_m(depth):
    # convert from depth completion scaling to meter, that means map range 0 .. 1 to range 0 .. 16,38m
    return depth * (2 ** 16 - 1) / 4000.

def convert_m_to_depth_completion_scaling(depth):
    # convert from meter to depth completion scaling, which maps range 0 .. 16,38m to range 0 .. 1
    return depth * 4000. / (2 ** 16 - 1)

def get_normalize(mean, std):
    normalize = transforms.Normalize(mean=mean, std=std)
    unnormalize = transforms.Normalize(mean=np.divide(-mean, std), std=(1. / std))
    return normalize, unnormalize

def get_pretrained_normalize():
    normalize = dict()
    unnormalize = dict()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalize['rgb'], unnormalize['rgb'] = get_normalize(mean, std)
    normalize['rgbd'], unnormalize['rgbd'] = get_normalize(np.concatenate((mean, [0.,]), axis=0), np.concatenate((std, [1.,]), axis=0))
    return normalize, unnormalize

def resize_sparse_depth(depths, valid_depths, size):
    device = depths.device
    orig_size = (depths.shape[1], depths.shape[2])
    col, row = torch.meshgrid(torch.tensor(range(orig_size[1])), torch.tensor(range(orig_size[0])), indexing='ij')
    rowcol2rowcol = torch.stack((row.t(), col.t()), -1)
    rowcol2rowcol = rowcol2rowcol.unsqueeze(0).expand(depths.shape[0], -1, -1, -1)
    image_index = torch.arange(depths.shape[0]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, orig_size[0], orig_size[1], 1)
    rowcol2rowcol = torch.cat((image_index, rowcol2rowcol), -1)
    factor_h, factor_w = float(size[0]) / float(orig_size[0]), float(size[1]) / float(orig_size[1])
    depths_out = torch.zeros((depths.shape[0], size[0], size[1]), device=device)
    valid_depths_out = torch.zeros_like(depths_out).bool()
    idx_row_col = rowcol2rowcol[valid_depths].to(device)
    idx_row_col_resized = idx_row_col
    scale = torch.tensor((1., factor_h, factor_w), device=idx_row_col.device)
    idx_row_col_resized = ((idx_row_col + 0.5) * scale).long()    
    depths_out[idx_row_col_resized[..., 0], idx_row_col_resized[..., 1], idx_row_col_resized[..., 2]] \
        = depths[idx_row_col[..., 0], idx_row_col[..., 1], idx_row_col[..., 2]]
    valid_depths_out[idx_row_col_resized[..., 0], idx_row_col_resized[..., 1], idx_row_col_resized[..., 2]] = True
    return depths_out, valid_depths_out

