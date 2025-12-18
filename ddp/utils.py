import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import math
from run_nerf_helpers import compute_weights, sample_pdf, raw2outputs

import os
import os.path

import cv2

from ddp.transforms import (
    convert_depth_completion_scaling_to_m,
    convert_m_to_depth_completion_scaling,
    get_pretrained_normalize,
    resize_sparse_depth
)
from load_dataset.llff import load_colmap_sparse_depth
from ddp import resnet18_skip


def select_coordinates(coords, N_rand):
    coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
    select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
    select_coords = coords[select_inds].long()  # (N_rand, 2)
    return select_coords

def precompute_depth_sampling(depth):
    depth_min = (depth[:, 0] - 3. * depth[:, 1])
    depth_max = depth[:, 0] + 3. * depth[:, 1]
    return torch.stack((depth[:, 0], depth_min, depth_max), -1)

def precompute_quadratic_samples(near, far, num_samples):
    # normal parabola between 0.1 and 1, shifted and scaled to have y range between near and far
    start = 0.1
    x = torch.linspace(0, 1, num_samples)
    c = near
    a = (far - near)/(1. + 2. * start)
    b = 2. * start * a
    return a * x.pow(2) + b * x + c

def is_not_in_expected_distribution(depth_mean, depth_var, depth_measurement_mean, depth_measurement_std):
    delta_greater_than_expected = ((depth_mean - depth_measurement_mean).abs() - depth_measurement_std) > 0.
    var_greater_than_expected = depth_measurement_std.pow(2) < depth_var
    return torch.logical_or(delta_greater_than_expected, var_greater_than_expected)

def compute_ddp_depth_loss(depth_map, z_vals, weights, target_depth, target_valid_depth):
    pred_mean = depth_map[target_valid_depth]
    if pred_mean.shape[0] == 0:
        return torch.zeros((1,), device=depth_map.device, requires_grad=True)
    pred_var = ((z_vals[target_valid_depth] - pred_mean.unsqueeze(-1)).pow(2) * weights[target_valid_depth]).sum(-1) + 1e-5
    target_mean = target_depth[..., 0][target_valid_depth]
    target_std = target_depth[..., 1][target_valid_depth]
    apply_depth_loss = is_not_in_expected_distribution(pred_mean, pred_var, target_mean, target_std)
    pred_mean = pred_mean[apply_depth_loss]
    if pred_mean.shape[0] == 0:
        return torch.zeros((1,), device=depth_map.device, requires_grad=True)
    pred_var = pred_var[apply_depth_loss]
    target_mean = target_mean[apply_depth_loss]
    target_std = target_std[apply_depth_loss]
    f = nn.GaussianNLLLoss(eps=0.001)
    return float(pred_mean.shape[0]) / float(target_valid_depth.shape[0]) * f(pred_mean, target_mean, pred_var)

def raw2depth(raw, z_vals, rays_d):
    weights = compute_weights(raw, z_vals, rays_d)
    depth = torch.sum(weights * z_vals, -1)
    std = (((z_vals - depth.unsqueeze(-1)).pow(2) * weights).sum(-1)).sqrt()
    return depth, std

def sample_3sigma(low_3sigma, high_3sigma, N, det, near, far):
    t_vals = torch.linspace(0., 1., steps=N, device=get_device())
    step_size = (high_3sigma - low_3sigma) / (N - 1)
    bin_edges = (low_3sigma.unsqueeze(-1) * (1.-t_vals) + high_3sigma.unsqueeze(-1) * (t_vals)).clamp(near, far)
    factor = (bin_edges[..., 1:] - bin_edges[..., :-1]) / step_size.unsqueeze(-1)
    x_in_3sigma = torch.linspace(-3., 3., steps=(N - 1), device=get_device())
    bin_weights = factor * (1. / math.sqrt(2 * np.pi) * torch.exp(-0.5 * x_in_3sigma.pow(2))).unsqueeze(0).expand(*bin_edges.shape[:-1], N - 1)
    return sample_pdf(bin_edges, bin_weights, N, det=det)


def compute_samples_around_depth(raw, z_vals, rays_d, N_samples, perturb, lower_bound, near, far):
    sampling_depth, sampling_std = raw2depth(raw, z_vals, rays_d)
    sampling_std = sampling_std.clamp(min=lower_bound)
    depth_min = sampling_depth - 3. * sampling_std
    depth_max = sampling_depth + 3. * sampling_std
    return sample_3sigma(depth_min, depth_max, N_samples, perturb == 0., near, far)

def forward_with_additonal_samples(z_vals, raw, z_vals_2, rays_o, rays_d, viewdirs, network_fn, network_query_fn, raw_noise_std, pytest):
    pts_2 = rays_o[...,None,:] + rays_d[...,None,:] * z_vals_2[...,:,None]
    raw_2 = network_query_fn(pts_2, viewdirs, network_fn)
    z_vals = torch.cat((z_vals, z_vals_2), -1)
    raw = torch.cat((raw, raw_2), 1)
    z_vals, indices = z_vals.sort()
    raw = torch.gather(raw, 1, indices.unsqueeze(-1).expand_as(raw))
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, pytest=pytest)
    return {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map, 'depth_map' : depth_map, 'z_vals' : z_vals, 'weights' : weights}

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device

def save_depth_outputs(depth_m, std_m, save_path_base):
    np.save(save_path_base + ".npy", depth_m)
    depth_for_png = (depth_m / depth_m.max().clip(min=1e-6) * 65535.).astype(np.uint16)
    cv2.imwrite(save_path_base + ".png", depth_for_png)
    np.save(save_path_base + "_std.npy", std_m)

def load_ddp_depths(datadir, factor, bd_factor=.75, save_dir=None, invalidate_large_std_threshold=-1.):
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0) 

    device = get_device()

    images, depths, valid_depths = load_colmap_sparse_depth(datadir, factor, bd_factor)

    # prepare input
    orig_size = (depths.shape[1], depths.shape[2])
    input_size = (240, 320) # depth_completion_model was trained using this dimensions
    images_tmp = images.permute(0, 3, 1, 2)
    depths_tmp = depths[..., 0]
    images_tmp = torchvision.transforms.functional.resize(images_tmp, input_size, \
        interpolation=torchvision.transforms.functional.InterpolationMode.NEAREST)
    depths_tmp, valid_depths_tmp = resize_sparse_depth(depths_tmp, valid_depths, input_size)
    normalize, _ = get_pretrained_normalize()
    depths_tmp[valid_depths_tmp] = convert_m_to_depth_completion_scaling(depths_tmp[valid_depths_tmp])

    # run depth completion
    with torch.no_grad():
        net = resnet18_skip(pretrained=False, map_location=device, input_size=input_size).to(device)
        net.eval()
        ckpt = torch.load('weights/depth_completion_model.tar')
        missing_keys, unexpected_keys = net.load_state_dict(ckpt['network_state_dict'], strict=False)
        print("Loading model: \n  missing keys: {}\n  unexpected keys: {}".format(missing_keys, unexpected_keys))

        depths_out = torch.empty_like(depths_tmp)
        depths_std_out = torch.empty_like(depths_tmp)
        for i, (rgb, depth) in enumerate(zip(images_tmp, depths_tmp)):
            rgb = normalize['rgb'](rgb)
            input = torch.cat((rgb, depth.unsqueeze(0)), 0).unsqueeze(0)
            pred = net(input.to(device))
            depths_out[i] = convert_depth_completion_scaling_to_m(pred[0])
            depths_std_out[i] = convert_depth_completion_scaling_to_m(pred[1])
        depths_out = torch.stack((depths_out, depths_std_out), 1)
        depths_out = torchvision.transforms.functional.resize(depths_out, orig_size, \
            interpolation=torchvision.transforms.functional.InterpolationMode.NEAREST)

    # apply max min filter
    max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1)
    depths_out_0 = depths_out.narrow(1, 0, 1).clamp(min=0)
    depths_out_max = max_pool(depths_out_0) + 0.01
    depths_out_min = -1. * max_pool(-1. * depths_out_0) - 0.01
    depths_out[:, 1, :, :] = torch.maximum(depths_out[:, 1, :, :], (depths_out_max - depths_out_min).squeeze(1))

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        for i in range(depths_out.shape[0]):
            save_base = os.path.join(save_dir, f"depth_{i:03d}")
            save_depth_outputs(depths_out[i, 0].numpy(), depths_out[i, 1].numpy(), save_base)

    # mask out depth with very large uncertainty
    depths_out = depths_out.permute(0, 2, 3, 1)
    valid_depths_out = torch.full_like(valid_depths, True)
    if invalidate_large_std_threshold > 0.:
        large_std_mask = depths_out[:, :, :, 1] > invalidate_large_std_threshold
        valid_depths_out[large_std_mask] = False
        depths_out[large_std_mask] = 0.
        print("Masked out {:.1f} percent of completed depth with standard deviation greater {:.2f}".format( \
            100. * (1. - valid_depths_out.sum() / valid_depths_out.numel()), invalidate_large_std_threshold))

    return depths_out, valid_depths_out