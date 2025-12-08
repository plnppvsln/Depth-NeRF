# Author: Yash Bhalgat

from math import exp, log, floor
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

from utils import hash

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def total_variation_loss(embeddings, min_resolution, max_resolution, level, log2_hashmap_size, n_levels=16):
    # Get resolution
    b = exp((log(max_resolution)-log(min_resolution))/(n_levels-1))
    resolution = torch.tensor(floor(min_resolution * b**level))

    # Cube size to apply TV loss
    min_cube_size = min_resolution - 1
    max_cube_size = 50 # can be tuned
    if min_cube_size > max_cube_size:
        print("ALERT! min cuboid size greater than max!")
        pdb.set_trace()
    cube_size = torch.floor(torch.clip(resolution/10.0, min_cube_size, max_cube_size)).int()

    # Sample cuboid
    min_vertex = torch.randint(0, resolution-cube_size, (3,))
    idx = min_vertex + torch.stack([torch.arange(cube_size+1) for _ in range(3)], dim=-1)
    cube_indices = torch.stack(torch.meshgrid(idx[:,0], idx[:,1], idx[:,2], indexing='ij'), dim=-1)

    hashed_indices = hash(cube_indices, log2_hashmap_size)
    cube_embeddings = embeddings(hashed_indices)
    #hashed_idx_offset_x = hash(idx+torch.tensor([1,0,0]), log2_hashmap_size)
    #hashed_idx_offset_y = hash(idx+torch.tensor([0,1,0]), log2_hashmap_size)
    #hashed_idx_offset_z = hash(idx+torch.tensor([0,0,1]), log2_hashmap_size)

    # Compute loss
    #tv_x = torch.pow(embeddings(hashed_idx)-embeddings(hashed_idx_offset_x), 2).sum()
    #tv_y = torch.pow(embeddings(hashed_idx)-embeddings(hashed_idx_offset_y), 2).sum()
    #tv_z = torch.pow(embeddings(hashed_idx)-embeddings(hashed_idx_offset_z), 2).sum()
    tv_x = torch.pow(cube_embeddings[1:,:,:,:]-cube_embeddings[:-1,:,:,:], 2).sum()
    tv_y = torch.pow(cube_embeddings[:,1:,:,:]-cube_embeddings[:,:-1,:,:], 2).sum()
    tv_z = torch.pow(cube_embeddings[:,:,1:,:]-cube_embeddings[:,:,:-1,:], 2).sum()

    return (tv_x + tv_y + tv_z)/cube_size

def sigma_sparsity_loss(sigmas):
    # Using Cauchy Sparsity loss on sigma values
    return torch.log(1.0 + 2*sigmas**2).sum(dim=-1)


def depth_loss(rendered_depth, target_depth, 
               weighted_loss=False, relative_loss=False, normalize_depth=False,
               ray_weights=None, max_depth=None):
    """
    Compute depth loss between rendered depth and target depth.
    
    Supports multiple loss modes:
    - Standard MSE: Mean squared error between rendered and target depth
    - Weighted MSE: MSE weighted by ray_weights (e.g., reprojection error)
    - Relative loss: Scale-invariant relative error
    - Normalized loss: Normalize depth difference by max_depth
    
    Args:
        rendered_depth: [N_rays] rendered depth values
        target_depth: [N_rays] target depth values
        weighted_loss: bool, if True use weighted MSE loss (requires ray_weights)
        relative_loss: bool, if True use relative loss (normalize by target_depth)
        normalize_depth: bool, if True normalize depth difference by max_depth (requires max_depth)
        ray_weights: [N_rays] optional weights for each ray (required if weighted_loss=True)
        max_depth: float, optional maximum depth for normalization (required if normalize_depth=True)
    
    Returns:
        loss: scalar depth loss value
    
    Raises:
        ValueError: if required parameters are missing for selected loss mode
        AssertionError: if shapes don't match
    """
    # Validate shapes
    assert rendered_depth.shape == target_depth.shape, \
        f"Shape mismatch: rendered_depth {rendered_depth.shape} vs target_depth {target_depth.shape}"
    
    # Compute depth difference
    depth_diff = rendered_depth - target_depth
    
    if weighted_loss:
        # Weighted MSE loss with optional normalization
        if ray_weights is None:
            raise ValueError("ray_weights must be provided when weighted_loss=True")
        
        if ray_weights.shape != rendered_depth.shape:
            raise ValueError(f"ray_weights shape {ray_weights.shape} must match depth shape {rendered_depth.shape}")
        
        if normalize_depth:
            # Normalize by max_depth to handle scale differences
            if max_depth is None:
                raise ValueError("max_depth must be provided when normalize_depth=True")
            
            # Convert max_depth to float if it's a numpy scalar
            if isinstance(max_depth, np.ndarray):
                max_depth = float(max_depth.item())
            elif not isinstance(max_depth, (int, float)):
                max_depth = float(max_depth)
            
            normalization_factor = max_depth + 1e-8  # Add epsilon to avoid division by zero
            depth_diff_normalized = depth_diff / normalization_factor
            loss = torch.mean((depth_diff_normalized ** 2) * ray_weights)
        else:
            # Standard weighted MSE
            loss = torch.mean((depth_diff ** 2) * ray_weights)
            
    elif relative_loss:
        # Relative loss: normalize by target depth to handle scale-invariant errors
        # Add epsilon to avoid division by zero for small/zero depths
        epsilon = 1e-6
        depth_relative = depth_diff / (target_depth + epsilon)
        loss = torch.mean(depth_relative ** 2)
        
    else:
        # Standard MSE loss
        loss = torch.mean((depth_diff ** 2))
    
    return loss

class SigmaLoss:
    def __init__(self, N_samples, perturb, raw_noise_std):
        super(SigmaLoss, self).__init__()
        self.N_samples = N_samples
        self.perturb = perturb
        self.raw_noise_std = raw_noise_std

    def calculate_loss(self, rays_o, rays_d, viewdirs, near, far, depths, run_func, network, err=1):
        raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

        N_rays = rays_o.shape[0]
        t_vals = torch.linspace(0., 1., steps=self.N_samples).to(device)
        t_vals = t_vals.expand([N_rays, self.N_samples])
        z_vals = near * (1.-t_vals) + far * (t_vals)
        if self.perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            upper = torch.cat([mids, z_vals[...,-1:]], -1)
            lower = torch.cat([z_vals[...,:1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(device)

            z_vals = lower + (upper - lower) * t_rand
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
        raw = run_func(pts, viewdirs, network)

        noise = 0.
        if self.raw_noise_std > 0.:
            noise = torch.randn(raw[...,3].shape) * self.raw_noise_std

        dists = z_vals[...,1:] - z_vals[...,:-1]
        dists = torch.cat([dists, torch.Tensor([1e10]).to(device).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

        dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

        # sigma = F.relu(raw[...,3] + noise)
        alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
        
        
        loss = -torch.log(weights + 1e-5) * torch.exp(-(z_vals - depths[:,None]) ** 2 / (2 * err)) * dists
        loss = torch.sum(loss, dim=1)
        
        return loss
