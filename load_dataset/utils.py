import numpy as np
import torch

from utils.ray_utils import get_rays, get_ray_directions, get_ndc_rays


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def get_bbox3d_for_blenderobj(camera_transforms, H, W, near=2.0, far=6.0):
    camera_angle_x = float(camera_transforms['camera_angle_x'])
    focal = 0.5*W/np.tan(0.5 * camera_angle_x)

    # ray directions in camera coordinates
    directions = get_ray_directions(H, W, focal)

    min_bound = [100, 100, 100]
    max_bound = [-100, -100, -100]

    points = []

    for frame in camera_transforms["frames"]:
        c2w = torch.FloatTensor(frame["transform_matrix"]).to('cuda:0')
        rays_o, rays_d = get_rays(directions, c2w)
        
        def find_min_max(pt):
            for i in range(3):
                if(min_bound[i] > pt[i]):
                    min_bound[i] = pt[i]
                if(max_bound[i] < pt[i]):
                    max_bound[i] = pt[i]
            return

        for i in [0, W-1, H*W-W, H*W-1]:
            min_point = rays_o[i] + near*rays_d[i]
            max_point = rays_o[i] + far*rays_d[i]
            points += [min_point, max_point]
            find_min_max(min_point)
            find_min_max(max_point)

    return (torch.tensor(min_bound)-torch.tensor([1.0,1.0,1.0]), torch.tensor(max_bound)+torch.tensor([1.0,1.0,1.0]))


def get_bbox3d_for_llff(poses, hwf, near=0.0, far=1.0, no_ndc=False):
    H, W, focal = hwf
    H, W = int(H), int(W)
    
    # ray directions in camera coordinates
    directions = get_ray_directions(H, W, focal)

    min_bound = [100, 100, 100]
    max_bound = [-100, -100, -100]

    points = []
    poses = torch.FloatTensor(poses)
    for pose in poses:
        rays_o, rays_d = get_rays(directions, pose.to('cuda:0'))
        if not no_ndc:
            rays_o, rays_d = get_ndc_rays(H, W, focal, 1.0, rays_o, rays_d)

        def find_min_max(pt):
            for i in range(3):
                if(min_bound[i] > pt[i]):
                    min_bound[i] = pt[i]
                if(max_bound[i] < pt[i]):
                    max_bound[i] = pt[i]
            return

        for i in [0, W-1, H*W-W, H*W-1]:
            min_point = rays_o[i] + near*rays_d[i]
            max_point = rays_o[i] + far*rays_d[i]
            points += [min_point, max_point]
            find_min_max(min_point)
            find_min_max(max_point)

    return (torch.tensor(min_bound)-torch.tensor([0.1,0.1,0.0001]), torch.tensor(max_bound)+torch.tensor([0.1,0.1,0.0001]))


def get_bbox3d_for_nerf_style(meta, H, W, near, far, device='cuda:0'):
    """
    Compute a 3D bounding box for custom datasets similar to Blender/LLFF scenes.
    Uses near/far for ray sampling and applies aabb_scale correctly.
    """

    # focal length
    focal = meta.get("fl_x", None)
    if focal is None:
        camera_angle_x = float(meta["camera_angle_x"])
        focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

    # ray directions in camera coordinates
    directions = get_ray_directions(H, W, focal)  # (H,W,3)

    min_bound = [100.0, 100.0, 100.0]
    max_bound = [-100.0, -100.0, -100.0]

    frames = meta["frames"]

    def update_bounds(pt):
        for j in range(3):
            min_bound[j] = min(min_bound[j], float(pt[j]))
            max_bound[j] = max(max_bound[j], float(pt[j]))

    for frame in frames:
        c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32).to(device)

        rays_o, rays_d = get_rays(directions, c2w)  # both (H*W,3)

        # Use only 4 corner rays for efficiency
        idxs = [0, W-1, H*W-W, H*W-1]

        for idx in idxs:
            p_near = rays_o[idx] + near * rays_d[idx]
            p_far  = rays_o[idx] + far  * rays_d[idx]
            update_bounds(p_near)
            update_bounds(p_far)

    # Apply padding
    min_bound = torch.tensor(min_bound) - torch.tensor([0.1, 0.1, 0.1])
    max_bound = torch.tensor(max_bound) + torch.tensor([0.1, 0.1, 0.1])

    # Apply aabb_scale like instant-ngp:
    # final AABB is scaled from [-1,1] * scale
    aabb_scale = meta.get("aabb_scale", 1)
    scale = float(aabb_scale)

    center = (min_bound + max_bound) * 0.5
    size   = (max_bound - min_bound) * 0.5 * scale

    min_bound_scaled = center - size
    max_bound_scaled = center + size

    return min_bound_scaled, max_bound_scaled


def compute_near_far_from_poses(poses):
    centers = np.array([p[:3, 3] for p in poses])
    cmean = centers.mean(axis=0)
    dists = np.linalg.norm(centers - cmean, axis=1)

    min_d = dists.min()
    max_d = dists.max()

    near = max(0.05, min_d * 0.1)
    far  = max_d * 2.0
    return near, far

