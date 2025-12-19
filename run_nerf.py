import os, sys
from datetime import datetime
import numpy as np
import imageio
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
import pickle

import matplotlib.pyplot as plt

from run_nerf_helpers import *
from utils.optimizer import MultiOptimizer # Возможно вообще не нужно
from utils.radam import RAdam # TODO заменить на torch.optim.RAdam
from losses.loss import sigma_sparsity_loss, total_variation_loss, SigmaLoss, depth_loss

from load_dataset.data import RayDataset
from torch.utils.data import DataLoader

from load_dataset.llff import load_llff_data, load_colmap_depth, load_colmap_llff
from load_dataset.deepvoxels import load_dv_data # Возможно вообще не нужно
from load_dataset.blender import load_blender_data
from load_dataset.scannet import load_scannet_data # Возможно вообще не нужно
from load_dataset.LINEMOD import load_LINEMOD_data # Возможно вообще не нужно
from load_dataset.nerf_style import load_nerf_style

from losses.sparse_depth_loss import local_depth_ranking_loss, spatial_continuity_loss # SparseNeRF depth ranking and continuity losses
from ddp.utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = True


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded, keep_mask = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs_flat[~keep_mask, -1] = 0 # set sigma to 0 for invalid points
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024*32, use_viewdirs=False, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], use_viewdirs, **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None, rays_depth=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
       camera while using other c2w argument for viewing directions.
      depths: #TODO 
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      depth_map: [batch_size]. Depth map.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    elif rays.shape[0] == 2:
        # use provided ray batch
        rays_o, rays_d = rays
    else:
        # use provided ray batch
        rays_o, rays_d, rays_depth = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)
    if rays_depth is not None:
        rays_depth = torch.reshape(rays_depth, [-1,3]).float()
        rays = torch.cat([rays, rays_depth], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, use_viewdirs, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'depth_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):

    H, W, focal = hwf
    near, far = render_kwargs['near'], render_kwargs['far']

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor
        K = K.copy()
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])
        
    rgbs = []
    depths = []
    disps = []
    psnrs = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        rgb, disp, depth, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        # normalize depth to [0,1]
        depth_norm = (depth - near) / (far - near)
        depths.append(depth_norm.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i==0:
            tqdm.write(f"[Render] rgb shape: {rgb.shape}, depth shape: {depth.shape}")

        # tqdm.write(f"[Render] Iter: {i} Time: {time.time() - t:.4f}")
        t = time.time()

        if gt_imgs is not None and render_factor==0:
            try:
                gt_img = gt_imgs[i].cpu().numpy()
            except:
                gt_img = gt_imgs[i]
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_img)))
            tqdm.write(f"{p}")
            psnrs.append(p)

        if savedir is not None:

            # Можно разные варианты сохранения изображений добавить
            # Здесь: первый одной картинкой сохраняет, второй - двумя разными

            # # save rgb and depth as a figure
            # fig = plt.figure(figsize=(25,15))
            # ax = fig.add_subplot(1, 2, 1)
            # rgb8 = to8b(rgbs[-1])
            # ax.imshow(rgb8)
            # ax.axis('off')
            # ax = fig.add_subplot(1, 2, 2)
            # ax.imshow(depths[-1], cmap='plasma', vmin=0, vmax=1)
            # ax.axis('off')
            # filename = os.path.join(savedir, '{:03d}.png'.format(i))
            # # save as png
            # plt.savefig(filename, bbox_inches='tight', pad_inches=0)
            # plt.close(fig)
            # # imageio.imwrite(filename, rgb8)
            
            rgb8 = to8b(rgbs[-1])
            rgb8[np.isnan(rgb8)] = 0
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)
            depth = depth.cpu().numpy()
            tqdm.write(f"max depth: {np.nanmax(depth)}")
            depth_valid = depth[~np.isnan(depth)]
            if len(depth_valid) > 0:
                depth_min = np.nanmin(depth)
                depth_max = np.nanmax(depth)
                if depth_max > depth_min:
                    depth_normalized = (depth - near) / (far - near)
                else:
                    depth_normalized = np.zeros_like(depth)
                depth_normalized = np.clip(depth_normalized, 0, 1)
                depth_normalized[np.isnan(depth_normalized)] = 0
                depth8 = (255 * depth_normalized).astype(np.uint8)
            else:
                depth8 = np.zeros_like(depth, dtype=np.uint8)
            imageio.imwrite(os.path.join(savedir, '{:03d}_depth.png'.format(i)), depth8)
            np.savez(os.path.join(savedir, '{:03d}.npz'.format(i)), rgb=rgb.cpu().numpy(), disp=disp.cpu().numpy(), acc=acc.cpu().numpy(), depth=depth)



    rgbs = np.stack(rgbs, 0)
    depths = np.stack(depths, 0)
    disps = np.stack(disps, 0)
    if gt_imgs is not None and render_factor==0:
        avg_psnr = sum(psnrs)/len(psnrs)
        print("Avg PSNR over Test set: ", avg_psnr)
        with open(os.path.join(savedir, "test_psnrs_avg{:0.2f}.pkl".format(avg_psnr)), "wb") as fp:
            pickle.dump(psnrs, fp)

    return rgbs, depths, disps


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args, i=args.i_embed)
    if args.i_embed==1:
        # hashed embedding table
        embedding_params = list(embed_fn.parameters())

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        # if using hashed for xyz, use SH for views
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args, i=args.i_embed_views)

    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]

    if args.i_embed==1:
        model = NeRFSmall(num_layers=2,
                        hidden_dim=64,
                        geo_feat_dim=15,
                        num_layers_color=3,
                        hidden_dim_color=64,
                        input_ch=input_ch, input_ch_views=input_ch_views)
    else:
        model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs)
    if args.multi_gpu:
        model = nn.DataParallel(model).to(device)
    else:
        model = model.to(device)
    grad_vars = list(model.parameters())

    model_fine = None

    if args.N_importance > 0:
        if args.i_embed==1:
            model_fine = NeRFSmall(num_layers=2,
                        hidden_dim=64,
                        geo_feat_dim=15,
                        num_layers_color=3,
                        hidden_dim_color=64,
                        input_ch=input_ch, input_ch_views=input_ch_views)
        else:
            model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs)
        if args.multi_gpu:
            model_fine = nn.DataParallel(model_fine).to(device)
        else:
            model = model.to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk*args.n_gpus)

    # Create optimizer
    if args.i_embed==1:
        optimizer = RAdam([
                            {'params': grad_vars, 'weight_decay': 1e-6},
                            {'params': embedding_params, 'eps': 1e-15}
                        ], lr=args.lrate, betas=(0.9, 0.99))
    else:
        optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])
        if args.i_embed==1:
            embed_fn.load_state_dict(ckpt['embed_fn_state_dict'])

    ##########################
    # pdb.set_trace()

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'embed_fn': embed_fn,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def render_rays(ray_batch,
                use_viewdirs,
                network_fn,
                network_query_fn,
                N_samples,
                precomputed_z_samples=None,
                embed_fn=None,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False,
                sigma_loss=None):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = None
    depth_range = None
    if use_viewdirs:
        viewdirs = ray_batch[:,8:11]
        if ray_batch.shape[-1] > 11:
            depth_range = ray_batch[:,11:14]
    else:
        if ray_batch.shape[-1] > 8:
            depth_range = ray_batch[:,8:11]
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]
    t_vals = torch.linspace(0., 1., steps=N_samples)

    # This will work only for ddp
    # sample and render rays for dense depth priors for nerf
    N_samples_half = N_samples // 2
    if precomputed_z_samples is not None:
        # compute a lower bound for the sampling standard deviation as the maximal distance between samples
        lower_bound = precomputed_z_samples[-1] - precomputed_z_samples[-2]
    # train time: use precomputed samples along the whole ray and additionally sample around the depth
    if depth_range is not None:
        valid_depth = depth_range[:,0] >= near[0, 0]
        invalid_depth = valid_depth.logical_not()
        # do a forward pass for the precomputed first half of samples
        z_vals = precomputed_z_samples.unsqueeze(0).expand((N_rays, N_samples_half))
        if perturb > 0.:
            z_vals = perturb_z_vals(z_vals, pytest)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]
        raw = network_query_fn(pts, viewdirs, network_fn)
        z_vals_2 = torch.empty_like(z_vals)
        # sample around the predicted depth from the first half of samples, if the input depth is invalid
        z_vals_2[invalid_depth] = compute_samples_around_depth(raw.detach()[invalid_depth], z_vals[invalid_depth], rays_d[invalid_depth], N_samples_half, perturb, lower_bound, near[0, 0], far[0, 0])
        # sample with in 3 sigma of the input depth, if it is valid
        z_vals_2[valid_depth] = sample_3sigma(depth_range[valid_depth, 1], depth_range[valid_depth, 2], N_samples_half, perturb == 0., near[0, 0], far[0, 0])
        return forward_with_additonal_samples(z_vals, raw, z_vals_2, rays_o, rays_d, viewdirs, network_fn, network_query_fn, raw_noise_std, pytest)
    # test time: use precomputed samples along the whole ray and additionally sample around the predicted depth from the first half of samples
    elif precomputed_z_samples is not None:
        z_vals = precomputed_z_samples.unsqueeze(0).expand((N_rays, N_samples_half))
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]
        raw = network_query_fn(pts, viewdirs, network_fn)
        z_vals_2 = compute_samples_around_depth(raw, z_vals, rays_d, N_samples_half, perturb, lower_bound, near[0, 0], far[0, 0])
        return forward_with_additonal_samples(z_vals, raw, z_vals_2, rays_o, rays_d, viewdirs, network_fn, network_query_fn, raw_noise_std, pytest)

    # sample and render rays for nerf
    elif not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        z_vals = perturb_z_vals(z_vals, pytest)

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]

    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    if N_importance > 0:

        rgb_map_0, disp_map_0, depth_map_0, acc_map_0, z_vals_0, weights_0 = rgb_map, disp_map, depth_map, acc_map, z_vals, weights

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
        # raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'depth_map' : depth_map, 'acc_map' : acc_map, 'z_vals' : z_vals, 'weights' : weights}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['depth0'] = depth_map_0
        ret['acc0'] = acc_map_0
        # ret['sparsity_loss0'] = sparsity_loss_0
        ret['z_vals0'] = z_vals_0
        ret['weights0'] = weights_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    if sigma_loss is not None and ray_batch.shape[-1] > 11:
        depths = ray_batch[:,8]
        ret['sigma_loss'] = sigma_loss.calculate_loss(rays_o, rays_d, viewdirs, near, far, depths, network_query_fn, network_fine)


    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            tqdm.write(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=1,
                        help='set 1 for hashed embedding, 0 for default positional encoding, 2 for spherical')
    parser.add_argument("--i_embed_views", type=int, default=2,
                        help='set 1 for hashed embedding, 0 for default positional encoding, 2 for spherical')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')
    parser.add_argument("--multi_gpu", action='store_true',
                        help='will use multiple gpu\'s if possible')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels / custom')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## scannet flags
    parser.add_argument("--scannet_sceneID", type=str, default='scene0000_00',
                        help='sceneID to load from scannet')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=6000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=1000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=7000,
                        help='frequency of render_poses video saving')

    parser.add_argument("--finest_res",   type=int, default=512,
                        help='finest resolultion for hashed embedding')
    parser.add_argument("--log2_hashmap_size",   type=int, default=19,
                        help='log2 of hashmap size')
    parser.add_argument("--tv-loss-weight", type=float, default=1e-6,
                        help='learning rate')

    # new experiment by kangle
    parser.add_argument("--N_iters", type=int, default=30_000, 
                        help='number of iters')
    parser.add_argument("--alpha_model_path", type=str, default=None,
                        help='predefined alpha model')
    parser.add_argument("--no_coarse", action='store_true',
                        help="Remove coarse network.")
    parser.add_argument("--train_scene", nargs='+', type=int,
                        help='id of scenes used to train')
    parser.add_argument("--test_scene", nargs='+', type=int,
                        help='id of scenes used to test')
    parser.add_argument("--colmap_depth", action='store_true',
                        help="Use depth supervision by colmap.")
    parser.add_argument("--depth_loss", action='store_true',
                        help="Use depth supervision by colmap - depth loss.")
    parser.add_argument("--using_sparse", action='store_true',
                        help="When enabled, all images are used for training (no test set).")
    parser.add_argument("--depth_lambda", type=float, default=0.1,
                        help="Depth lambda used for loss.")
    parser.add_argument("--sigma_loss", action='store_true',
                        help="Use depth supervision by colmap - sigma loss.")
    parser.add_argument("--sigma_lambda", type=float, default=0.1,
                        help="Sigma lambda used for loss.")
    parser.add_argument("--weighted_loss", action='store_true',
                        help="Use weighted loss by reprojection error.")
    parser.add_argument("--relative_loss", action='store_true',
                        help="Use relative loss.")
    parser.add_argument("--depth_with_rgb", action='store_true',
                    help="single forward for both depth and rgb")
    parser.add_argument("--normalize_depth", action='store_true',
                    help="normalize depth before calculating loss")
    parser.add_argument("--depth_rays_prop", type=float, default=0.5,
                        help="Proportion of depth rays.")
    
    #SparseNerf paramenters parser
    parser.add_argument("--use_dpt_ranking", action='store_true',
                    help="Use DPT depth for ranking + continuity loss")
    parser.add_argument("--lambda_rank", type=float, default=0.2,
                        help="Weight for depth ranking loss")
    parser.add_argument("--lambda_cont", type=float, default=0.02,
                        help="Weight for spatial continuity loss")

    #DDPNerf paramenters parser
    parser.add_argument("--use_ddp", action='store_true',
                    help="Use DDP method for sampling")
    parser.add_argument("--ddp_depth_loss_weight", type=float, default=0.004,
                        help='weight of the depth loss, values <=0 do not apply depth loss')
    parser.add_argument("--invalidate_large_std_threshold", type=float, default=1.,
                        help='invalidate completed depth values with standard deviation greater than threshold in m, \
                            thresholds <=0 deactivate invalidation')
    parser.add_argument("--save_depth_completion_dir", type=str, default=None,
                        help='path to save ddp depth completion (None by default)')

    parser.add_argument("--eval", action='store_true',
                    help="Turns on evaluation mode (use only with --render_only)")

    return parser


def train():

    parser = config_parser()
    args = parser.parse_args()

    # Multi-GPU
    args.n_gpus = torch.cuda.device_count()
    print(f"Using {args.n_gpus} GPU(s).")

    # Load data
    K = None
    if args.dataset_type == 'llff':
        if args.colmap_depth:
            depth_gts = load_colmap_depth(args.datadir, factor=args.factor, bd_factor=.75)
        if args.use_ddp:
            dense_depths, valid_depths = load_ddp_depths(args.datadir, factor=args.factor, bd_factor=.75, save_dir=args.save_depth_completion_dir)
        images, poses, render_poses, i_test, bounding_box, near, far, dpt_depths = load_llff_data(args.datadir, args.factor,
                                                                                      recenter=True, bd_factor=.75,
                                                                                      spherify=args.spherify,
                                                                                      no_ndc=args.no_ndc,
                                                                                      use_dpt_ranking=args.use_dpt_ranking) # dpt_depths for SparseNeRF
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        args.bounding_box = bounding_box
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)

        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]
        
        # Если using_sparse включен, используем все изображения для обучения
        if args.using_sparse:
            print('using_sparse enabled: using all images for training (no test set)')
            i_test = []

        # Преобразуем i_test в numpy array для консистентности
        i_test = np.array(i_test)

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        if args.eval:
            i_test = i_train

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split, bounding_box = load_blender_data(args.datadir, args.half_res, args.testskip)
        args.bounding_box = bounding_box
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'scannet':
        images, poses, render_poses, hwf, i_split, bounding_box = load_scannet_data(args.datadir, args.scannet_sceneID, args.half_res)
        args.bounding_box = bounding_box
        print('Loaded scannet', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 0.1
        far = 10.0

    elif args.dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res, args.testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.
    elif args.dataset_type == 'nerf':
        images, poses, render_poses, hwf, K, i_split, bounding_box, near, far = load_nerf_style(args.datadir, 
                                                                                                half_res=args.half_res, 
                                                                                                testskip=args.testskip)
        args.bounding_box = bounding_box
        print('Loaded custom dataset', images.shape, poses.shape, render_poses.shape, hwf, args.datadir)
        print(f'Computed near/far from poses: near={near:.3f}, far={far:.3f}')
        i_train, i_val, i_test = i_split

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if args.render_test:
        if len(i_test) > 0:
            render_poses = np.array(poses[i_test])
        else:
            print('Warning: render_test=True but no test images available. Using render_poses spiral path instead.')

    # Create log dir and copy the config file
    basedir = args.basedir
    if args.i_embed==1:
        args.expname += "_hashXYZ"
    elif args.i_embed==0:
        args.expname += "_posXYZ"
    # if args.i_embed_views==2:
    #     args.expname += "_sphereVIEW"
    # elif args.i_embed_views==0:
    #     args.expname += "_posVIEW"
    # if args.colmap_depth:
    #     args.expname += "_ds"        
    args.expname += "_fine"+str(args.finest_res) + "_log2T"+str(args.log2_hashmap_size)
    args.expname += "_lr"+str(args.lrate) + "_decay"+str(args.lrate_decay)
    args.expname += "_RAdam"
    # if args.sparse_loss_weight > 0:
    #     args.expname += "_sparse" + str(args.sparse_loss_weight)
    args.expname += "_TV" + str(args.tv_loss_weight)
    # TODO лучше это дело записать в отдельный файл
    expname = args.expname

    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }

    if args.use_ddp and args.ddp_depth_loss_weight > 0.:
        precomputed_z_samples = precompute_quadratic_samples(near, far, args.N_samples // 2)

        if precomputed_z_samples.shape[0] % 2 == 1:
            precomputed_z_samples = precomputed_z_samples[:-1]

        print("Computed {} samples between {} and {}".format(precomputed_z_samples.shape[0], precomputed_z_samples[0], precomputed_z_samples[-1]))
    else:
        precomputed_z_samples = None

    scene_sample_params = {
        'precomputed_z_samples' : precomputed_z_samples,
    }


    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    render_kwargs_train.update(scene_sample_params)
    render_kwargs_test.update(scene_sample_params)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        tqdm.write(f"[RENDER ONLY]")
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                if len(i_test) > 0:
                    images = images[i_test]
                else:
                    print('Warning: render_test=True but no test images available. Using render_poses path instead.')
                    images = None
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            tqdm.write(f"[RENDER ONLY] Test poses shape: {render_poses.shape}")
            # TODO DS nerf uses render_test_ray() here 
            rgbs, pred_depths, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
            tqdm.write(f"[RENDER ONLY] Done rendering {testsavedir}")
            imageio.mimwrite(os.path.join(testsavedir, 'rgb.mp4'), to8b(rgbs), fps=30, quality=8)
            # disps[np.isnan(disps)] = 0
            print('Depth stats', np.mean(pred_depths), np.max(pred_depths), np.percentile(pred_depths, 95))
            imageio.mimwrite(os.path.join(testsavedir, 'depth.mp4'), to8b(pred_depths / np.percentile(pred_depths, 95)), fps=30, quality=8)

            if args.eval:
                import glob
                img_dir = os.path.join(args.datadir, 'images_4')
                if os.path.exists(img_dir):
                    # Для LLFF датасета
                    img_files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')) + 
                                        glob.glob(os.path.join(img_dir, '*.JPG')) +
                                        glob.glob(os.path.join(img_dir, '*.png')) +
                                        glob.glob(os.path.join(img_dir, '*.PNG')))
                    img_names = [os.path.basename(f) for f in img_files]
                else:
                    # Если папка images не найдена, используем нумерацию
                    img_names = [f'{i:03d}.png' for i in range(len(images))]
                gt_dir = os.path.join(args.datadir, 'images_4')
                os.makedirs(gt_dir, exist_ok=True)
                for i, gt_img in enumerate(images):
                    try:
                        gt_img_np = gt_img.cpu().numpy() if hasattr(gt_img, 'cpu') else gt_img
                        # Уменьшаем ground truth изображение с тем же render_factor
                        if args.render_factor > 1:
                            from skimage.transform import resize
                            H_orig, W_orig = gt_img_np.shape[:2]
                            H_new = H_orig // args.render_factor
                            W_new = W_orig // args.render_factor
                            gt_img_np = resize(gt_img_np, (H_new, W_new), order=5, anti_aliasing=True)
                            # gt_img_np = torchvision.transforms.functional.resize(gt_img_np, (H_new, W_new), \
                            # interpolation=torchvision.transforms.functional.InterpolationMode.NEAREST)
                        imageio.imwrite(os.path.join(gt_dir, f'{i:03d}.png'), to8b(gt_img_np))
                    except Exception as e:
                        print(f"Warning: Could not save ground truth image {i}: {e}")
                
                # Run evaluation script
                import subprocess
                eval_cmd = [
                    'python', 'scripts/eval.py',
                    '--ground_truth', gt_dir,
                    '--predicted', testsavedir,
                    '-o', os.path.join(basedir, expname, 'metrics.txt'),
                    '-n', expname
                ]
                print(f"Running evaluation: {' '.join(eval_cmd)}")
                try:
                    subprocess.run(eval_cmd, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Evaluation failed: {e}")
                
                # Удаляем predicted изображения из основной папки (только файлы 000.png, 001.png, etc.)
                try:
                    for f in os.listdir(gt_dir):
                        if f.endswith('.png') and f[:-4].isdigit() and len(f[:-4]) == 3:
                            # Это файл вида 000.png, 001.png, etc.
                            os.remove(os.path.join(gt_dir, f))
                    print(f"Cleaned up predicted images from: {gt_dir}")
                except Exception as e:
                    print(f"Warning: Could not remove predicted images from {gt_dir}: {e}")


            return
    
    # Prepare raybatch tensor if batching random rays
    # N_rand = args.N_rand N_rand меняем на N_rgb и N_depth
    if not args.colmap_depth:
        N_rgb = args.N_rand
    else:
        N_depth = int(args.N_rand * args.depth_rays_prop)
        N_rgb = args.N_rand - N_depth
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)

        rays_depth = None
        if args.colmap_depth:
            print('get depth rays')
            rays_depth_list = []
            for i in i_train:
                rays_depth = np.stack(get_rays_by_coord_np(H, W, focal, poses[i,:3,:4], depth_gts[i]['coord']), axis=0) # 2 x N x 3
                # print(rays_depth.shape)
                rays_depth = np.transpose(rays_depth, [1,0,2])
                depth_value = np.repeat(depth_gts[i]['depth'][:,None,None], 3, axis=2) # N x 1 x 3
                weights = np.repeat(depth_gts[i]['error'][:,None,None], 3, axis=2) # N x 1 x 3
                rays_depth = np.concatenate([rays_depth, depth_value, weights], axis=1) # N x 4 x 3
                rays_depth_list.append(rays_depth)

            rays_depth = np.concatenate(rays_depth_list, axis=0)
            print('rays_weights mean:', np.mean(rays_depth[:,3,0]))
            print('rays_weights std:', np.std(rays_depth[:,3,0]))
            print('rays_weights max:', np.max(rays_depth[:,3,0]))
            print('rays_weights min:', np.min(rays_depth[:,3,0]))
            print('rays_depth.shape:', rays_depth.shape)
            rays_depth = rays_depth.astype(np.float32)
            print('shuffle depth rays')
            np.random.shuffle(rays_depth)

            max_depth = np.max(rays_depth[:,3,0])
        print('done')
        i_batch = 0

    # Move training data to GPU
    images = torch.Tensor(images).to(device)

    # DDP
    if args.use_ddp:
        dense_depths = dense_depths.to(device)
        valid_depths = valid_depths.to(device)

    #SparseNerf
    if args.use_dpt_ranking:
        dpt_depths = [torch.from_numpy(d).to(device) for d in dpt_depths]  # list of [H,W]
    else:
        dpt_depths = None



    poses = torch.Tensor(poses).to(device)
    if use_batching:
        # rays_rgb = torch.Tensor(rays_rgb).to(device)
        # rays_depth = torch.Tensor(rays_depth).to(device) if rays_depth is not None else None
        raysRGB_iter = iter(DataLoader(RayDataset(rays_rgb), 
                                            batch_size = N_rgb, 
                                            shuffle=True, 
                                            num_workers=0, 
                                            generator=torch.Generator(device='cuda')))
        raysDepth_iter = iter(DataLoader(RayDataset(rays_depth), 
                                              batch_size = N_depth, 
                                              shuffle=True, 
                                              num_workers=0, 
                                              generator=torch.Generator(device='cuda'))) if rays_depth is not None else None



    N_iters = args.N_iters + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))

    loss_list = []
    psnr_list = []
    time_list = []
    start = start + 1
    time0 = time.time()
    for i in trange(start, N_iters):
        # Sample random ray batch
        if use_batching:
            # Random over all images
            # batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            if args.use_dpt_ranking:
                if not args.no_batching:
                    raise NotImplementedError("DDPNeRF sampling requires --no_batching.")
            try:
                batch = next(raysRGB_iter).to(device)
            except StopIteration:
                raysRGB_iter = iter(DataLoader(RayDataset(rays_rgb), 
                                                    batch_size = N_rgb, 
                                                    shuffle=True, 
                                                    num_workers=0, 
                                                    generator=torch.Generator(device='cuda')))
                batch = next(raysRGB_iter).to(device)
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            if args.colmap_depth:
                # batch_depth = rays_depth[i_batch:i_batch+N_rand]
                try:
                    batch_depth = next(raysDepth_iter).to(device)
                except StopIteration:
                    raysDepth_iter = iter(DataLoader(RayDataset(rays_depth), 
                                                          batch_size = N_depth, 
                                                          shuffle=True, 
                                                          num_workers=0, 
                                                          generator=torch.Generator(device='cuda')))
                    batch_depth = next(raysDepth_iter).to(device)
                batch_depth = torch.transpose(batch_depth, 0, 1)
                batch_rays_depth = batch_depth[:2] # 2 x B x 3
                target_depth = batch_depth[2,:,0] # B
                ray_weights = batch_depth[3,:,0]

            # i_batch += N_rand
            # if i_batch >= rays_rgb.shape[0]:
            #     tqdm.write(f"Shuffle data after an epoch!")
            #     rand_idx = torch.randperm(rays_rgb.shape[0])
            #     rays_rgb = rays_rgb[rand_idx]
            #     i_batch = 0

        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            # target = torch.Tensor(target).to(device)# TODO

            if args.use_ddp and args.ddp_depth_loss_weight > 0.:
                target_depth = dense_depths[img_i]
                target_valid_depth = valid_depths[img_i]

            pose = poses[img_i, :3,:4]

            if N_rgb is not None:
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH),
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW),
                            indexing='ij'
                        ), -1)
                    if i == start:
                        tqdm.write(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")
                else:
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(0, H-1, H), 
                            torch.linspace(0, W-1, W), 
                            indexing='ij'
                        ), -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_rgb], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                # batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

                # This ensures select_coords is on the same device 
                # as target_depth and target_valid_depth before indexing.
                select_coords = select_coords
                if args.use_ddp and args.ddp_depth_loss_weight > 0.:
                    target_d = target_depth[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 1) or (N_rand, 2)
                    target_vd = target_valid_depth[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 1)
                    depth_range = precompute_depth_sampling(target_d)
                    batch_rays = torch.stack([rays_o, rays_d, depth_range], 0)  # (3, N_rand, 3)
                else:
                    batch_rays = torch.stack([rays_o, rays_d], 0)  # (2, N_rand, 3)

            if args.colmap_depth:
                # Get depth data for the selected image
                # depth_gts is loaded at the beginning of train() if args.colmap_depth is True
                depth_data = depth_gts[img_i]
                
                # Apply precrop filtering if needed (same as for RGB rays)
                depth_coords, depth_depths, depth_errors = filter_depth_by_precrop(
                    depth_data, H, W, i, args.precrop_iters, args.precrop_frac
                )
                
                num_depth_points = len(depth_coords)
                if num_depth_points > 0:
                    # Sample N_depth random indices from available depth points
                    select_depth_inds = np.random.choice(num_depth_points, size=[min(N_depth, num_depth_points)], replace=False)
                    
                    # Get rays for selected depth coordinates
                    rays_depth_o, rays_depth_d = get_rays_by_coord_np(H, W, focal, pose.cpu().numpy(), depth_coords[select_depth_inds])
                    batch_rays_depth = torch.stack([
                        torch.Tensor(rays_depth_o.copy()).to(device),
                        torch.Tensor(rays_depth_d.copy()).to(device)
                    ], 0)  # (2, N_depth, 3)
                    
                    target_depth = torch.Tensor(depth_depths[select_depth_inds]).to(device)  # (N_depth,)
                    ray_weights = torch.Tensor(depth_errors[select_depth_inds]).to(device) if depth_errors is not None else torch.ones_like(target_depth)  # (N_depth,)
                else:
                    raise NotImplementedError('Something went wront: num_depth_points <= 0')

        #####  Core optimization loop  #####

        if args.colmap_depth:
            N_batch = batch_rays.shape[1]
            batch_rays = torch.cat([batch_rays, batch_rays_depth], 1) # (2, 2 * N_rand, 3)


        rgb, disp, depth, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)

        if args.colmap_depth and not args.depth_with_rgb:
            # _, _, depth_col, _, extras_col = render(H, W, focal, chunk=args.chunk, rays=batch_rays_depth,
            #                                     verbose=i < 10, retraw=True, depths=target_depth,
            #                                     **render_kwargs_train)
            rgb = rgb[:N_batch, :]
            disp = disp[:N_batch]
            acc = acc[:N_batch]
            depth, depth_col = depth[:N_batch], depth[N_batch:]
            extras = {x:extras[x][:N_batch] for x in extras}
            extras_col = {x:extras[x][N_batch:] for x in extras}

        elif args.colmap_depth and args.depth_with_rgb:
            depth_col = depth

        # Compute SparseNeRF ranking and continuity losses using DPT depth (only in --no_batching mode)
        ranking_loss = torch.tensor(0.0, device=device)
        continuity_loss = torch.tensor(0.0, device=device)
        if args.use_dpt_ranking:
            if not args.no_batching:
                raise NotImplementedError("SparseNeRF ranking loss requires --no_batching.")
            else:
                # In non-batched mode, use the already sampled rays and their coordinates
                dpt_map = dpt_depths[img_i]  # [H, W]
                # `select_coords` contains [N_rand, 2] pixel coordinates (y, x)
                ranking_loss = local_depth_ranking_loss(depth, dpt_map.cpu().numpy(), coords=select_coords.cpu().numpy())
                continuity_loss = spatial_continuity_loss(depth, dpt_map.cpu().numpy(), coords=select_coords.cpu().numpy())


        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        
        # Compute DDPNeRF losses using DDP
        ddp_loss = 0.0
        if args.use_ddp and args.ddp_depth_loss_weight > 0.:
            ddp_loss = compute_ddp_depth_loss(depth, extras['z_vals'], extras['weights'], target_d, target_vd)
        
        # Depth loss calculation
        depth_loss_value = 0.0
        if args.depth_loss:
            # Validate dependencies for weighted loss
            if args.weighted_loss and not args.colmap_depth:
                raise ValueError("weighted_loss requires colmap_depth to be enabled")
            
            # Compute depth loss using the dedicated function
            depth_loss_value = depth_loss(
                rendered_depth=depth_col,
                target_depth=target_depth,
                weighted_loss=args.weighted_loss,
                relative_loss=args.relative_loss,
                normalize_depth=args.normalize_depth,
                ray_weights=ray_weights if args.weighted_loss else None,
                max_depth=max_depth if args.normalize_depth else None
            )

        sigma_loss = 0
        if args.sigma_loss:
            sigma_loss = extras_col['sigma_loss'].mean()
            # print(sigma_loss)
        # trans = extras['raw'][...,-1]
        if args.use_dpt_ranking:
            # SparseNeRF mode: only RGB + ranking + continuity losses
            loss = img_loss + args.lambda_rank * ranking_loss + args.lambda_cont * continuity_loss
        elif args.use_ddp:
            loss = img_loss + args.ddp_depth_loss_weight * ddp_loss
        else:
            # DSNeRF mode
            loss = img_loss + args.depth_lambda * depth_loss_value + args.sigma_lambda * sigma_loss
        psnr = mse2psnr(img_loss)
        

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        # sparsity_loss = args.sparse_loss_weight*(extras["sparsity_loss"].sum() + extras["sparsity_loss0"].sum())
        # loss = loss + sparsity_loss

        # add Total Variation loss
        if args.i_embed==1:
            n_levels = render_kwargs_train["embed_fn"].n_levels
            min_res = render_kwargs_train["embed_fn"].base_resolution
            max_res = render_kwargs_train["embed_fn"].finest_resolution
            log2_hashmap_size = render_kwargs_train["embed_fn"].log2_hashmap_size
            TV_loss = sum(total_variation_loss(render_kwargs_train["embed_fn"].embeddings[i], \
                                              min_res, max_res, \
                                              i, log2_hashmap_size, \
                                              n_levels=n_levels) for i in range(n_levels))
            loss = loss + args.tv_loss_weight * TV_loss
            if i>1000:
                args.tv_loss_weight = 0.0

        loss.backward()
        # pdb.set_trace()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        t = time.time()-time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            if args.i_embed==1:
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                    'embed_fn_state_dict': render_kwargs_train['embed_fn'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
            else:
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
            tqdm.write(f"[CHECKPOINT] Saved checkpoints at {path}")

        if i%args.i_video==0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, pred_depths, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, render_factor=args.render_factor)
            tqdm.write(f"[VIDEO] Done, saving {rgbs.shape, pred_depths.shape}")
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'depth.mp4', to8b(pred_depths / np.max(pred_depths)), fps=30, quality=8)

            # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     with torch.no_grad():
            #         rgbs_still, _, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        if i%args.i_testset==0 and i > 0 and len(i_test) > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            tqdm.write(f"[IMAGES] test poses shape {poses[i_test].shape}")
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir, render_factor=args.render_factor)
            tqdm.write(f"[IMAGES] Saved test set")



        if i%args.i_print==0:
            if args.use_dpt_ranking:
                tqdm.write(
                    f"[TRAIN] Iter: {i} "
                    f"Loss: {loss.item():.5f} "
                    f"PSNR: {psnr.item():.4f} "
                    f"Rank: {ranking_loss.item():.5f} "
                    f"Cont: {continuity_loss.item():.5f}"
                )
            else:
                tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item():.5f}  PSNR: {psnr.item():.4f}")
            loss_list.append(loss.item())
            psnr_list.append(psnr.item())
            time_list.append(t)
            loss_psnr_time = {
                "losses": loss_list,
                "psnr": psnr_list,
                "time": time_list
            }
            with open(os.path.join(basedir, expname, "loss_vs_time.pkl"), "wb") as fp:
                pickle.dump(loss_psnr_time, fp)

        global_step += 1


if __name__=='__main__':
    torch.set_default_device(device)

    train()
