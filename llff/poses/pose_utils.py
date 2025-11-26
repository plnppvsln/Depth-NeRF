import numpy as np
import os
import sys
import imageio
# import skimage.transform

from load_llff import _minify as llff_minify

from llff.poses.colmap_wrapper import run_colmap
import llff.poses.colmap_read_model as read_model

def save_views(realdir,names):
    with open(os.path.join(realdir,'view_imgs.txt'), mode='w') as f:
        f.writelines('\n'.join(names))
    f.close()


def remove_unregistered_images(basedir):
    """
    Remove images from the images/ directory that are not in view_imgs.txt.
    This is useful when COLMAP couldn't register all images.
    
    Args:
        basedir: Base directory containing images/ and view_imgs.txt
    """
    view_imgs_file = os.path.join(basedir, 'view_imgs.txt')
    images_dir = os.path.join(basedir, 'images')
    
    if not os.path.exists(view_imgs_file):
        print(f'view_imgs.txt not found in {basedir}, skipping image cleanup')
        return
    
    if not os.path.exists(images_dir):
        print(f'images/ directory not found in {basedir}, skipping image cleanup')
        return
    
    # Read registered image names from view_imgs.txt
    with open(view_imgs_file, 'r') as f:
        registered_names = set(line.strip() for line in f.readlines() if line.strip())
    
    if not registered_names:
        print('No registered images found in view_imgs.txt, skipping image cleanup')
        return
    
    # Get all image files in the images directory
    image_extensions = ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG']
    all_images = []
    for f in os.listdir(images_dir):
        if any(f.endswith(ext) for ext in image_extensions):
            all_images.append(f)
    
    # Find images to delete (not in registered_names)
    images_to_delete = []
    for img in all_images:
        if img not in registered_names:
            images_to_delete.append(img)
    
    # Delete unregistered images
    deleted_count = 0
    for img in images_to_delete:
        img_path = os.path.join(images_dir, img)
        try:
            os.remove(img_path)
            deleted_count += 1
        except Exception as e:
            print(f'Warning: Could not delete {img_path}: {e}')
    
    if deleted_count > 0:
        print(f'Removed {deleted_count} unregistered image(s) from {images_dir}')
    else:
        print(f'All {len(all_images)} images are registered, no cleanup needed')

def load_colmap_data(realdir):
    
    camerasfile = os.path.join(realdir, 'sparse/0/cameras.bin')
    camdata = read_model.read_cameras_binary(camerasfile)
    
    list_of_keys = list(camdata.keys())
    cam = camdata[list_of_keys[0]]
    print( 'Cameras', cam)

    h, w, f = cam.height, cam.width, cam.params[0]
    hwf = np.array([h,w,f]).reshape([3,1])
    
    imagesfile = os.path.join(realdir, 'sparse/0/images.bin')
    imdata = read_model.read_images_binary(imagesfile)
    
    real_ids = [k for k in imdata]
    
    w2c_mats = []
    bottom = np.array([0,0,0,1.]).reshape([1,4])
    
    names = [imdata[k].name for k in imdata]
    print( 'Images #', len(names))
    perm = np.argsort(names)
    sort_names = [names[i] for i in perm]
    save_views(realdir, sort_names)
    
    for k in imdata:
        im = imdata[k]
        R = im.qvec2rotmat()
        t = im.tvec.reshape([3,1])
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        w2c_mats.append(m)
    
    w2c_mats = np.stack(w2c_mats, 0)
    c2w_mats = np.linalg.inv(w2c_mats)
    
    poses = c2w_mats[:, :3, :4].transpose([1,2,0])
    poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1,1,poses.shape[-1]])], 1)
    
    points3dfile = os.path.join(realdir, 'sparse/0/points3D.bin')
    pts3d = read_model.read_points3d_binary(points3dfile)
    
    # must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t]
    poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]], 1)
    
    return poses, pts3d, perm, real_ids


def save_poses(basedir, poses, pts3d, perm, real_ids):
    pts_arr = []
    vis_arr = []
    for k in pts3d:
        pts_arr.append(pts3d[k].xyz)
        cams = [0] * poses.shape[-1]
        for ind in pts3d[k].image_ids:
            if ind not in real_ids:
                continue
            idx = real_ids.index(ind)
            if idx >= len(cams):
                print('ERROR: the correct camera poses for current points cannot be accessed')
                return
            cams[idx] = 1
        vis_arr.append(cams)

    pts_arr = np.array(pts_arr)
    vis_arr = np.array(vis_arr)
    print( 'Points', pts_arr.shape, 'Visibility', vis_arr.shape )
    
    zvals = np.sum(-(pts_arr[:, np.newaxis, :].transpose([2,0,1]) - poses[:3, 3:4, :]) * poses[:3, 2:3, :], 0)
    valid_z = zvals[vis_arr==1]
    print( 'Depth stats', valid_z.min(), valid_z.max(), valid_z.mean() )
    
    save_arr = []
    for i in perm:
        vis = vis_arr[:, i]
        zs = zvals[:, i]
        zs = zs[vis==1]
        close_depth, inf_depth = np.percentile(zs, .5), np.percentile(zs, 99.5)
        
        save_arr.append(np.concatenate([poses[..., i].ravel(), np.array([close_depth, inf_depth])], 0))
    save_arr = np.array(save_arr)
    
    np.save(os.path.join(basedir, 'poses_bounds.npy'), save_arr)


def minify(basedir, factors=[], resolutions=[]):
    llff_minify(basedir, factors=factors, resolutions=resolutions)
        
        
def load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
    bds = poses_arr[:, -2:].transpose([1,0])
    
    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape
    
    sfx = ''
    
    if factor is not None:
        sfx = '_{}'.format(factor)
        minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1
    
    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return
    
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]) )
        return
    
    sh = imageio.imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1./factor
    
    if not load_imgs:
        return poses, bds
    
    # imgs = [imageio.imread(f, ignoregamma=True)[...,:3]/255. for f in imgfiles]
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)
        
    imgs = [imread(f)[...,:3]/255. for f in imgfiles]
    imgs = np.stack(imgs, -1)  
    
    print('Loaded image data', imgs.shape, poses[:,-1,0])
    return poses, bds, imgs


def gen_poses(basedir, match_type, factors=None, remove_unregistered=False):
    """
    Generate camera poses from images using COLMAP.
    
    Args:
        basedir: Base directory containing images/ subdirectory
        match_type: COLMAP matcher type ('exhaustive_matcher' or 'sequential_matcher')
        factors: Optional list of downsampling factors for minification
        remove_unregistered: If True, remove images that weren't registered by COLMAP (default: True)
    """
    files_needed = ['{}.bin'.format(f) for f in ['cameras', 'images', 'points3D']]
    if os.path.exists(os.path.join(basedir, 'sparse/0')):
        files_had = os.listdir(os.path.join(basedir, 'sparse/0'))
    else:
        files_had = []
    if not all([f in files_had for f in files_needed]):
        print( 'Need to run COLMAP' )
        run_colmap(basedir, match_type)
    else:
        print('Don\'t need to run COLMAP')
        
    print( 'Post-colmap')

    poses, pts3d, perm, real_ids = load_colmap_data(basedir)
    
    save_poses(basedir, poses, pts3d, perm, real_ids)
    
    # Remove images that weren't registered by COLMAP (if requested)
    if remove_unregistered:
        remove_unregistered_images(basedir)
    
    if factors is not None:
        print( 'Factors:', factors)
        minify(basedir, factors)
    
    print( 'Done with imgs2poses' )
    
    return True
    
