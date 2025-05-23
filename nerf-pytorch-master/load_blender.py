import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2
import sys

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


def load_blender_data(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
            
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

        
    return imgs, poses, render_poses, [H, W, focal], i_split




# For saving test ground truth
# def load_blender_data(basedir, half_res=False, testskip=1):
#     splits = ['train', 'val', 'test']
#     metas = {}
#     for s in splits:
#         with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
#             metas[s] = json.load(fp)

#     all_imgs = []
#     all_poses = []
#     counts = [0]
#     for s in splits:
#         meta = metas[s]
#         imgs = []
#         poses = []
#         if s=='train' or testskip==0:
#             skip = 1
#         else:
#             skip = testskip
            
#         for frame in meta['frames'][::skip]:
#             fname = os.path.join(basedir, frame['file_path'] + '.png')
#             imgs.append(imageio.imread(fname))
#             poses.append(np.array(frame['transform_matrix']))
#         print()
#         imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
#         poses = np.array(poses).astype(np.float32)
#         counts.append(counts[-1] + imgs.shape[0])
#         all_imgs.append(imgs)
#         all_poses.append(poses)

#         # If half_res is True and the current split is 'test', downsample the test images
#         if half_res and s == 'test':
#             H, W = imgs[0].shape[:2]
#             H_half, W_half = H // 2, W // 2
#             imgs_half_res = np.zeros((imgs.shape[0], H_half, W_half, 4))
#             for i, img in enumerate(imgs):
#                 imgs_half_res[i] = cv2.resize(img, (W_half, H_half), interpolation=cv2.INTER_AREA)
#             imgs_half_res = imgs_half_res[...,:3]*imgs_half_res[...,-1:] + (1.-imgs_half_res[...,-1:])
#             save_dir='test_gt/drums'
#             os.makedirs(save_dir, exist_ok=True)
#             for i, img in enumerate(imgs_half_res):
#                 save_path = os.path.join(save_dir, f"{i:02d}.png")  # Save with format 00, 01, 02...
#                 imageio.imwrite(save_path, (img * 255).astype(np.uint8))
#             sys.exit()
    
#     i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
#     imgs = np.concatenate(all_imgs, 0)
#     poses = np.concatenate(all_poses, 0)
    
#     H, W = imgs[0].shape[:2]
#     camera_angle_x = float(meta['camera_angle_x'])
#     focal = .5 * W / np.tan(.5 * camera_angle_x)
    
#     render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    
#     if half_res:
#         H = H//2
#         W = W//2
#         focal = focal/2.

#         imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
#         for i, img in enumerate(imgs):
#             imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
#         imgs = imgs_half_res
#         # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

        
#     return imgs, poses, render_poses, [H, W, focal], i_split
