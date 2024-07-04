import os
import json
import copy
import itertools
import glob
import numpy as np
import cv2
import copy
from random import randint
from natsort import natsorted
from typing import Iterator, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, BatchSampler, SequentialSampler
from torchvision.utils import save_image

from helpers import setup_camera, l1_loss_v1, l1_loss_v2, weighted_l2_loss_v1, weighted_l2_loss_v2, quat_mult, \
    o3d_knn, params2rendervar, params2cpu, save_params
from loss_utils import fast_ssim
from external import calc_ssim, calc_psnr, build_rotation, densify, update_params_and_optimizer

from diff_gauss import GaussianRasterizer, GaussianRasterizationSettings
from gsplat import rasterization

from diff_gaussian_rasterization import SparseGaussianAdam
from torch.optim import Adam

from easyvolcap.utils.base_utils import *
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.data_utils import to_tensor, to_cuda
from easyvolcap.utils.sh_utils import SH2RGB, RGB2SH
from easyvolcap.utils.gaussian_utils import convert_to_gaussian_camera
from easyvolcap.utils.loss_utils import mIoU_loss


# def get_dataset(t, md, seq):
#     dataset = []
#     for c in range(len(md['fn'][t])):
#         w, h, k, w2c = md['w'], md['h'], md['k'][t][c], md['w2c'][t][c]
#         cam = setup_camera(w, h, k, w2c, near=1.0, far=100)
#         fn = md['fn'][t][c]
#         im = np.array(copy.deepcopy(Image.open(f"./data/{seq}/ims/{fn}")))
#         im = torch.tensor(im).float().cuda().permute(2, 0, 1) / 255
#         seg = np.array(copy.deepcopy(Image.open(f"./data/{seq}/seg/{fn.replace('.jpg', '.png')}"))).astype(np.float32)
#         seg = torch.tensor(seg).float().cuda()
#         seg_col = torch.stack((seg, torch.zeros_like(seg), 1 - seg))
#         dataset.append({'cam': cam, 'im': im, 'seg': seg_col, 'id': c})
#     return dataset


class MiniGS(nn.Module):
    def __init__(self,
                 _xyz, 
                 _featreus_dc,
                 _features_rest,
                 _scaling,
                 _rotation,
                 _opacity,
                 use_iso: bool = False,
                 ):
        super().__init__()
        self._xyz = nn.Parameter(_xyz)
        self._features_dc = nn.Parameter(_featreus_dc)
        self._features_rest = nn.Parameter(_features_rest)
        self._scaling = nn.Parameter(_scaling)
        self._opacity = nn.Parameter(_opacity)
        self.use_iso = use_iso
        if use_iso: 
            rotation = torch.zeros_like(_rotation)
            rotation[..., 0] = 1.0
            self._rotation = nn.Parameter(rotation, requires_grad=False)
        else:
            self._rotation = nn.Parameter(_rotation)
        self.setup_functions()
    
    @property
    def get_scaling(self):
        if self.use_iso: return self.scaling_activation(self._scaling).mean(dim=-1, keepdim=True).repeat(1, 3)
        else: return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=-1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def setup_functions(self):
        self.scaling_activation = torch.exp
        self.inverse_scaling_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = torch.logit
        self.rotation_activation = torch.nn.functional.normalize


class MiniDataset(Dataset):
    def __init__(self,
                 images: np.ndarray,
                 masks: np.ndarray,
                 cameras: dotdict,
                 camera_names: list,
                 near: float = 0.01,
                 far: float = 10.0,
                 H: int = None,
                 W: int = None,
                 ):
        super().__init__()
    
        self.images = images
        self.masks = masks
        self.cameras = cameras
        self.camera_names = camera_names
        self.near = near
        self.far = far
        self.H = H
        self.W = W

    def __len__(self):
        return len(self.camera_names)
    
    def __getitem__(self, index):
        if self.images is not None:
            image = torch.from_numpy(self.images[index]).float()
        else:
            image = None
        if self.masks is not None:
            mask = torch.from_numpy(self.masks[index]).float()[..., None]
        else: 
            mask = None

        meta = dotdict()
        if image is not None:
            meta.H = image.shape[0]
            meta.W = image.shape[1]
        else:
            meta.H = self.H
            meta.W = self.W

        camera = self.cameras[self.camera_names[index]]
        meta.K = torch.from_numpy(camera['K'].reshape(3, 3)).float()
        meta.R = torch.from_numpy(camera['R'].reshape(3, 3)).float()
        meta.T = torch.from_numpy(camera['T'].reshape(3, 1)).float()
        
        meta.n = self.near
        meta.f = self.far

        out = dotdict({
            'image': image if image is not None else torch.empty(0),
            'mask': mask if mask is not None else torch.empty(0),
            'camera_name': self.camera_names[index],
        })
        out.update(meta)
        out.meta = meta
        return out


def get_dataset(data_root: str, frame: int, cameras: dict, images_dir: str = 'images', masks_dir: str = 'masks', scale: float = 1.0, skip_images: bool = False):
    _cameras = dict()
    for k, v in cameras.items():
        _cameras[k] = {
            'K': v['K'].copy(),
            'R': v['R'].copy(),
            'T': v['T'].copy(),
        }
    cameras = _cameras

    if not skip_images:
        images = []
        masks = []
        camera_names = []
        for k in natsorted(cameras.keys()):
            images.append(os.path.join(data_root, images_dir, k, f'{frame:06d}.jpg'))
            masks.append(os.path.join(data_root, masks_dir, k, f'{frame:06d}.jpg'))
            camera_names.append(k)
        
        images = [cv2.imread(x, cv2.IMREAD_UNCHANGED) for x in tqdm(images, desc='Loading images')]
        images = [cv2.cvtColor(x, cv2.COLOR_BGR2RGB) for x in images]
        masks = [cv2.imread(x, cv2.IMREAD_UNCHANGED) for x in tqdm(masks, desc='Loading masks')]
        if scale != 1.0:
            image_height = int(images[0].shape[0] * scale)
            image_width = int(images[0].shape[1] * scale)
            h_scale = image_height / images[0].shape[0]
            w_scale = image_width / images[0].shape[1]
            images = [cv2.resize(x, (image_width, image_height), interpolation=cv2.INTER_AREA) for x in tqdm(images, desc='resizing images')]
            masks = [cv2.resize(x, (image_width, image_height), interpolation=cv2.INTER_AREA) for x in tqdm(masks, desc='resizing masks')]
            for v in cameras.values():
                v['K'][0] *= w_scale
                v['K'][1] *= h_scale       

        images = np.stack(images, axis=0) / 255.0
        masks = np.stack(masks, axis=0) / 255.0
        H = None
        W = None
    else:
        images = []
        masks = []
        camera_names = []
        for k in natsorted(cameras.keys()):
            images.append(os.path.join(data_root, images_dir, k, f'{frame:06d}.jpg'))
            masks.append(os.path.join(data_root, masks_dir, k, f'{frame:06d}.jpg'))
            camera_names.append(k)
        image = cv2.imread(images[0], cv2.IMREAD_UNCHANGED)
        H, W = image.shape[:2]
        images = None
        masks = None
    
    dataset = MiniDataset(images, masks, cameras, camera_names, H=H, W=W) 
    return dataset


def initialize_params(ply_path, sh_deg: int = 0, use_iso: bool = False):
    from plyfile import PlyData, PlyElement
    plydata = PlyData.read(ply_path)
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    xyz = torch.from_numpy(xyz).float()

    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
    opacities = torch.from_numpy(opacities).float()

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
    features_dc = torch.from_numpy(features_dc).float()

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    if len(extra_f_names) == 3 * (sh_deg + 1) ** 2 - 3:
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (sh_deg + 1) ** 2 - 1))
        features_rest = torch.from_numpy(features_extra).float()
    else:
        features_rest = torch.zeros((features_dc.shape[0], 3, (sh_deg + 1) ** 2 - 1), dtype=torch.float)

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
    scales = torch.from_numpy(scales).float()

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
    rots = torch.from_numpy(rots).float()

    gs = MiniGS(
        _xyz=xyz,
        _featreus_dc=features_dc,
        _features_rest=features_rest,
        _scaling=scales,
        _rotation=rots,
        _opacity=opacities,
        use_iso=use_iso,
    )

    return gs

def initialize_optimizer(named_params,
                         # Default parameters
                         lr: float = 5e-3,
                         eps: float = 1e-15,
                         weight_decay: float = 0.0,
                         
                         lr_table: dotdict = dotdict(),  # empty special learning rate table
                         eps_table: dotdict = dotdict(),  # empty table
                         weight_decay_table: dotdict = dotdict(),  # empty table):
                         ):
    if isinstance(named_params, Iterator):
        first = next(named_params)
        if isinstance(first, Tuple):
            named_params = itertools.chain([first], named_params)
        elif isinstance(first, nn.Parameter):
            log(yellow(f'Passed in a list of parameters, assuming they are named sequentially.'))
            named_params = {str(i): first for i, first in enumerate(named_params)}.items()
        else:
            raise NotImplementedError
    elif isinstance(named_params, Dict):
        named_params = named_params.items()
    else:
        raise NotImplementedError
    
    params = []
    for key, value in named_params:
        if not value.requires_grad:
            continue  # skip non-optimizable paramters
        v_lr = lr
        v_eps = eps
        v_weight_decay = weight_decay
        keys = key.split('.')
        for item in keys:
            if item in lr_table:
                v_lr = lr_table[item]
                break
        for item in keys:
            if item in eps_table:
                v_eps = eps_table[item]
                break
        for item in keys:
            if item in weight_decay_table:
                v_weight_decay = weight_decay_table[item]
                break
        params.append(
            dotdict(
                params=[value],
                lr=v_lr,
                eps=v_eps,
                weight_decay=v_weight_decay,
                name=key
            )
        )

    return Adam(params, lr=lr, eps=eps)


def write_params(path, _xyz, _features_dc, _features_rest, _opacity, _scaling, _rotation):
    from plyfile import PlyData, PlyElement

    def construct_list_of_attributes():
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(_features_dc.shape[1] * _features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(_features_rest.shape[1] * _features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(_scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(_rotation.shape[1]):
            l.append('rot_{}'.format(i))
        
        return l
    
    xyz = _xyz.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = _features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = _features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = _opacity.detach().cpu().numpy()
    scales = _scaling.detach().cpu().numpy()
    rotation = _rotation.detach().cpu().numpy()
    
    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes()]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scales, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)


def train(args):
    # if os.path.exists(f"./data/results/{exp}"):
    #     print(f"Experiment '{exp}' for sequence '{seq}' already exists. Exiting.")
    #     return
    # md = json.load(open(f"./data/{seq}/train_meta.json", 'r'))  # metadata
    # num_timesteps = len(md['fn'])

    # Load cameras
    from easyvolcap.utils.easy_utils import read_camera
    cameras = read_camera(args.data_root)
    camera_centers = []
    for v in cameras.values():
        R = v['R'].reshape(3, 3)
        T = v['T'].reshape(3,)
        c = -R.T @ T
        camera_centers.append(c)
    
    # camera_centers = np.stack(camera_centers, axis=0)
    # scene_radius = 1.1 * np.max(np.linalg.norm(camera_centers - np.mean(camera_centers, 0)[None], axis=-1))
    # scene_radius = args.scene_scale
    # log(blue(f'scene_radius: {scene_radius}'))

    # init opt
    lr_table = dotdict(
        _xyz=0.00016,
        _features_dc=0.0025,
        _features_rest=0.0025 / 20,
        _scaling=0.001,
        _rotation=0.001,
        _opacity=0.05,
    )

    # Load initial gaussian
    head_gs_list = natsorted(glob.glob(os.path.join(args.init_gs_path, 'head-*.ply')))
    body_gs_list = natsorted(glob.glob(os.path.join(args.init_gs_path, 'body-*.ply')))
    lhand_gs_list = natsorted(glob.glob(os.path.join(args.init_gs_path, 'lhand-*.ply')))
    rhand_gs_list = natsorted(glob.glob(os.path.join(args.init_gs_path, 'rhand-*.ply')))


    frame_start = max(args.frame_start, 0)
    name = os.path.basename(head_gs_list[-1]).split('.')[0]
    frame = int(name.split('-')[-1])
    frame_end = frame + 1 if args.frame_end is None else args.frame_end

    for idx in range(len(head_gs_list)):
        name = os.path.basename(head_gs_list[idx]).split('.')[0]
        frame = int(name.split('-')[-1])
        if frame < frame_start: continue
        if frame >= frame_end: break

        head_gs = initialize_params(head_gs_list[idx], sh_deg=args.sh_deg, use_iso=args.use_head_iso)
        body_gs = initialize_params(body_gs_list[idx], sh_deg=args.sh_deg, use_iso=args.use_body_iso)
        lhand_gs = initialize_params(lhand_gs_list[idx], sh_deg=args.sh_deg, use_iso=args.use_hand_iso)
        rhand_gs = initialize_params(rhand_gs_list[idx], sh_deg=args.sh_deg, use_iso=args.use_hand_iso)

        n_head = head_gs._xyz.shape[0]
        n_body = body_gs._xyz.shape[0]
        n_lhand = lhand_gs._xyz.shape[0]
        n_rhand = rhand_gs._xyz.shape[0]
        head_range = np.array([0, n_head])
        body_range = np.array([n_head, n_head + n_body])
        lhand_range = np.array([n_head + n_body, n_head + n_body + n_lhand])
        rhand_range = np.array([n_head + n_body + n_lhand, n_head + n_body + n_lhand + n_rhand])
        if args.write_mode == 'full':
            save_path = os.path.join(args.save_dir, f"range.npz")
            if not os.path.exists(save_path):
                np.savez(save_path, head=head_range, body=body_range, lhand=lhand_range, rhand=rhand_range)

        head_optimizer = initialize_optimizer(named_params=((k, v) for k, v in head_gs.named_parameters() if v.requires_grad), lr=0.0, lr_table=lr_table)
        body_optimizer = initialize_optimizer(named_params=((k, v) for k, v in body_gs.named_parameters() if v.requires_grad), lr=0.0, lr_table=lr_table)
        lhand_optimizer = initialize_optimizer(named_params=((k, v) for k, v in lhand_gs.named_parameters() if v.requires_grad), lr=0.0, lr_table=lr_table)
        rhand_optimizer = initialize_optimizer(named_params=((k, v) for k, v in rhand_gs.named_parameters() if v.requires_grad), lr=0.0, lr_table=lr_table)

        head_gs = head_gs.cuda()
        body_gs = body_gs.cuda()
        lhand_gs = lhand_gs.cuda()
        rhand_gs = rhand_gs.cuda()

        dataset = get_dataset(args.data_root, frame, cameras, scale=args.scale)
        sampler = RandomSampler(dataset, replacement=True, num_samples=args.iters)
        batch_sampler = BatchSampler(sampler, batch_size=1, drop_last=False)
        dataloader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=4)
        progress_bar = tqdm(range(args.iters), desc=f"training frame {frame}")
        for batch_id, batch in enumerate(dataloader):
            camera_name = batch.pop('camera_name')
            meta = batch.pop('meta')
            batch = to_cuda(to_tensor(batch), device='cuda')
            loss = 0

            for i in range(len(meta.H)):
                # get data
                camera = convert_to_gaussian_camera(
                    K=batch.K[i], R=batch.R[i], T=batch.T[i],
                    H=batch.H[i], W=batch.W[i], n=batch.n[i], f=batch.f[i],
                    cpu_K=meta.K[i], cpu_R=meta.R[i], cpu_T=meta.T[i],
                    cpu_H=meta.H[i], cpu_W=meta.W[i], cpu_n=meta.n[i], cpu_f=meta.f[i],
                )
                im = batch.image[i]
                mask = batch.mask[i]

                # get gs
                means = torch.cat([head_gs.get_xyz, body_gs.get_xyz, lhand_gs.get_xyz, rhand_gs.get_xyz], dim=0)
                quats = torch.cat([head_gs.get_rotation, body_gs.get_rotation, lhand_gs.get_rotation, rhand_gs.get_rotation], dim=0)
                scales = torch.cat([head_gs.get_scaling, body_gs.get_scaling, lhand_gs.get_scaling, rhand_gs.get_scaling], dim=0)
                opacities = torch.cat([head_gs.get_opacity, body_gs.get_opacity, lhand_gs.get_opacity, rhand_gs.get_opacity], dim=0)
                features = torch.cat([head_gs.get_features, body_gs.get_features, lhand_gs.get_features, rhand_gs.get_features], dim=0)
                # colors = SH2RGB(features).reshape(-1, 3)

                # render
                viewmats = torch.eye(4, device='cuda')
                viewmats[:3, :3] = camera['R']
                viewmats[:3, 3] = camera['T'].squeeze()
                rgb, acc, info = rasterization(
                    means=means,
                    quats=quats,
                    scales=scales,
                    opacities=opacities[..., 0],
                    colors=features.mT,
                    viewmats=viewmats[None],
                    Ks=camera['K'][None],
                    width=meta.W[i],
                    height=meta.H[i],
                    rasterize_mode='classic',
                    sh_degree=args.sh_deg,
                )
                radii = info['radii']
                rgb = rgb[0]
                acc = acc[0]

                # raster_settings = GaussianRasterizationSettings(
                #     image_height=camera.image_height,
                #     image_width=camera.image_width,
                #     tanfovx=camera.tanfovx,
                #     tanfovy=camera.tanfovy,
                #     bg=torch.full([3], 0.0, device='cuda'),  # GPU
                #     scale_modifier=1.0,
                #     viewmatrix=camera.world_view_transform,
                #     projmatrix=camera.full_proj_transform,
                #     sh_degree=args.sh_deg,
                #     campos=camera.camera_center,
                #     prefiltered=False,
                #     debug=False,
                # )
                # raster = GaussianRasterizer(raster_settings)
                # rgb, acc, dpt, radii = raster(
                #     means3D=means,
                #     means2D=torch.zeros_like(means, device='cuda', requires_grad=False),
                #     rotations=quats,
                #     opacities=opacities,
                #     scales=scales,
                #     shs=features,
                #     # colors_precomp=colors,
                # )
                # save_image(rgb, f'tmp/pred-{batch_id:06d}.jpg')
                # save_image(im.permute(2, 0, 1), f'tmp/gt-{batch_id:06d}.jpg')
                # rgb = rgb.permute(2, 0, 1)
                # acc = acc.permute(2, 0, 1)
                
                bg = torch.rand_like(rgb, requires_grad=False)
                pred = rgb * acc + (1 - acc) * bg
                pred = pred.permute(2, 0, 1)
                gt = im * mask + (1 - mask) * bg
                gt = gt.permute(2, 0, 1)
                loss += 0.8 * l1_loss_v1(pred, gt) + 0.2 * (1.0 - fast_ssim(pred, gt)) + 2.0 * scales.mean()
            
            loss.backward()
            
            if isinstance(head_optimizer, SparseGaussianAdam):
                head_radii = radii[head_range[0]:head_range[1]]
                body_radii = radii[body_range[0]:body_range[1]]
                lhand_radii = radii[lhand_range[0]:lhand_range[1]]
                rhand_radii = radii[rhand_range[0]:rhand_range[1]]
                visible = head_radii > 0
                head_optimizer.step(visible, head_radii.shape[0])
                visible = body_radii > 0
                body_optimizer.step(visible, body_radii.shape[0])
                visible = lhand_radii > 0
                lhand_optimizer.step(visible, lhand_radii.shape[0])
                visible = rhand_radii > 0
                rhand_optimizer.step(visible, rhand_radii.shape[0])
            else:
                head_optimizer.step()
                body_optimizer.step()
                lhand_optimizer.step()
                rhand_optimizer.step()

            head_optimizer.zero_grad(set_to_none=True)
            body_optimizer.zero_grad(set_to_none=True)
            lhand_optimizer.zero_grad(set_to_none=True)
            rhand_optimizer.zero_grad(set_to_none=True)
            
            with torch.no_grad():
                if (batch_id + 1) % args.log_interval == 0:
                    gt = (im * mask).permute(2, 0, 1).contiguous()
                    pred = (rgb * acc).permute(2, 0, 1).contiguous()
                    psnr = calc_psnr(pred, gt).mean()
                    if torch.isnan(psnr):
                        breakpoint()
                    log(blue(f'training psnr of iter {batch_id}: {psnr:.{4}f}'))
                    progress_bar.update(args.log_interval)

        progress_bar.close()
        # paramscpu = params2cpu(params, is_initial_timestep)
        if args.write_mode == 'full':
            save_path = os.path.join(args.save_dir, f"full-{frame:06d}.ply")
            _xyz = torch.cat([head_gs._xyz, body_gs._xyz, lhand_gs._xyz, rhand_gs._xyz], dim=0)
            _features_dc = torch.cat([head_gs._features_dc, body_gs._features_dc, lhand_gs._features_dc, rhand_gs._features_dc], dim=0)
            _features_rest = torch.cat([head_gs._features_rest, body_gs._features_rest, lhand_gs._features_rest, rhand_gs._features_rest], dim=0)
            _scaling = torch.cat([head_gs._scaling, body_gs._scaling, lhand_gs._scaling, rhand_gs._scaling], dim=0)
            _rotation = torch.cat([head_gs._rotation, body_gs._rotation, lhand_gs._rotation, rhand_gs._rotation], dim=0)
            _opacity = torch.cat([head_gs._opacity, body_gs._opacity, lhand_gs._opacity, rhand_gs._opacity], dim=0)
            write_params(save_path, _xyz, _features_dc, _features_rest, _opacity, _scaling, _rotation)
        elif args.write_mode == 'sep':
            save_path = os.path.join(args.save_dir, f"head-{frame:06d}.ply")
            write_params(save_path, head_gs._xyz, head_gs._features_dc, head_gs._features_rest, head_gs._opacity, head_gs._scaling, head_gs._rotation)
            save_path = os.path.join(args.save_dir, f"body-{frame:06d}.ply")
            write_params(save_path, body_gs._xyz, body_gs._features_dc, body_gs._features_rest, body_gs._opacity, body_gs._scaling, body_gs._rotation)
            save_path = os.path.join(args.save_dir, f"lhand-{frame:06d}.ply")
            if args.use_hand_iso:
                _scaling = lhand_gs.inverse_scaling_activation(lhand_gs.get_scaling)
                write_params(save_path, lhand_gs._xyz, lhand_gs._features_dc, lhand_gs._features_rest, lhand_gs._opacity, _scaling, lhand_gs._rotation)
            else:
                write_params(save_path, lhand_gs._xyz, lhand_gs._features_dc, lhand_gs._features_rest, lhand_gs._opacity, lhand_gs._scaling, lhand_gs._rotation)
            save_path = os.path.join(args.save_dir, f"rhand-{frame:06d}.ply")
            if args.use_hand_iso:
                _scaling = rhand_gs.inverse_scaling_activation(rhand_gs.get_scaling)
                write_params(save_path, rhand_gs._xyz, rhand_gs._features_dc, rhand_gs._features_rest, rhand_gs._opacity, _scaling, rhand_gs._rotation)
            else:
                write_params(save_path, rhand_gs._xyz, rhand_gs._features_dc, rhand_gs._features_rest, rhand_gs._opacity, rhand_gs._scaling, rhand_gs._rotation)


def test(args, part='hand'):
    out_dir = os.path.join(args.save_dir + '_render', f'{part}')

    from easyvolcap.utils.easy_utils import read_camera
    cameras = read_camera(args.data_root)
    camera_centers = []
    for v in cameras.values():
        R = v['R'].reshape(3, 3)
        T = v['T'].reshape(3,)
        c = -R.T @ T
        camera_centers.append(c)
    for k in cameras.keys():
        os.makedirs(os.path.join(out_dir, 'images', k), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'masks', k), exist_ok=True)

    ply_list = []
    if part == 'hand':
        lhand_list = glob.glob(os.path.join(args.save_dir, 'lhand*.ply'))
        ply_list.append(natsorted(lhand_list))
        rhand_list = glob.glob(os.path.join(args.save_dir, 'rhand*.ply'))
        ply_list.append(natsorted(rhand_list))

    for idx in tqdm(range(len(ply_list[0]))):
        name = os.path.basename(ply_list[0][idx]).split('.')[0]
        frame = int(name.split('-')[-1])
        gs_list = []
        for i in range(len(ply_list)):
            _name = os.path.basename(ply_list[i][idx]).split('.')[0]
            _frame = int(_name.split('-')[-1])
            assert frame == _frame
            gs = initialize_params(ply_list[i][idx], sh_deg=args.sh_deg, use_iso=args.use_hand_iso)
            gs = gs.cuda()
            for p in gs.parameters():
                p.requires_grad = False
            gs_list.append(gs)
        
        dataset = get_dataset(args.data_root, frame, cameras, scale=args.scale, skip_images=True)
        sampler = SequentialSampler(dataset)
        batch_sampler = BatchSampler(sampler, batch_size=1, drop_last=False)
        dataloader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=4)
        for batch_id, batch in enumerate(dataloader):
            camera_name = batch.pop('camera_name')
            meta = batch.pop('meta')
            batch = to_cuda(to_tensor(batch), device='cuda')
            for i in range(len(meta.H)):
                # get data
                camera = convert_to_gaussian_camera(
                    K=batch.K[i], R=batch.R[i], T=batch.T[i],
                    H=batch.H[i], W=batch.W[i], n=batch.n[i], f=batch.f[i],
                    cpu_K=meta.K[i], cpu_R=meta.R[i], cpu_T=meta.T[i],
                    cpu_H=meta.H[i], cpu_W=meta.W[i], cpu_n=meta.n[i], cpu_f=meta.f[i],
                )
                # im = batch.image[i]
                # mask = batch.mask[i]

                # get gs
                means = torch.cat([gs.get_xyz for gs in gs_list], dim=0)
                quats = torch.cat([gs.get_rotation for gs in gs_list], dim=0)
                scales = torch.cat([gs.get_scaling for gs in gs_list], dim=0)
                opacities = torch.cat([gs.get_opacity for gs in gs_list], dim=0)
                features = torch.cat([gs.get_features for gs in gs_list], dim=0)

                # render
                viewmats = torch.eye(4, device='cuda')
                viewmats[:3, :3] = camera['R']
                viewmats[:3, 3] = camera['T'].squeeze()
                rgb, acc, info = rasterization(
                    means=means,
                    quats=quats,
                    scales=scales,
                    opacities=opacities[..., 0],
                    colors=features.mT,
                    viewmats=viewmats[None],
                    Ks=camera['K'][None],
                    width=meta.W[i],
                    height=meta.H[i],
                    rasterize_mode='classic',
                    sh_degree=args.sh_deg,
                )
                rgb = rgb[0].permute(2, 0, 1)
                acc = acc[0].permute(2, 0, 1)

                save_path = os.path.join(out_dir, f"images/{camera_name[i]}/{frame:06d}.jpg")
                save_image(rgb, save_path)
                save_path = os.path.join(out_dir, f"masks/{camera_name[i]}/{frame:06d}.jpg")
                save_image(acc, save_path)

        del gs_list
        torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--init_gs_path", type=str)
    parser.add_argument("--mode", type=str, choices=['train', 'test-hand', 'test-body', 'test-head', 'test-all'], default='train')
    parser.add_argument("--save_dir", type=str, default='./data/results')
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--sh_deg", type=int, default=1)
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--frame_start", type=int, default=0)
    parser.add_argument("--frame_end", type=int, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--write_mode", type=str, choices=['full', 'sep'], default='sep')
    parser.add_argument("--use_head_iso", action='store_true', default=False)
    parser.add_argument("--use_body_iso", action='store_true', default=False)
    parser.add_argument("--use_hand_iso", action='store_true', default=False)
    args = parser.parse_args()

    if args.exp_name == None:
        seq = os.path.dirname(args.data_root).split('/')[-1]
        args.save_dir = os.path.join(args.save_dir, seq)
    else:
        args.save_dir = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test-hand':
        test(args, 'hand')
    torch.cuda.empty_cache()
