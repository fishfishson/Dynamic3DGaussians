import os
import json
import copy
import numpy as np
from PIL import Image
import cv2
import copy
from random import randint
# from tqdm import tqdm
from natsort import natsorted

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, BatchSampler
from torchvision.utils import save_image

from helpers import setup_camera, l1_loss_v1, l1_loss_v2, weighted_l2_loss_v1, weighted_l2_loss_v2, quat_mult, \
    o3d_knn, params2rendervar, params2cpu, save_params
from loss_utils import fast_ssim
from external import calc_ssim, calc_psnr, build_rotation, densify, update_params_and_optimizer
from diff_gauss import GaussianRasterizer, GaussianRasterizationSettings
# from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings
from diff_gaussian_rasterization import SparseGaussianAdam

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


class MiniDataset(Dataset):
    def __init__(self,
                 images: np.ndarray,
                 masks: np.ndarray,
                 cameras: dotdict,
                 camera_names: list,
                 near: float = 0.01,
                 far: float = 10.0,
                 ):
        super().__init__()
    
        self.images = images
        self.masks = masks
        self.cameras = cameras
        self.camera_names = camera_names
        self.near = near
        self.far = far

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = torch.from_numpy(self.images[index]).float().permute(2, 0, 1)
        mask = torch.from_numpy(self.masks[index]).float()[None]
        
        meta = dotdict()
        meta.H = image.shape[1]
        meta.W = image.shape[2]
        
        camera = self.cameras[self.camera_names[index]]
        meta.K = torch.from_numpy(camera['K'].reshape(3, 3)).float()
        meta.R = torch.from_numpy(camera['R'].reshape(3, 3)).float()
        meta.T = torch.from_numpy(camera['T'].reshape(3, 1)).float()
        
        meta.n = self.near
        meta.f = self.far

        out = dotdict({
            'image': image,
            'mask': mask,
            'camera_name': self.camera_names[index],
        })
        out.update(meta)
        out.meta = meta
        return out


def get_dataset(data_root: str, frame: int, cameras: dict, images_dir: str = 'images', masks_dir: str = 'masks', scale: float = 1.0):
    _cameras = dict()
    for k, v in cameras.items():
        _cameras[k] = {
            'K': v['K'].copy(),
            'R': v['R'].copy(),
            'T': v['T'].copy(),
        }
    cameras = _cameras

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
    dataset = MiniDataset(images, masks, cameras, camera_names) 
    return dataset

def get_batch(todo_dataset, dataset):
    if not todo_dataset:
        todo_dataset = dataset.copy()
    curr_data = todo_dataset.pop(randint(0, len(todo_dataset) - 1))
    return curr_data


# def initialize_params(seq, md):
#     init_pt_cld = np.load(f"./data/{seq}/init_pt_cld.npz")["data"]
#     seg = init_pt_cld[:, 6]
#     max_cams = 50
#     sq_dist, _ = o3d_knn(init_pt_cld[:, :3], 3)
#     mean3_sq_dist = sq_dist.mean(-1).clip(min=0.0000001)
#     params = {
#         'means3D': init_pt_cld[:, :3],
#         'rgb_colors': init_pt_cld[:, 3:6],
#         'seg_colors': np.stack((seg, np.zeros_like(seg), 1 - seg), -1),
#         'unnorm_rotations': np.tile([1, 0, 0, 0], (seg.shape[0], 1)),
#         'logit_opacities': np.zeros((seg.shape[0], 1)),
#         'log_scales': np.tile(np.log(np.sqrt(mean3_sq_dist))[..., None], (1, 3)),
#         'cam_m': np.zeros((max_cams, 3)),
#         'cam_c': np.zeros((max_cams, 3)),
#     }
#     params = {k: torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True)) for k, v in
#               params.items()}
#     cam_centers = np.linalg.inv(md['w2c'][0])[:, :3, 3]  # Get scene radius
#     scene_radius = 1.1 * np.max(np.linalg.norm(cam_centers - np.mean(cam_centers, 0)[None], axis=-1))
#     variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
#                  'scene_radius': scene_radius,
#                  'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(),
#                  'denom': torch.zeros(params['means3D'].shape[0]).cuda().float()}
#     return params, variables


def initialize_params(ply_path, fix_occ=True, fix_scaling=True):
    from plyfile import PlyData, PlyElement
    plydata = PlyData.read(ply_path)
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    xyz = torch.from_numpy(xyz).float()

    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
    opacities = torch.from_numpy(opacities).float()

    features_dc = np.zeros((xyz.shape[0], 3))
    features_dc[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])
    features_dc = torch.from_numpy(features_dc).float()
    rgb = SH2RGB(features_dc).clip(0, 1)

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

    params = {
        'means3D': xyz,
        'rgb_colors': rgb,
        'unnorm_rotations': rots,
        'logit_opacities': opacities,
        'log_scales': scales,
    }
    params = {k: torch.nn.Parameter(v.cuda().contiguous().requires_grad_(True)) for k, v in params.items()}
    if fix_occ: params['logit_opacities'].requires_grad = False
    if fix_scaling: params['log_scales'].requires_grad = False
    params['unnorm_rotations'].requires_grad = False # NOTE: we use iso setting!

    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'denom': torch.zeros(params['means3D'].shape[0]).cuda().float()}
    return params, variables


def write_params(path, params):
    from plyfile import PlyData, PlyElement
    _xyz = params['means3D']
    _features_dc = torch.zeros((_xyz.shape[0], 1, 3), dtype=_xyz.dtype, device=_xyz.device)
    _features_dc[:, 0, :] = RGB2SH(params['rgb_colors'])
    _features_rest = torch.zeros((_xyz.shape[0], 0, 3), dtype=_xyz.dtype, device=_xyz.device)
    _opacity = params['logit_opacities']
    _scaling = params['log_scales']
    _rotation = params['unnorm_rotations']

    def construct_list_of_attributes():
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(_features_dc[:, 0, :].shape[1]):
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
    f_dc = _features_dc[:, 0, :].detach().cpu().numpy()
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


def initialize_optimizer(params, scene_radius=1.0):
    lrs = {
        'means3D': 0.00016 * scene_radius,
        'rgb_colors': 0.0025,
        'seg_colors': 0.0,
        'unnorm_rotations': 0.001,
        'logit_opacities': 0.05,
        'log_scales': 0.001,
        'cam_m': 1e-4,
        'cam_c': 1e-4,
    }
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items() if v.requires_grad == True]
    # return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)
    return SparseGaussianAdam(param_groups, lr=0.0, eps=1e-15)


def get_loss(params, curr_data, variables, raster_settings, is_initial_timestep):
    losses = {}

    rendervar = params2rendervar(params)
    # rendervar['means2D'].retain_grad()
    im, dpt, acc, radii = GaussianRasterizer(raster_settings=raster_settings)(**rendervar)
    # im, _ = GaussianRasterizer(raster_settings=raster_settings)(**rendervar)
    # dpt = torch.zeros_like(im[:1])
    # acc = torch.ones_like(im[:1])
    out = dotdict(img=im, dpt=dpt, acc=acc, radii=radii)
    
    # Assume all images has the same exposure and white balance
    # curr_id = curr_data['id']
    # im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]
    # add random noise
    bg = torch.rand_like(im, requires_grad=False)
    _im = im * acc + (1 - acc) * bg
    # _im = im * acc
    _gt = curr_data['im'] * curr_data['mask'] + (1 - curr_data['mask']) * bg
    # _gt = curr_data['im'] * curr_data['mask']

    # save_image(_im, 'im.jpg')
    # save_image(_gt, 'gt.jpg')
    # exit()

    # losses['im'] = 0.8 * l1_loss_v1(_im, _gt) + 0.2 * (1.0 - calc_ssim(_im, _gt))
    losses['im'] = 0.8 * l1_loss_v1(_im, _gt) + 0.2 * (1.0 - fast_ssim(_im, _gt))
    # variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification

    # losses['seg'] = mIoU_loss(acc.flatten(-2, -1).permute(1, 0)[None], curr_data['mask'].flatten(-2, -1).permute(1, 0)[None])
    # segrendervar = params2rendervar(params)
    # segrendervar['colors_precomp'] = params['seg_colors']
    # seg, _, _, = Renderer(raster_settings=curr_data['cam'])(**segrendervar)
    # losses['seg'] = 0.8 * l1_loss_v1(seg, curr_data['seg']) + 0.2 * (1.0 - calc_ssim(seg, curr_data['seg']))

    if not is_initial_timestep:
        # is_fg = (params['seg_colors'][:, 0] > 0.5).detach()
        fg_pts = rendervar['means3D']
        # fg_rot = rendervar['rotations']

        # rel_rot = quat_mult(fg_rot, variables["prev_inv_rot_fg"])
        # rot = build_rotation(rel_rot)
        neighbor_pts = fg_pts[variables["neighbor_indices"]]
        curr_offset = neighbor_pts - fg_pts[:, None]
        # curr_offset_in_prev_coord = (rot.transpose(2, 1)[:, None] @ curr_offset[:, :, :, None]).squeeze(-1)
        curr_offset_in_prev_coord = curr_offset
        losses['rigid'] = weighted_l2_loss_v2(curr_offset_in_prev_coord, variables["prev_offset"],
                                              variables["neighbor_weight"])

        # losses['rot'] = weighted_l2_loss_v2(rel_rot[variables["neighbor_indices"]], rel_rot[:, None],
        #                                     variables["neighbor_weight"])

        curr_offset_mag = torch.sqrt((curr_offset ** 2).sum(-1) + 1e-20)
        losses['iso'] = weighted_l2_loss_v1(curr_offset_mag, variables["neighbor_dist"], variables["neighbor_weight"])

        # losses['floor'] = torch.clamp(fg_pts[:, 1], min=0).mean()

        # bg_pts = rendervar['means3D'][~is_fg]
        # bg_rot = rendervar['rotations'][~is_fg]
        # losses['bg'] = l1_loss_v2(bg_pts, variables["init_bg_pts"]) + l1_loss_v2(bg_rot, variables["init_bg_rot"])

        losses['soft_col_cons'] = l1_loss_v2(params['rgb_colors'], variables["prev_col"])

    loss_weights = {'im': 1.0, 'seg': 3.0, 'rigid': 4.0, 'rot': 4.0, 'iso': 2.0, 'floor': 2.0, 'bg': 20.0,
                    'soft_col_cons': 0.01}
    loss = sum([loss_weights[k] * v for k, v in losses.items()])
    # seen = radius > 0
    # variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    # variables['seen'] = seen
    return loss, variables, out


def initialize_per_timestep(params, variables, optimizer):
    pts = params['means3D']
    # rot = torch.nn.functional.normalize(params['unnorm_rotations'])
    new_pts = pts + (pts - variables["prev_pts"])
    # new_rot = torch.nn.functional.normalize(rot + (rot - variables["prev_rot"]))

    # is_fg = params['seg_colors'][:, 0] > 0.5
    # prev_inv_rot_fg = rot
    # prev_inv_rot_fg[:, 1:] = -1 * prev_inv_rot_fg[:, 1:]
    fg_pts = pts
    prev_offset = fg_pts[variables["neighbor_indices"]] - fg_pts[:, None]
    # variables['prev_inv_rot_fg'] = prev_inv_rot_fg.detach()
    variables['prev_offset'] = prev_offset.detach()
    variables["prev_col"] = params['rgb_colors'].detach()
    variables["prev_pts"] = pts.detach()
    # variables["prev_rot"] = rot.detach()

    # new_params = {'means3D': new_pts, 'unnorm_rotations': new_rot}
    new_params = {'means3D': new_pts}
    params = update_params_and_optimizer(new_params, params, optimizer)

    return params, variables


def initialize_post_first_timestep(params, variables, optimizer, num_knn=20):
    # is_fg = params['seg_colors'][:, 0] > 0.5
    init_fg_pts = params['means3D']
    # init_bg_pts = params['means3D'][~is_fg]
    # init_bg_rot = torch.nn.functional.normalize(params['unnorm_rotations'][~is_fg])
    neighbor_sq_dist, neighbor_indices = o3d_knn(init_fg_pts.detach().cpu().numpy(), num_knn)
    neighbor_weight = np.exp(-2000 * neighbor_sq_dist)
    neighbor_dist = np.sqrt(neighbor_sq_dist)
    variables["neighbor_indices"] = torch.tensor(neighbor_indices).cuda().long().contiguous()
    variables["neighbor_weight"] = torch.tensor(neighbor_weight).cuda().float().contiguous()
    variables["neighbor_dist"] = torch.tensor(neighbor_dist).cuda().float().contiguous()

    # variables["init_bg_pts"] = init_bg_pts.detach()
    # variables["init_bg_rot"] = init_bg_rot.detach()
    variables["prev_pts"] = params['means3D'].detach()
    # variables["prev_rot"] = torch.nn.functional.normalize(params['unnorm_rotations']).detach()
    params_to_fix = ['logit_opacities', 'log_scales', 'cam_m', 'cam_c']
    for param_group in optimizer.param_groups:
        if param_group["name"] in params_to_fix:
            param_group['lr'] = 0.0
    return variables


def report_progress(params, data, i, progress_bar, every_i=100):
    if i % every_i == 0:
        im, _, _, = Renderer(raster_settings=data['cam'])(**params2rendervar(params))
        curr_id = data['id']
        im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]
        psnr = calc_psnr(im, data['im']).mean()
        progress_bar.set_postfix({"train img 0 PSNR": f"{psnr:.{7}f}"})
        progress_bar.update(every_i)


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
    camera_centers = np.stack(camera_centers, axis=0)
    # scene_radius = 1.1 * np.max(np.linalg.norm(camera_centers - np.mean(camera_centers, 0)[None], axis=-1))
    scene_radius = args.scene_scale
    log(blue(f'scene_radius: {scene_radius}'))

    # Load initial gaussian
    params, variables = initialize_params(args.init_gs_path, fix_occ=True, fix_scaling=True)
    device = params['means3D'].device
    optimizer = initialize_optimizer(params, scene_radius)
    # output_params = []

    frames = range(args.frame_start, args.frame_end)
    for f in frames:
        dataset = get_dataset(args.data_root, f, cameras, scale=args.scale)
        is_initial_timestep = (f == args.frame_start)
        if not is_initial_timestep:
            params, variables = initialize_per_timestep(params, variables, optimizer)
        num_iter_per_timestep = 0 if is_initial_timestep else args.num_iter_per_timestep
        if num_iter_per_timestep > 0:
            sampler = RandomSampler(dataset, replacement=True, num_samples=num_iter_per_timestep)
            batch_sampler = BatchSampler(sampler, batch_size=1, drop_last=False)
            dataloader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=4)
            progress_bar = tqdm(range(num_iter_per_timestep), desc=f"training frame {f}")
            for batch_id, batch in enumerate(dataloader):
                camera_name = batch.pop('camera_name')
                meta = batch.pop('meta')
                batch = to_cuda(to_tensor(batch), device=device)
                for i in range(len(meta.H)):
                    camera = convert_to_gaussian_camera(
                        K=batch.K[i], R=batch.R[i], T=batch.T[i],
                        H=batch.H[i], W=batch.W[i], n=batch.n[i], f=batch.f[i],
                        cpu_K=meta.K[i], cpu_R=meta.R[i], cpu_T=meta.T[i],
                        cpu_H=meta.H[i], cpu_W=meta.W[i], cpu_n=meta.n[i], cpu_f=meta.f[i],
                    )
                    raster_settings = GaussianRasterizationSettings(
                        image_height=camera.image_height,
                        image_width=camera.image_width,
                        tanfovx=camera.tanfovx,
                        tanfovy=camera.tanfovy,
                        bg=torch.full([3], 0.0, device=device),  # GPU
                        scale_modifier=1.0,
                        viewmatrix=camera.world_view_transform,
                        projmatrix=camera.full_proj_transform,
                        sh_degree=0,
                        campos=camera.camera_center,
                        prefiltered=False,
                        debug=False,
                    )
                    curr_data = {
                        'im': batch.image[i],
                        'mask': batch.mask[i]
                    }
                    loss, variables, out = get_loss(params, curr_data, variables, raster_settings, is_initial_timestep)
                loss.backward()
                if isinstance(optimizer, SparseGaussianAdam):
                    radii = out['radii']
                    visible = radii > 0
                    optimizer.step(visible, radii.shape[0])
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    if (batch_id + 1) % args.log_interval == 0:
                        gt = curr_data['im'] * curr_data['mask']
                        psnr = calc_psnr(out['img'], gt).mean()
                        log(blue(f'training psnr of iter {batch_id}: {psnr:.{4}f}'))
                        progress_bar.update(args.log_interval)
                #     report_progress(params, dataset[0], batch_id, progress_bar)
                #     if is_initial_timestep:
                #         params, variables = densify(params, variables, optimizer, i)

            progress_bar.close()
            # paramscpu = params2cpu(params, is_initial_timestep)
        save_path = os.path.join(args.save_dir, f"{f:06d}.ply")
        write_params(save_path, params)
        # output_params.append(paramscpu)
        if is_initial_timestep:
            variables = initialize_post_first_timestep(params, variables, optimizer)
        # torch.cuda.empty_cache()
    # save_params(output_params, seq, exp)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--init_gs_path", type=str)
    parser.add_argument("--save_dir", type=str, default='./data/results')
    parser.add_argument("--frame_start", type=int, default=400)
    parser.add_argument("--frame_end", type=int, default=401)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--scene_scale", type=float, default=2.0)
    parser.add_argument("--num_iter_per_timestep", type=int, default=1200)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--log_interval", type=int, default=200)
    args = parser.parse_args()

    if args.exp_name == None:
        seq = os.path.dirname(args.data_root).split('/')[-1]
        args.save_dir = os.path.join(args.save_dir, seq)
    else:
        args.save_dir = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(args.save_dir, exist_ok=True)

    train(args)
    torch.cuda.empty_cache()
