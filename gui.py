import sys
sep_ind = sys.argv.index('--') if '--' in sys.argv else len(sys.argv)
our_args = sys.argv[1:sep_ind]
evc_args = sys.argv[sep_ind + 1:]
sys.argv = [sys.argv[0]] + ['-t', 'train'] + evc_args + []  # disable log and use custom logging mechanism

# from easyvolcap.scripts.main import gui
from easyvolcap.engine import cfg

import os
import torch
import numpy as np

from helpers import params2rendervar
from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer

from easyvolcap.utils.base_utils import *
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.easy_utils import read_camera
from easyvolcap.utils.data_utils import to_cuda
from easyvolcap.utils.sh_utils import SH2RGB, RGB2SH
from easyvolcap.runners.custom_viewer import Viewer


class CustomViewer(Viewer):
    def __init__(self,
                 window_size=[1080, 1920],  # height, width
                 window_title: str = f'EasyVolcap Viewer Custom Window',  # MARK: global config
                 fullscreen: bool = False,
                 camera_cfg: dotdict = None,
                 params: dotdict = None,
                ):
        super(CustomViewer, self).__init__(
            window_size=window_size,
            window_title=window_title,
            fullscreen=fullscreen,
            camera_cfg=camera_cfg)
        self.params = params

    def custom_render(self, batch):
        """
        def convert_to_gaussian_camera(K: torch.Tensor,
                               R: torch.Tensor,
                               T: torch.Tensor,
                               H: torch.Tensor,
                               W: torch.Tensor,
                               n: torch.Tensor,
                               f: torch.Tensor,
                               cpu_K: torch.Tensor,
                               cpu_R: torch.Tensor,
                               cpu_T: torch.Tensor,
                               cpu_H: int,
                               cpu_W: int,
                               cpu_n: float = 0.01,
                               cpu_f: float = 100.,
                               ):

        gaussian_camera = convert_to_gaussian_camera(
            K=to_cuda(batch.K),
            R=to_cuda(batch.R),
            T=to_cuda(batch.T),
            H=to_cuda(batch.H),
            W=to_cuda(batch.W),
            n=torch.tensor(0.01, device='cuda', dtype=torch.float32),
            f=torch.tensor(100., device='cuda', dtype=torch.float32),
            cpu_K=batch.K,
            cpu_R=batch.R,
            cpu_T=batch.T,
            cpu_H=batch.H,
            cpu_W=batch.W,
            cpu_n=0.01,
            cpu_f=100.,
        )
        """
        K_np = batch.K.numpy()
        R_np = batch.R.numpy()
        T_np = batch.T.numpy().reshape(3,)
        
        camera = batch.camera
        
        raster_settings = GaussianRasterizationSettings(
            image_height=camera.image_height,
            image_width=camera.image_width,
            tanfovx=camera.tanfovx,
            tanfovy=camera.tanfovy,
            bg=torch.full([3], 0.0, device='cuda'),  # GPU
            scale_modifier=1.0,
            viewmatrix=camera.world_view_transform,
            projmatrix=camera.full_proj_transform,
            sh_degree=0,
            campos=camera.camera_center,
            prefiltered=False,
            debug=False,
        )
        rendervar = params2rendervar(self.params, retain2D=False)
        im, dpt, acc, radius = GaussianRasterizer(raster_settings=raster_settings)(**rendervar)

        im = im.permute(1, 2, 0).contiguous()
        im = torch.cat([im, torch.ones_like(im[..., :1])], dim=-1)
        return im


def initialize_gaussian(ply_path, sh_deg=0, scene_radius=1.0):
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
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (sh_deg + 1) ** 2 - 1))
    features_rest = torch.from_numpy(features_extra).float()

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
        'rgb_colors': SH2RGB(features_dc)[..., 0],
        # 'seg_colors': np.stack((seg, np.zeros_like(seg), 1 - seg), -1),
        'unnorm_rotations': rots,
        'logit_opacities': opacities,
        'log_scales': scales,
        # 'cam_m': np.zeros((max_cams, 3)),
        # 'cam_c': np.zeros((max_cams, 3)),
    }
    params = {k: torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True)) for k, v in
              params.items()}
    # cam_centers = np.linalg.inv(md['w2c'][0])[:, :3, 3]  # Get scene radius
    # scene_radius = 1.1 * np.max(np.linalg.norm(cam_centers - np.mean(cam_centers, 0)[None], axis=-1))
    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'scene_radius': scene_radius,
                 'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'denom': torch.zeros(params['means3D'].shape[0]).cuda().float()}
    return params, variables


def main(args):
    params, variables = initialize_gaussian(args.ply_path)    
    
    # with open(os.path.join(model.data_root, "transforms_test.json")) as json_file:
    #     contents = json.load(json_file)
    # frames = contents['frames']
    # idx = 0
    # frame = frames[0]
    # timestamp = frame.get('time', 0.0)
    # if model.frame_ratio > 1:
    #     timestamp /= model.frame_ratio
    # w2c = np.array(frame["transform_matrix"])
    # R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
    # T = w2c[:3, 3]
    # FovX = FovY = -1.0
    # fl_x = frame['fl_x']
    # fl_y = frame['fl_y']
    # cx = frame['cx']
    # cy = frame['cy']
    # test_cam_infos = [
    #     CameraInfo(
    #         uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, 
    #         image=None, depth=None,
    #         image_path="", 
    #         image_name="", 
    #         width=int(cx)*2, height=int(cy)*2, 
    #         timestamp=timestamp,
    #         fl_x=fl_x, fl_y=fl_y, cx=cx, cy=cy
    #     )
    # ]
    # test_cam = cameraList_from_camInfos(test_cam_infos, resolution_scale=1.0, args=model)

    cameras = read_camera(args.data_root)
    camera_names = sorted(list(cameras.keys()))
    default_camera = cameras[camera_names[0]]
    # downsample
    K = default_camera['K'].copy()
    if 'flame_salmon' in args.data_root:
        K[:2] *= 0.5
        default_camera['K'] = K
        default_camera['W'] = int(K[0,2]*2)
        default_camera['H'] = int(K[1,2]*2)
        camera_cfg = dotdict(
            H=default_camera['H'],
            W=default_camera['W'],
            K=default_camera['K'],
            R=default_camera['R'],
            T=default_camera['T'],
        )
    elif 'bike' in args.data_root:
        print('bike scene')
        default_camera['K'] = K
        default_camera['H'] = 1080
        default_camera['W'] = 1080
        world_up = [0, -1, 0]
        camera_cfg = dotdict(
            H=default_camera['H'],
            W=default_camera['W'],
            K=default_camera['K'],
            R=default_camera['R'],
            T=default_camera['T'],
            world_up=world_up,
        )
    elif 'goodcha' in args.data_root:
        print('goodcha scene')
        R = np.array([[0.7493726045, 0.0747454784, -0.6579162660], 
                      [-0.0250241057, 0.9960953519, 0.0846631210], 
                      [0.6616755199, -0.0469804573, 0.7483170070]
                    ])
        T = np.array([1.775169, -0.025225, 1.74254])
        K = np.array([[1515.7024299390, 0.0000000000, 527.2500000000], 
                      [0.0000000000, 1514.0517651239, 940.0000000000], 
                      [0.0000000000, 0.0000000000, 1.0000000000]
                    ])
        world_up = [0, -1, 0]
        origin = [0.68, 1.06, 5.32]
        default_camera['K'] = K
        default_camera['H'] = 1880
        default_camera['W'] = 1539
        default_camera['R'] = R
        default_camera['T'] = T
        camera_cfg = dotdict(
            H=default_camera['H'],
            W=default_camera['W'],
            K=default_camera['K'],
            R=default_camera['R'],
            T=default_camera['T'],
            world_up=world_up,
            origin=origin,
        )
    else:
        default_camera['K'] = K
        default_camera['H'] = 1080
        default_camera['W'] = 1920
        camera_cfg = dotdict(
            H=default_camera['H'],
            W=default_camera['W'],
            K=default_camera['K'],
            R=default_camera['R'],
            T=default_camera['T'],
        )

    viewer = CustomViewer(
        window_size=[default_camera['H'], default_camera['W']],
        camera_cfg=camera_cfg,
    )
        
    viewer.run()


if __name__ == '__main__':
    args = dotdict(
        data_root='data',
        ply_path='ply',
    )
    args = dotdict(vars(build_parser(args, description=__doc__).parse_args(our_args)))
    main(args)
    
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--data_root', type=str)
    # parser.add_argument('--ply_path', type=str)
    # args = parser.parse_args()
    # print(args)
    # main(args)
