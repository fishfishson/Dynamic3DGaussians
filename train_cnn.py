import os
import glob
import numpy as np
from natsort import natsorted
import argparse
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset 
import pytorch_lightning as pl

from easyvolcap.utils.base_utils import *
from easyvolcap.utils.console_utils import *
# from easyvolcap.utils.mvp_utils import MVPEncoder, MlpMapsDecoder
from easyvolcap.utils.easy_utils import read_camera


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


class MvRgbDataset(Dataset):
    def __init__(
        self,
        cam_dir,
        render_dir,
        ply_dir,
        mode: str = 'hand',
        sh_deg: int = 1,
        use_head_iso: bool = False,
        use_body_iso: bool = False,
        use_hand_iso: bool = False,
    ):
        super(MvRgbDataset, self).__init__()
        self.cam_dir = cam_dir
        self.render_dir = render_dir
        self.ply_dir = ply_dir
        self.mode = mode
        self.sh_deg = sh_deg
        self.use_head_iso = use_head_iso
        self.use_body_iso = use_body_iso
        self.use_hand_iso = use_hand_iso

        self.load_cam_data()
        self.load_ply_data()

        self.view_list = list(range(self.view_num))
        self.ply_list = list(range(self.ply_num))
        
        self.data_list = []
        for ply_idx in self.ply_list:
            for view_idx in self.view_list:
                self.data_list.append([ply_idx, view_idx])

        print('# Dataset contains %d items' % len(self))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.getitem(index)

    def load_cam_data(self):
        cam_data = read_camera(self.cam_dir)
        self.cam_names = natsorted(list(cam_data.keys()))
        self.view_num = len(self.cam_names)
        self.Ks = []
        self.Rs = []
        self.Ts = []
        for view_idx in range(self.view_num):
            K = np.array(cam_data[self.cam_names[view_idx]]['K'], np.float32).reshape(3, 3)
            R = np.array(cam_data[self.cam_names[view_idx]]['R'], np.float32).reshape(3, 3)
            T = np.array(cam_data[self.cam_names[view_idx]]['T'], np.float32).reshape(3, 1)
            self.Ks.append(K)
            self.Rs.append(R)
            self.Ts.append(T)
        self.Ks = np.stack(self.Ks, axis=0)
        self.Rs = np.stack(self.Rs, axis=0)
        self.Ts = np.stack(self.Ts, axis=0)

    def load_ply_data(self):
        self.ply = {}
        self.ply_names = {}
        if self.mode == 'head' or self.mode == 'all':    
            self.ply['head'] = []
            self.ply_names['head'] = []
            head_ply = glob.glob(os.path.join(self.ply_dir, 'head*.ply'))
            head_ply = natsorted(head_ply)
            for ply in tqdm(head_ply, desc='loading head ply'):
                gs = initialize_params(ply, sh_deg=self.sh_deg, use_iso=self.use_head_iso)
                for p in gs.parameters(): p.requires_grad = False
                self.ply['head'].append(gs)
                self.ply_names['head'].append(os.path.basename(ply))
        if self.mode == 'body' or self.mode == 'all':
            self.ply['body'] = []
            self.ply_names['body'] = []
            body_ply = glob.glob(os.path.join(self.ply_dir, 'body*.ply'))
            body_ply = natsorted(body_ply)
            for ply in tqdm(body_ply, desc='loading body ply'):
                gs = initialize_params(ply, sh_deg=self.sh_deg, use_iso=self.use_body_iso)
                for p in gs.parameters(): p.requires_grad = False
                self.ply['body'].append(gs)
                self.ply_names['body'].append(os.path.basename(ply))
        if self.mode == 'hand' or self.mode == 'all':
            self.ply['lhand'] = []
            self.ply['rhand'] = []
            self.ply_names['lhand'] = []
            self.ply_names['rhand'] = []
            lhand_ply = glob.glob(os.path.join(self.ply_dir, 'lhand*.ply'))
            lhand_ply = natsorted(lhand_ply)
            for ply in tqdm(lhand_ply, desc='loading lhand ply'):
                gs = initialize_params(ply, sh_deg=self.sh_deg, use_iso=self.use_hand_iso)
                for p in gs.parameters(): p.requires_grad = False
                self.ply['lhand'].append(gs)
                self.ply_names['lhand'].append(os.path.basename(ply))
            rhand_ply = glob.glob(os.path.join(self.ply_dir, 'rhand*.ply'))
            rhand_ply = natsorted(rhand_ply)
            for ply in tqdm(rhand_ply, desc='loading rhand ply'):
                gs = initialize_params(ply, sh_deg=self.sh_deg, use_iso=self.use_hand_iso)
                for p in gs.parameters(): p.requires_grad = False
                self.ply['rhand'].append(gs)
                self.ply_names['rhand'].append(os.path.basename(ply))
        for k in self.ply.keys():
            self.ply_num = len(self.ply[k])
            break

    def getitem(self, index):
        ply_idx, view_idx = self.data_list[index]

        ply_data = {}
        for k in self.ply.keys():
            ply_data[k] = {}
            ply_data[k]['xyz'] = self.ply[k][ply_idx].get_xyz
            ply_data[k]['features'] = self.ply[k][ply_idx].get_features
            ply_data[k]['scaling'] = self.ply[k][ply_idx].get_scaling
            ply_data[k]['rotation'] = self.ply[k][ply_idx].get_rotation
            ply_data[k]['opacity'] = self.ply[k][ply_idx].get_opacity

        ply_name = self.ply_names[k][ply_idx]
        frame = ply_name.split('.')[0].split('-')[-1]
        img = cv2.imread()

        return {
            'K': self.Ks[view_idx],
            'R': self.Rs[view_idx],
            'T': self.Ts[view_idx],
            'ply_data': ply_data,
            'view_idx': view_idx,
            'ply_idx': ply_idx,
            'frame_idx': int(frame),
        }

def main(args):
    # encoder = MVPEncoder(ninputs=2, size=(640,270), code_dim=256)
    # inp = torch.randn(1, 6, 640, 270)
    # code, kldiv = encoder(inp)
    # decoder = MlpMapsDecoder(inch=256, geo_ch=10, rgb_ch=12)
    # geo, rgb = decoder(code)

    dataset = MvRgbDataset(
        cam_dir=args.cam_dir,
        render_dir=args.render_dir,
        ply_dir=args.ply_dir,
    )
    data = dataset[0]
    breakpoint()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam_dir", type=str)
    parser.add_argument("--render_dir", type=str)
    parser.add_argument("--ply_dir", type=str)
    args = parser.parse_args()
    main(args)
