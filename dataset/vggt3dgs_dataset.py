#----------------------------------------------------------------#
# ReconDrive                                                     #
# Source code: https://github.com/TuojingAI/ReconDrive           #
# Copyright (c) TuojingAI. All rights reserved.                  #
#----------------------------------------------------------------#

import os
from typing import Any, Dict

import numpy as np
import torch
from nuscenes.utils import splits

from dataset.data_util import align_dataset
from dataset.vggt4dgs_dataset import NuScenesdataset4D, custom_collate_fn


class NuScenesdataset3D(NuScenesdataset4D):
    """
    NuScenes single-frame dataset for stage1 3D Gaussian training.
    Samples all frames from all scenes without temporal context.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bwd = 0
        self.fwd = 0
        self.has_context = False

        # 新增：Occ 数据加载配置
        self.enable_occ_supervision = kwargs.get('enable_occ_supervision', False)
        self.filter_missing_occ = kwargs.get('filter_missing_occ', False)
        if self.enable_occ_supervision:
            # 从配置中读取 Occ 数据路径
            self.occ_base_path = kwargs.get('occ_data_path', None)
            if self.occ_base_path is None:
                raise ValueError("启用 Occ 监督时必须提供 occ_data_path 配置")
            if not os.path.exists(self.occ_base_path):
                raise FileNotFoundError(f"Occ 数据目录未找到: {self.occ_base_path}")
            print(f"启用 Occ 数据加载，路径: {self.occ_base_path}")

        self.rebuild_sample_index(announce=True)

    def _sample_has_occ_data(self, sample_token):
        if not self.enable_occ_supervision or not self.filter_missing_occ:
            return True
        occ_file = os.path.join(self.occ_base_path, sample_token, 'labels.npz')
        return os.path.exists(occ_file)

    def __getitem__(self, idx: int, context_frame_idx: int = -1, return_all: bool = False) -> Dict[str, Any]:
        actual_idx = idx
        frame_idx = self.sample_tokens[actual_idx]
        scene_name, scene_token, scene_data, scene_sample_count, local_index_in_scene, scene_idx = self.get_scene_index_and_count(actual_idx)

        cur_sample = self.get_frame(
            idx=idx,
            frame_idx=frame_idx,
            scene_token=scene_token,
            scene_name=scene_name,
            scene_idx=scene_idx,
            is_key_frame=False,
        )
        cur_sample = align_dataset(cur_sample)
        if 'K' in cur_sample:
            cur_sample['intrinsics'] = cur_sample['K'][..., :3, :3]
        if 'c2e_extr' in cur_sample:
            cur_sample['extrinsics'] = cur_sample['c2e_extr']

        all_context_dict = {}
        all_dict = {}
        for k, v in cur_sample.items():
            if torch.is_tensor(v):
                all_dict[k] = v
                all_context_dict[k] = v
            elif isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                all_dict[k] = v
                all_context_dict[k] = v
            else:
                all_dict[k] = v
                all_context_dict[k] = v

        ret_sample = {
            'cur_sample': cur_sample,
            'context_frames': all_context_dict,
            'target_frames': {},
            'all_dict': all_dict,
        }
        return ret_sample

    def get_frame(self, idx, frame_idx, scene_token, scene_name, scene_idx, is_key_frame=False):
        """重写父类方法以添加 Occ 数据加载"""
        # 调用父类方法获取基础数据
        data = super().get_frame(idx, frame_idx, scene_token, scene_name, scene_idx, is_key_frame)

        # 新增：加载 Occ 数据
        if self.enable_occ_supervision:
            occ_file = os.path.join(self.occ_base_path, frame_idx, 'labels.npz')
            if os.path.exists(occ_file):
                try:
                    occ_data = np.load(occ_file)
                    occ_semantics = occ_data['semantics']
                    occ_mask_camera = occ_data.get('mask_camera', None)
                    occ_mask_lidar = occ_data.get('mask_lidar', None)
                    visible_mask = occ_mask_lidar if occ_mask_lidar is not None else occ_mask_camera
                    surface_occ = None
                    if visible_mask is not None:
                        surface_occ = (occ_semantics > 0) & (visible_mask > 0)
                    data.update({
                        'occ_semantics': occ_semantics,  # [200,200,16] uint8
                        'occ_mask_camera': occ_mask_camera,
                        'occ_mask_lidar': occ_mask_lidar,
                        'occ_surface': surface_occ.astype(np.uint8) if surface_occ is not None else None,
                        'occ_visible_mask': visible_mask,
                    })
                except Exception as e:
                    print(f"警告: 加载 Occ 数据失败 {occ_file}: {e}")
                    data.update({
                        'occ_semantics': None,
                        'occ_mask_camera': None,
                        'occ_mask_lidar': None,
                        'occ_surface': None,
                        'occ_visible_mask': None,
                    })
            else:
                # 如果文件不存在，设置为 None
                data.update({
                    'occ_semantics': None,
                    'occ_mask_camera': None,
                    'occ_mask_lidar': None,
                    'occ_surface': None,
                    'occ_visible_mask': None,
                })

        return data
