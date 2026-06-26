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

    def _surroundocc_samples_dir(self):
        nested = os.path.join(self.occ_base_path, 'nuscenes_extra', 'nuscenes_occ', 'samples')
        if os.path.isdir(nested):
            return nested
        return self.occ_base_path

    def _occ_file_for_sample(self, sample_token):
        sample = self.dataset.get('sample', sample_token)
        lidar_sample = self.dataset.get('sample_data', sample['data']['LIDAR_TOP'])
        lidar_name = os.path.basename(lidar_sample['filename'])
        return os.path.join(self._surroundocc_samples_dir(), lidar_name + '.npy')

    def _sample_has_occ_data(self, sample_token):
        if not self.enable_occ_supervision or not self.filter_missing_occ:
            return True
        return os.path.exists(self._occ_file_for_sample(sample_token))

    def _load_occ_data(self, occ_file):
        sparse_occ = np.load(occ_file, allow_pickle=False)
        if sparse_occ.ndim != 2 or sparse_occ.shape[1] < 4:
            raise ValueError(f"SurroundOcc 文件格式错误: {occ_file}, shape={sparse_occ.shape}")

        coords = sparse_occ[:, :3].astype(np.int64)
        labels = sparse_occ[:, 3].astype(np.uint8)
        valid = (
            (coords[:, 0] >= 0) & (coords[:, 0] < 200) &
            (coords[:, 1] >= 0) & (coords[:, 1] < 200) &
            (coords[:, 2] >= 0) & (coords[:, 2] < 16)
        )
        coords = coords[valid]
        labels = labels[valid]

        occ_semantics = np.full((200, 200, 16), 17, dtype=np.uint8)
        occ_semantics[coords[:, 0], coords[:, 1], coords[:, 2]] = labels
        occ_visible_mask = np.zeros((200, 200, 16), dtype=np.uint8)
        occ_visible_mask[coords[:, 0], coords[:, 1], coords[:, 2]] = 1
        occ_surface = (occ_semantics != 17) & (occ_visible_mask > 0)
        return {
            'occ_semantics': occ_semantics,
            'occ_mask_camera': None,
            'occ_mask_lidar': occ_visible_mask,
            'occ_surface': occ_surface.astype(np.uint8),
            'occ_visible_mask': occ_visible_mask,
        }

    def _empty_occ_data(self):
        return {
            'occ_semantics': None,
            'occ_mask_camera': None,
            'occ_mask_lidar': None,
            'occ_surface': None,
            'occ_visible_mask': None,
        }

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

        # Stage1 3D training only consumes context_frames for reconstruction and
        # all_dict for rendering. Avoid returning large unused tensors multiple
        # times: color_org is the original 900x1600 image and costs ~100MB per
        # sample copy after collate.
        context_keys = {
            'idx', 'token', 'scene_token', 'scene_name', 'scene_idx', 'sensor_name',
            'filename', 'timestamp', 'gt_depth', 'ego_pose', 'mask',
            'occ_render_mask', 'occ_semantics', 'occ_mask_camera', 'occ_mask_lidar',
            'occ_surface', 'occ_visible_mask', 'K', 'c2e_extr',
            'intrinsics', 'extrinsics', ('color_aug', 0),
        }
        render_keys = {
            'idx', 'token', 'scene_token', 'scene_name', 'scene_idx', 'sensor_name',
            'filename', 'timestamp', 'gt_depth', 'ego_pose', 'mask',
            'K', 'c2e_extr', 'intrinsics', 'extrinsics', ('color_aug', 0),
        }
        slim_cur_keys = {
            'idx', 'token', 'scene_token', 'scene_name', 'scene_idx',
            'sensor_name', 'filename', 'timestamp',
        }

        all_context_dict = {k: v for k, v in cur_sample.items() if k in context_keys}
        all_dict = {k: v for k, v in cur_sample.items() if k in render_keys}
        slim_cur_sample = {k: v for k, v in cur_sample.items() if k in slim_cur_keys}

        ret_sample = {
            'cur_sample': slim_cur_sample,
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
            occ_file = self._occ_file_for_sample(frame_idx)
            if os.path.exists(occ_file):
                try:
                    data.update(self._load_occ_data(occ_file))
                except Exception as e:
                    print(f"警告: 加载 Occ 数据失败 {occ_file}: {e}")
                    data.update(self._empty_occ_data())
            else:
                # 如果文件不存在，设置为 None
                data.update(self._empty_occ_data())

        return data
