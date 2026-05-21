#----------------------------------------------------------------#
# ReconDrive                                                     #
# Source code: https://github.com/TuojingAI/ReconDrive           #
# Copyright (c) TuojingAI. All rights reserved.                  #
#----------------------------------------------------------------#

from typing import Any, Dict

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
        self._build_all_frame_tokens()

    def _build_all_frame_tokens(self):
        if self.stage == 'train':
            official_scene_names = splits.train
        elif self.stage == 'val':
            official_scene_names = splits.val
        elif self.stage == 'test':
            official_scene_names = [
                'scene-0014', 'scene-0018', 'scene-0906', 'scene-0098',
                'scene-0100', 'scene-0103', 'scene-0270', 'scene-0271',
                'scene-0278', 'scene-0553', 'scene-0558',
                'scene-0802', 'scene-0968', 'scene-1065',
            ]
        else:
            raise ValueError("stage should be 'train' / 'val'/ 'test' ")

        self.sample_tokens = []
        self.scenes_data = []
        self.scene_names = []
        self.scene_tokens = []

        for scene in self.dataset.scene:
            if scene['name'] not in official_scene_names:
                continue
            scene_name = scene['name']
            scene_token = scene['token']
            sample_token = scene['first_sample_token']
            scene_sample_tokens = []
            while sample_token:
                scene_sample_tokens.append(sample_token)
                sample = self.dataset.get('sample', sample_token)
                sample_token = sample['next']

            if len(scene_sample_tokens) > 0:
                self.sample_tokens.extend(scene_sample_tokens)
                self.scenes_data.append(scene_sample_tokens)
                self.scene_names.append(scene_name)
                self.scene_tokens.append(scene_token)

        print('Num of samlpe_tokens: ', len(self.sample_tokens))

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
