from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
# from diffusion_policy.common.sampler import (
#     SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.common.full_seq_sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer

# for visualizing
import pygame
from skvideo.io import vwrite

# from ..common.pytorch_util import dict_apply
# from ..common.replay_buffer import ReplayBuffer
# from ..common.sampler import (
#     SequenceSampler, get_val_mask, downsample_mask)
# from ..model.common.normalizer import LinearNormalizer
# from ..dataset.base_dataset import BaseImageDataset
# from ..common.normalize_util import get_image_range_normalizer



class PushTImageDataset(BaseImageDataset):
    def __init__(self,
            zarr_path,
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None
            ):

        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['img', 'state', 'action'])
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'][...,:2]
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:,:2].astype(np.float32) # (agent_posx2, block_posex3)
        image = np.moveaxis(sample['img'],-1,1)/255

        data = {
            'obs': {
                'image': image, # T, 3, 96, 96
                'agent_pos': agent_pos, # T, 2
            },
            'action': sample['action'].astype(np.float32) # T, 2
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


def test():
    import os
    zarr_path = os.path.expanduser('data/pusht/pusht_cchi_v7_replay.zarr')
    dataset = PushTImageDataset(zarr_path, horizon=16)

    from matplotlib import pyplot as plt
    normalizer = dataset.get_normalizer()
    nactions = normalizer['action'].normalize(dataset.replay_buffer['action'])
    diff = np.diff(nactions, axis=0)
    dists = np.linalg.norm(np.diff(nactions, axis=0), axis=-1)
    print("dists", dists)

    # sample from dataset
    for instance in range(206):
        idx = instance
        # sample = dataset.sampler.sample_sequence(idx)
        sample = dataset.sampler.sample_full_sequence(idx)
        print("sample", sample.keys())
        print("sample imgs", sample['img'].shape)
        print("sample state", sample['state'].shape)
        print("sample action", sample['action'].shape)
        data = dataset._sample_to_data(sample)
        # print("data", data)
        torch_data = dict_apply(data, torch.from_numpy)

        from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
        env = PushTImageEnv()
        # get first observation
        obs, info = env.reset()
        imgs = []
        for i in range(sample['state'].shape[0]):
            env.set_to_state(sample['state'][i])
            env.step(sample['action'][i])
            imgs.append(env.render(mode='rgb_array'))

        from IPython.display import Video
        vwrite(f'vis_traj_each_demo/vis_traj_{instance}.mp4', imgs)
        Video(f'vis_traj_each_demo/vis_traj_{instance}.mp4', embed=True, width=256, height=256)


    # pygame.init()
    # pygame.display.init()
    # window_size = 96
    # canvas = pygame.Surface((window_size, window_size))
    # canvas.fill((255, 255, 255))
    # screen = canvas

if __name__ == '__main__':
    test()
