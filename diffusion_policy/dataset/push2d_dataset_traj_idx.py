from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
import pdb
from transformers import BertTokenizer, BertModel


class Push2dLowdimDataset(BaseLowdimDataset):
    def __init__(self, 
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            obs_key='keypoint',
            state_key='state',
            action_key='action',
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None
            ):
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=[obs_key, state_key, action_key])
        
        

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
            episode_mask=train_mask
            )
        self.obs_key = obs_key
        self.state_key = state_key
        self.action_key = action_key
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after


        self.dbert = 256
        

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
        # pdb.set_trace()
        data = self._sample_to_data(self.replay_buffer)
        normalizer = LinearNormalizer()
        # pdb.set_trace()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer[self.action_key])

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        keypoint = sample[self.obs_key]
        state = sample[self.state_key]
        agent_pos = state[:,:2]
        obs = np.concatenate([
            keypoint.reshape(keypoint.shape[0], -1), 
            agent_pos], axis=-1)
        
        
        # concatenate random conditioning to obs, conditioning should be 1d
        # conditioning = np.ones((obs.shape[0], 1))
        # multiply by a uniform random number between 0 and 1
        # conditioning *= np.random.uniform(0, 1, conditioning.shape)
        # obs = np.concatenate([obs, conditioning], axis=-1)
        
        if 'idx' not in sample:
            trajectory_idx = self.replay_buffer.get_episode_idxs()
        else:
            # print(f'idx: {idx}')
            
            # trajectory_idx = np.expand_dims(self.replay_buffer.get_episode_idxs()[idx], 0)
            trajectory_idx = sample['idx']
            # assert that all are the same
            # assert np.var(trajectory_idx) == 0
            # print(f'trajectory_idx: {trajectory_idx}')
            # pdb.set_trace()

        # random_input between -1 and 1 of shape trajectory_idx.shape
        # random_input = np.random.uniform(-1, 1, trajectory_idx.shape)
        # pdb.set_trace()
        # set trajectory_idx values to 0 if they are even and 1 if they are odd
        conditioning_context = np.zeros((trajectory_idx.shape[0], self.dbert))
        for i in range(trajectory_idx.shape[0]):
            # assert that all values in trajectory_idx are the same
            # assert np.var(trajectory_idx[i]) == 0
            # conditioning_context[i] = trajectory_idx[i]
            # print("trajectory_idx", trajectory_idx[i])
        #     # duplicate the same context for all the time steps in the trajectory
            conditioning_context[i] = np.ones(self.dbert) * trajectory_idx[i]
        # print("conditioning_context", conditioning_context.shape)
        # print("obs", obs.shape)
        # print("trajectory_idx", np.array([trajectory_idx]))
        # pdb.set_trace()
        data = {
            'obs': obs, # T, D_o
            'action': sample[self.action_key], # T, D_a
            'idx': conditioning_context # scalar
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        
        # print("idx", idx)
        
        data = self._sample_to_data(sample)
        # print("data", data['idx'])
        # pdb.set_trace()

        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
