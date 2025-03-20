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
        self.original_replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=[obs_key, state_key, action_key])
        
        self.val_ratio = val_ratio
        self.max_train_episodes = max_train_episodes
        self.seed = seed
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.obs_key = obs_key
        self.state_key = state_key
        self.action_key = action_key

        


    def set_subset(self, episode_idxs):
        # create copy of replay buffer
        self.replay_buffer = copy.deepcopy(self.original_replay_buffer)
        list_to_include = []
        for ep_idx in range(self.original_replay_buffer.n_episodes):
            if ep_idx in episode_idxs:
                list_to_include.append(ep_idx)
        self.replay_buffer.include_only_episode_indices(list_to_include)
        # pdb.set_trace()
        self.initialize_after_subset()
        

    def initialize_after_subset(self):
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=self.val_ratio,
            seed=self.seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=self.max_train_episodes, 
            seed=self.seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=train_mask
            )
        self.obs_key = self.obs_key
        self.state_key = self.state_key
        self.action_key = self.action_key
        self.train_mask = train_mask
        self.horizon = self.horizon
        self.pad_before = self.pad_before
        self.pad_after = self.pad_after


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
    
    def get_combined_normalizer(self, recovery_dataset):
        data = self._sample_to_data(self.replay_buffer)
        recovery_data = recovery_dataset._sample_to_data(recovery_dataset.replay_buffer)
        data['obs'] = np.concatenate([data['obs'], recovery_data['obs']], axis=0)
        data['action'] = np.concatenate([data['action'], recovery_data['action']], axis=0)
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode='limits')
        return normalizer


    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer[self.action_key])

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        # print("sample", sample)
        keypoint = sample[self.obs_key]
        state = sample[self.state_key]
        agent_pos = state[:,:2]
        obs = np.concatenate([
            keypoint.reshape(keypoint.shape[0], -1), 
            agent_pos], axis=-1)

        data = {
            'obs': obs, # T, D_o
            'action': sample[self.action_key], # T, D_a
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # print("idx", idx)
        # pdb.set_trace()
        sample = self.sampler.sample_sequence(idx)
        
        
        # print("idx", idx)
        
        data = self._sample_to_data(sample)
        # print("data", data['idx'])
        # pdb.set_trace()

        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
