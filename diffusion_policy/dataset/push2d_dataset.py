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

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.context_encoder = BertModel.from_pretrained("bert-base-uncased",output_hidden_states = True)
        self.dbert = 768
        self.set_context_lookup()

        # move context encoder to device
        # device = torch.device("cuda:0")
        # self.context_encoder.to(device)
    def set_context_lookup(self):
        # self.device = device
        # self.context_encoder.to(device)
        self.context_lookup = {}
        text_label = 'right'
        tokenized_inputs = self.tokenizer(text_label, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad(): 
            outputs = self.context_encoder(**tokenized_inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
        # store the embeddings
        self.context_lookup[text_label] = embeddings[0].cpu().numpy()

        text_label = 'left'
        tokenized_inputs = self.tokenizer(text_label, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.context_encoder(**tokenized_inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
        # store the embeddings
        self.context_lookup[text_label] = embeddings[0].cpu().numpy()

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
            # print(f'trajectory_idx: {trajectory_idx}')
            # pdb.set_trace()

        # random_input between -1 and 1 of shape trajectory_idx.shape
        # random_input = np.random.uniform(-1, 1, trajectory_idx.shape)
        # pdb.set_trace()
        # set trajectory_idx values to 0 if they are even and 1 if they are odd
        random_input = trajectory_idx % 2

        # make vector of zeros of shape trajectory_idx.shape by self.dbert
        conditioning_context = np.zeros((trajectory_idx.shape[0], self.dbert))
        for i in range(trajectory_idx.shape[0]):
            
            # text_1 = "Replace me by any text you'd like."
            if random_input[i] % 2 == 0:
                text_label = 'right'
            else:
                text_label = 'left'
            # if idx is not None:
            #     pdb.set_trace()
            conditioning_context[i] = self.context_lookup[text_label]
        # print("conditioning_context", conditioning_context)
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
