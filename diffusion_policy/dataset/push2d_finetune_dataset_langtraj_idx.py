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
            train_zarr_path, 
            target_zarr_path,
            use_target=False,
            horizon=1,
            pad_before=0,
            pad_after=0,
            obs_key='keypoint',
            state_key='state',
            action_key='action',
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            utterance_input=None,
            ):
        if use_target:
            zarr_path = target_zarr_path
        else:
            zarr_path = train_zarr_path
        # print("\n\nzarr path", zarr_path)
        super().__init__()

        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=[obs_key, state_key, action_key])
        
        self.utterance_input = utterance_input

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


        self.dtraj = 256
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.context_encoder = BertModel.from_pretrained("bert-base-uncased",output_hidden_states = True)
        self.dbert = 768
        self.set_context_lookup()
        if self.utterance_input is not None:
            self.set_utterance_lookup()


    def set_context_lookup(self):
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

    def set_utterance_lookup(self):
        for idx in self.utterance_input:
            text_label = self.utterance_input[idx]
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
        
        
        # get conditioning context for demo idx
        if 'idx' not in sample:
            trajectory_idx = self.replay_buffer.get_episode_idxs()
        else:
            trajectory_idx = sample['idx']

        conditioning_context = np.zeros((trajectory_idx.shape[0], self.dtraj))
        for i in range(trajectory_idx.shape[0]):
            conditioning_context[i] = np.ones(self.dtraj) * trajectory_idx[i]

        # get language conditioning for demo idx
        evenodd_input = trajectory_idx % 2
        lang_conditioning_context = np.zeros((trajectory_idx.shape[0], self.dbert))
        for i in range(trajectory_idx.shape[0]):
            # pdb.set_trace()
            if evenodd_input[i] % 2 == 0:
                text_label = 'right'
            else:
                text_label = 'left'

            if self.utterance_input is not None:
                # pdb.set_trace()
                # print("utterance input", self.utterance_input)
                # print("trajectory idx", trajectory_idx[i])
                # print("self.context_lookup", self.context_lookup.keys())
                # print("type self.utterance_input[trajectory_idx[i]]", type(self.utterance_input[trajectory_idx[i]]))
                lang_conditioning_context[i] = self.context_lookup[self.utterance_input[int(trajectory_idx[i])]]
            else:
                lang_conditioning_context[i] = self.context_lookup[text_label]

        data = {
            'obs': obs, # T, D_o
            'action': sample[self.action_key], # T, D_a
            # 'idx': conditioning_context, # scalar
            'utterance': lang_conditioning_context
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
