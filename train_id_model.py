"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from torch.utils.data import DataLoader
import pdb
import pickle
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from omegaconf import DictConfig
OmegaConf.register_new_resolver("eval", eval, replace=True)

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
def main(checkpoint, output_dir, device):
    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    # pdb.set_trace()
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # set to original checkpoint
    original_ckpt = 'data/outputs/2025.02.18/17.48.31_train_diffusion_unet_lowdim_push2d_lowdim/checkpoints/latest.ckpt'
    original_ckpt = 'data/outputs/2025.02.21/03.01.50_train_diffusion_unet_lowdim_push2d_lowdim/checkpoints/latest.ckpt'
    
    payload = torch.load(open(original_ckpt, 'rb'), pickle_module=dill)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    original_cfg = payload['cfg']

    # configure Training Demos dataset
    demos_dataset: BaseLowdimDataset
    cfg.task.dataset.use_target = False
    demos_dataset = hydra.utils.instantiate(original_cfg.task.dataset)
    train_dataloader = DataLoader(demos_dataset, **original_cfg.dataloader)
    print("len(train_dataset): ", len(demos_dataset))
    print("num eps", demos_dataset.replay_buffer.n_episodes)
    print("fg.task.dataset.train_zarr_path", original_cfg.task.dataset.zarr_path)


    # configure Target Demos dataset
    target_dataset: BaseLowdimDataset
    cfg.task.dataset.use_target = True

    # load target utterance_dict
    filename_for_target_lang = cfg.task.dataset.target_zarr_path.split('.zarr')[0] + '_lang.pkl'
    with open(filename_for_target_lang, 'rb') as f:
        target_utterance_dict = pickle.load(f)

    cfg.task.dataset.utterance_input = target_utterance_dict

    target_dataset = hydra.utils.instantiate(cfg.task.dataset)
    print("len(target_dataset): ", len(target_dataset))
    # dataset.set_device(device)
    assert isinstance(target_dataset, BaseLowdimDataset)
    target_dataloader = DataLoader(target_dataset, **cfg.dataloader)
    # target_normalizer = target_dataset.get_normalizer() We will not use target normalizer

    demo_idx = 0
    demo_end_idx = target_dataset.replay_buffer.episode_ends[demo_idx]
    demo_data = target_dataset.replay_buffer.get_episode(0) # keys are keypoint, state, action

    # configure validation dataset
    val_dataset = target_dataset.get_validation_dataset()
    val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

    class InverseDynamicsModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Sequential(
                torch.nn.Linear(80, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 16)
            )

        def forward(self, x):
            return self.model(x)
        
    # create model
    inv_model = InverseDynamicsModel()
    inv_model.to(device)
    optimizer = torch.optim.Adam(inv_model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    total_loss = []
    for ep in range(20):
        # train inverse dynamics model
        total_val = 0
        for t in range(len(demos_dataset)-8):
            # get demo data
            demo_data = demos_dataset[t]['obs'][:cfg.n_obs_steps,:] # 2, 20
            # next timestep 
            demo_data_t1 = demos_dataset[t+8]['obs'][:cfg.n_obs_steps,:] # 2, 20
            # action
            demo_action = demos_dataset[t]['action'][:cfg.n_action_steps,cfg.n_latency_steps:] # 8, 2

            input_x = torch.cat([demo_data, demo_data_t1], dim=0)
            input_x_flat = input_x.flatten() # 80
            target_y = demo_action.flatten() # 16

            # train model
            optimizer.zero_grad()
            pred_y = inv_model(input_x_flat.to(device))
            loss = loss_fn(pred_y, target_y.to(device))
            loss.backward()
            optimizer.step()
            total_val += loss.item()



            # if ep % 100 == 0:
            print("ep", ep, "loss", loss.item())
        total_loss.append(total_val)
    # save model
    torch.save(inv_model.state_dict(), os.path.join(output_dir, 'inv_model.pth'))

    plt.plot(total_loss)
    plt.show()

    # evaluate model on target dataset
    inv_model.eval()
    inv_model.load_state_dict(torch.load(os.path.join(output_dir, 'inv_model.pth')))
    inv_model.to(device)
    t_to_loss = {}
    for t in range(len(target_dataset)-8):
        # get demo data
        demo_data = target_dataset[t]['obs'][:cfg.n_obs_steps,:]
        # next timestep
        demo_data_t1 = target_dataset[t+8]['obs'][:cfg.n_obs_steps,:]
        # action
        demo_action = target_dataset[t]['action'][:cfg.n_action_steps,cfg.n_latency_steps:]
        # input
        input_x = torch.cat([demo_data, demo_data_t1], dim=0)
        input_x_flat = input_x.flatten()
        target_y = demo_action.flatten()
        # predict
        pred_y = inv_model(input_x_flat.to(device))
        loss = loss_fn(pred_y, target_y.to(device))
        print("t", t, "loss", loss.item())
        t_to_loss[t] = loss.item()
    
    # plot loss over time
    plt.plot(t_to_loss.keys(), t_to_loss.values())
    plt.show()

    # print the time step with the highest loss
    sorted_t = sorted(t_to_loss.items(), key=lambda x: x[1], reverse=True)
    print("sorted_t", sorted_t)
    print("highest loss", sorted_t[0])
    print("lowest loss", sorted_t[-1])
    print("t_to_loss", t_to_loss)



if __name__ == '__main__':
    main()
    # main_mixed_policy()
