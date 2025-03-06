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
from omegaconf import OmegaConf
from omegaconf import DictConfig
OmegaConf.register_new_resolver("eval", eval, replace=True)

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
def main(checkpoint, output_dir, device):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
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

    


    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    # policy.eval()

    np.random.seed(0)
    torch.manual_seed(0)

    cfg.task.env_runner.n_train = 0
    cfg.task.env_runner.n_test = 1
    cfg.task.env_runner.n_test_vis = 1
    cfg.task.env_runner.max_steps = 400
    # pdb.set_trace()
    cfg.task.env_runner._target_ = 'diffusion_policy.env_runner.push2d_keypoints_runner_eval_w_langtraj.Push2dKeypointsRunner'
    
    # run eval
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir)
    
    # set replay buffer
    env_runner.set_target_replay_buffer(target_dataset, target_dataloader)
    env_runner.set_train_replay_buffer(demos_dataset, train_dataloader)

    utterance_to_use = target_utterance_dict[demo_idx]
    # utterance_to_use = "right"
    # runner_log, total_divergence = env_runner.run_with_target(policy, utterance_to_use)
    # print("total_divergence", total_divergence)
    # runner_log = env_runner.run(policy, utterance_to_use)
    runner_log = env_runner.run_w_context_opt(policy, utterance_to_use)
    # runner_log = env_runner.run_w_lang_opt(policy, utterance_to_use, 0)
    
    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)


# @click.command()
# @click.option('-c', '--checkpoint', required=True)
# @click.option('-o', '--output_dir', required=True)
# @click.option('-d', '--device', default='cuda:0')
@hydra.main(
    version_base=None,
    config_path='diffusion_policy/config',
    config_name='finetune_diffusion_unet_lowdim_workspace.yaml'
)
def main_mixed_policy(cfg: DictConfig):
    OmegaConf.resolve(cfg)

    checkpoint = 'data/outputs/2025.02.18/22.45.32_finetune_diffusion_unet_lowdim_push2d_lowdim/checkpoints/epoch=0750-test_mean_score=0.000.ckpt'
    checkpoint = 'data/outputs/2025.02.20/22.11.40_finetune_diffusion_unet_lowdim_push2d_lowdim/checkpoints/latest.ckpt'
    checkpoint = 'data/outputs/2025.02.20/22.58.43_finetune_diffusion_unet_lowdim_push2d_lowdim/checkpoints/latest.ckpt'
    
    output_dir = 'data/compare_langtraj_open_rollout'
    device = 'cuda:0'
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    # cfg = payload['cfg']
    # pdb.set_trace()
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # set to original checkpoint
    
    original_ckpt = 'data/outputs/2025.02.18/17.48.31_train_diffusion_unet_lowdim_push2d_lowdim/checkpoints/latest.ckpt'
    original_payload = torch.load(open(original_ckpt, 'rb'), pickle_module=dill)
    # original_cfg = original_payload['cfg']
    original_cls = hydra.utils.get_class(cfg._target_)
    original_workspace: BaseWorkspace
    original_workspace = original_cls(cfg, output_dir=output_dir)
    original_workspace.load_payload(original_payload, exclude_keys=None, include_keys=None)

    # configure Training Demos dataset
    demos_dataset: BaseLowdimDataset
    cfg.task.dataset.use_target = False
    demos_dataset = hydra.utils.instantiate(cfg.task.dataset)
    train_dataloader = DataLoader(demos_dataset, **cfg.dataloader)
    print("len(train_dataset): ", len(demos_dataset))


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

    


    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)


    # get original policy from workspace
    original_policy = original_workspace.model
    if cfg.training.use_ema:
        original_policy = original_workspace.ema_model
    
    original_policy.to(device)

    # policy.eval()

    np.random.seed(0)
    torch.manual_seed(0)

    cfg.task.env_runner.n_train = 0
    cfg.task.env_runner.n_test = 1
    cfg.task.env_runner.n_test_vis = 1
    cfg.task.env_runner.max_steps = 400
    # pdb.set_trace()
    cfg.task.env_runner._target_ = 'diffusion_policy.env_runner.push2d_keypoints_runner_eval_w_langtraj.Push2dKeypointsRunner'
    
    # run eval
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir)
    
    # set replay buffer
    env_runner.set_target_replay_buffer(target_dataset, target_dataloader)
    env_runner.set_train_replay_buffer(demos_dataset, train_dataloader)

    utterance_to_use = target_utterance_dict[demo_idx]
    # runner_log, total_divergence = env_runner.run_with_target(policy, utterance_to_use)
    # print("total_divergence", total_divergence)
    # runner_log = env_runner.run(policy, utterance_to_use)
    runner_log = env_runner.run_mixed_policy(policy, original_policy, utterance_to_use)
    # runner_log = env_runner.run_w_context_opt(policy, utterance_to_use)
    # runner_log = env_runner.run_w_lang_opt(policy, utterance_to_use, 0)
    
    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    main()
    # main_mixed_policy()
