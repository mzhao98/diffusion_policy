if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import numpy as np
import random
import wandb
import tqdm
import shutil

from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusers.training_utils import EMAModel
import pdb
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

OmegaConf.register_new_resolver("eval", eval, replace=True)

# %%
class TrainDiffusionUnetLowdimWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetLowdimPolicy
        self.model = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetLowdimPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.training.resume:
            # lastest_ckpt_path = self.get_checkpoint_path()
            lastest_ckpt_path = 'data/outputs/2025.01.27/22.27.31_train_diffusion_unet_lowdim_push2d_lowdim/checkpoints/epoch=0150-test_mean_score=0.941.ckpt'
            # if lastest_ckpt_path.is_file():
            print(f"Resuming from checkpoint {lastest_ckpt_path}")
            self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseLowdimDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseLowdimDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()
        # pdb.set_trace()

        demo_idx = 0
        demo_end_idx = dataset.replay_buffer.episode_ends[demo_idx]
        demo_data = dataset.replay_buffer.get_episode(0) # keys are keypoint, state, action
        # create a batch of data from 0 to demo_end_idx


        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr schedulerfrom scipy.stats import gaussian_kde
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env runner
        env_runner: BaseLowdimRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseLowdimRunner)


        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # for index from 0 to demo_end_idx
        # get the batch from the dataset
        # get the observation from the batch
        # get the action from the batch
        # get the prediction from the model
        # get the mse between the prediction and the action
        self.model.eval()   
        for i in range(demo_end_idx):
            single_batch = dataset[i]
            # pdb.set_trace()
            obs_dict = {'obs': single_batch['obs'].unsqueeze(0)}
            gt_action = single_batch['action'].unsqueeze(0)

            # move data to device
            obs_dict = dict_apply(obs_dict, lambda x: x.to(device))
            gt_action = gt_action.to(device)
            
            # Sample K= 30 predictions
            K = 30
            sampled_predictions = {}
            for k in range(K):
                result = self.model.predict_action(obs_dict)
                pred_action = result['action'][0]
                # flatten
                for t in range(pred_action.shape[0]):
                    pred_act_t = pred_action[t]
                    if t not in sampled_predictions:
                        sampled_predictions[t] = []
                # flatten data
                # flatten_pred_action = pred_action.view(-1)
                    flatten_pred_action = pred_act_t.view(-1)
                    sampled_predictions[t].append(flatten_pred_action.detach().cpu().numpy())

            # fit a kde model on the sampled predictions
            # pdb.set_trace()
            # sampled_predictions = np.array(sampled_predictions)
            # kde = gaussian_kde(sampled_predictions.T)

            # get the likelihood of the ground truth action
            # flatten gt_action[0]
            start = cfg.n_obs_steps - 1
            end = start + cfg.n_action_steps
            gt_action = gt_action[:,start:end]

            for t in sampled_predictions:
                sampled_predictions[t] = np.array(sampled_predictions[t])
                kde = gaussian_kde(sampled_predictions[t].T)
                gt_action_t = gt_action[0,t]
                gt_action_t = gt_action_t.view(-1).detach().cpu().numpy()
                likelihood = kde(gt_action_t)
                print(f"likelihood of gt: {likelihood}")

                # get likelihood of predictions
                for sample in sampled_predictions[t]:
                    likelihood_sample = kde(sample)
                    print(f"likelihood of pred: {likelihood_sample}")
                
                # plot density manifold
                plot_density_manifold(kde, sampled_predictions[t], gt_action_t, i, t, grid_size=100, cmap='coolwarm')
                # pdb.set_trace()
        
            gt_action = single_batch['action'].unsqueeze(0)
                        
            result = self.model.predict_action(obs_dict)
            if cfg.pred_action_steps_only:
                pred_action = result['action']
                start = cfg.n_obs_steps - 1
                end = start + cfg.n_action_steps
                gt_action = gt_action[:,start:end]
            else:
                pred_action = result['action_pred']

            # move both to device
            pred_action = pred_action.to(device)
            gt_action = gt_action.to(device)
            mse = torch.nn.functional.mse_loss(pred_action, gt_action)
            print(f"mse: {mse}")
            # pdb.set_trace()

    


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetLowdimWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()


def plot_density_manifold(kde, sampled_predictions, gt_action_t, ep_idx, subtimestep, grid_size=100, cmap='coolwarm'):
    
    """
    Plots a density manifold from a given Nx2 list of points.

    Parameters:
        data (list of list): An Nx2 list of points (x, y).
        grid_size (int): The resolution of the density grid. Default is 100.
        cmap (str): The colormap for the plot. Default is 'coolwarm'.
    """
    

    # Create a grid for density evaluation
    # grid_x, grid_y = np.meshgrid(
    #     np.linspace(min(x), max(x), grid_size),
    #     np.linspace(min(y), max(y), grid_size)
    # )
    # create of grid of 450x450
    grid_x, grid_y = np.meshgrid(
        np.linspace(0, 450, grid_size),
        np.linspace(0, 450, grid_size)
    )

    

    grid_density = kde(np.vstack([grid_x.ravel(), grid_y.ravel()]))
    grid_density = grid_density.reshape(grid_x.shape)

    # Plot the density manifold
    plt.figure(figsize=(8, 6))
    plt.xlim(0,450)
    plt.ylim(0,450)
    plt.contourf(grid_x, grid_y, grid_density, levels=50, cmap=cmap)  # Blue to red colormap
    plt.colorbar(label='Density')
    # plot sampled predictions
    # pdb.set_trace()
    for j in range(sampled_predictions.shape[0]):
        plt.scatter(sampled_predictions[j][0], sampled_predictions[j][1], s=50, alpha=0.5)

    # plot gt action with a black star
    plt.scatter(gt_action_t[0], gt_action_t[1], s=100, c='black', marker='*')

    plt.legend()
    plt.title(f"Density Manifold Plot:")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.savefig(f'distribution_viz/density_manifold_{ep_idx}_subt_{subtimestep}.png')
    plt.show()
    # return kde

