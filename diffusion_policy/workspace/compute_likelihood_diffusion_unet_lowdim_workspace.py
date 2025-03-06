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
import pickle

# import jax
# import jax.numpy as jnp
# import numpy as np
# from scipy import integrate

# def get_div_fn(fn):
#   """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

#   def div_fn(x, t, eps):
#     grad_fn = lambda data: jnp.sum(fn(data, t) * eps)
#     grad_fn_eps = jax.grad(grad_fn)(x)
#     return jnp.sum(grad_fn_eps * eps, axis=tuple(range(1, len(x.shape))))

#   return div_fn

# def batch_add(a, b):
#   return jax.vmap(lambda a, b: a + b)(a, b)


# def batch_mul(a, b):
#   return jax.vmap(lambda a, b: a * b)(a, b)

# def to_flattened_numpy(x):
#   """Flatten a JAX array `x` and convert it to numpy."""
#   return np.asarray(x.reshape((-1,)))


# def from_flattened_numpy(x, shape):
#   """Form a JAX array with the given `shape` from a flattened numpy array `x`."""
#   return jnp.asarray(x).reshape(shape)

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

        
    
    def generate_diverse_trajectories(self, ground_truth_actions: torch.Tensor, 
                                  num_trajectories: int = 10, 
                                  sim_hz: int = 240, control_hz: int = 30,
                                  k_p: float = 100, k_v: float =20,
                                  noise_scale: float = 0.1) -> torch.Tensor:
        """
        Generate K diverse but smooth trajectories using PD-controlled point mass dynamics,
        starting from the first ground truth state.

        Args:
            ground_truth_actions (torch.Tensor): Ground truth action sequence of shape (B, T, D).
            num_trajectories (int): Number of diverse trajectories to generate (K).
            sim_hz (int): Simulation frequency (e.g., 240 Hz).
            control_hz (int): Control frequency (e.g., 30 Hz).
            k_p (float): Proportional gain in PD control.
            k_v (float): Velocity damping gain in PD control.
            noise_scale (float): Noise scale for action perturbation.

        Returns:
            torch.Tensor: Generated trajectories of shape (B, K, T, D).
        """
        B, T, D = ground_truth_actions.shape  # Batch size, time steps, action dimension
        device = ground_truth_actions.device
        dt = 1.0 / sim_hz  # Simulation timestep
        n_steps = 8  # Number of physics steps per control step

        # Initialize output trajectories (B, K, T, D)
        diverse_trajectories = torch.zeros((B, num_trajectories, T, D), device=device)

        for k in range(num_trajectories):
            # Perturb the ground truth actions with Gaussian noise
            noise = torch.randn_like(ground_truth_actions) * noise_scale
            perturbed_actions = ground_truth_actions + noise

            # Simulate dynamics using PD control
            positions = torch.zeros((B, T, D), device=device)
            velocities = torch.zeros((B, T, D), device=device)

            # randomly sample initial velocity

            velocities[:, 0, :] = torch.randn_like(ground_truth_actions[:, 0, :]) * noise_scale 
            # add gausian noise to the initial velocity

            # add gasuian noise to the initial position
            positions[:, 0, :] = ground_truth_actions[:, 0, :] + torch.randn_like(ground_truth_actions[:, 0, :]) * noise_scale

            # Set initial position to match ground truth
            positions[:, 0, :] = ground_truth_actions[:, 0, :]

            for t in range(1, T):
                for _ in range(n_steps*4):
                    acceleration = (k_p * (perturbed_actions[:, t-1, :] - positions[:, t-1, :]) - k_v * velocities[:, t-1, :])
                    velocities[:, t-1, :] += acceleration * dt 
                    positions[:, t-1, :] += velocities[:, t-1, :] * dt

                    # add random noise to the velocity
                    # velocities[:, t-1, :] += torch.randn_like(velocities[:, t-1, :]) * noise_scale

                # Store computed state
                positions[:, t, :] = positions[:, t-1, :]
                velocities[:, t, :] = velocities[:, t-1, :] 

            # Store generated trajectory
            diverse_trajectories[:, k, :, :] = positions

        return diverse_trajectories

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.training.resume:
            # lastest_ckpt_path = self.get_checkpoint_path()
            # lastest_ckpt_path = 'data/outputs/2025.01.27/22.27.31_train_diffusion_unet_lowdim_push2d_lowdim/checkpoints/epoch=0150-test_mean_score=0.941.ckpt'
            lastest_ckpt_path = 'data/outputs/2025.02.07/11.11.21_train_diffusion_unet_lowdim_push2d_lowdim/checkpoints/latest.ckpt'
            
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
        idx_to_likelihood = {}
        self.model.eval()   
        for buffer_idx in range(dataset.replay_buffer.n_episodes):
            print(f"idx: {buffer_idx} of {dataset.replay_buffer.n_episodes}")
            episode_to_replay = dataset.replay_buffer.get_episode(buffer_idx)
            total_likelihood = 0

            # get index of episode start and ends
            demo_end_idx = dataset.replay_buffer.episode_ends[buffer_idx]
            if buffer_idx == 0:
                demo_start_idx = 0
            else:
                demo_start_idx = dataset.replay_buffer.episode_ends[buffer_idx-1]


            for t in range(demo_start_idx, demo_end_idx):
                print(f"timestep: {t}")
                # obs_dict = {'obs': episode_to_replay['obs'][t].unsqueeze(0),
                #             'action': episode_to_replay['action'][t].unsqueeze(0)
                #             }
                # index is the buffer idx repeated for the batch size
                # obs_dict['idx'] = torch.tensor([buffer_idx]).repeat(1)
                single_batch = dataset[t]
                obs_dict = {'obs': single_batch['obs'].unsqueeze(0),
                            'action': single_batch['action'].unsqueeze(0),
                            'idx': single_batch['idx'].unsqueeze(0),}
                

                # # pdb.set_trace()
                # obs_dict = {'obs': single_batch['obs'].unsqueeze(0)}
                # gt_action = single_batch['action'].unsqueeze(0)
                # # gt_action = episode_to_replay['action'][t].unsqueeze(0)

                # # move data to device
                obs_dict = dict_apply(obs_dict, lambda x: x.to(device))
                # gt_action = gt_action.to(device)
                            
                # result = self.model.predict_action(obs_dict)
                # if cfg.pred_action_steps_only:
                #     pred_action = result['action']
                #     start = cfg.n_obs_steps - 1
                #     end = start + cfg.n_action_steps
                #     gt_action = gt_action[:,start:end]
                # else:
                #     pred_action = result['action_pred']

                # move both to device
                # pred_action = pred_action.to(device)
                # gt_action = gt_action.to(device)
                # mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                # print(f"mse: {mse}")
                # unsqueeze for all keys in single_batch
                # for keyname in single_batch.keys():
                #     single_batch[keyname] = single_batch[keyname].unsqueeze(0)

                # score = self.model.get_score(single_batch, 1)
                # pdb.set_trace()
                likelihood = self.model.get_log_likelihood(obs_dict)
                likelihood = torch.sum(likelihood).item()
                print(f"likelihood: {likelihood}")
                # pdb.set_trace()
                total_likelihood += likelihood

            idx_to_likelihood[buffer_idx] = total_likelihood
            # save the likelihoods to a file
            if buffer_idx % 2 == 0:
                with open('likelihoods.pkl', 'wb') as f:
                    pickle.dump(idx_to_likelihood, f)
                print(f"saved likelihoods to file")
            # print("score: ", score)
            # print("log_likelihood: ", log_likelihood)
            # pdb.set_trace()
            # diverse_trajectories = self.generate_diverse_trajectories(gt_action, num_trajectories=10, noise_scale=10)
            # plot in matplotlib the ground truth action and the diverse trajectories
            # plot the ground truth action
            # plot the diverse trajectories
            # for i in range(gt_action.shape[1]):
            #     plt.scatter(gt_action[0, i], 0, color='red')

            # move the diverse trajectories to cpu and plot them

            # gt_action = gt_action.cpu().detach().numpy()

            # plt.plot(gt_action[0,:,0], gt_action[0,:,1], label='ground truth action', color='black')
            # for i in range(diverse_trajectories.shape[1]):
            #     data_to_plot = diverse_trajectories[:,i,:,:].cpu().detach().numpy()
            #     plt.plot(data_to_plot[0,:,0], data_to_plot[0,:,1], label=f'diverse trajectory {i}')
            #     # plt.plot(diverse_trajectories[i][0,:,0],diverse_trajectories[i][0,:,1], label=f'diverse trajectory {i}')
            # plt.legend()
            # plt.title(f'GT and samples for timestep {idx}')
            # plt.savefig(f'distribution_viz/gt_samples_timestep_{idx}.png')
            # plt.close()
            # likelihood_list = {0: likelihood}
            # for i in range(diverse_trajectories.shape[1]):
            #     single_batch['action'] = diverse_trajectories[:,i,:,:]
            #     likelihood = self.model.get_log_likelihood(single_batch)
            #     likelihood_list[i+1] = likelihood
            # plot the likelihood
            # print(likelihood_list)


            # for key in likelihood_list.keys():
            #     likelihood = likelihood_list[key]
            #     likelihood = likelihood.cpu().detach().numpy()
            #     # take the product of the likelihood
            #     total_likelihood = np.sum(likelihood)
            #     likelihood_list[key] = total_likelihood

            

            # plt.bar(likelihood_list.keys(), likelihood_list.values())
            # plt.title(f'Log likelihoods for timestep {idx}')
            # plt.savefig(f'distribution_viz/log_probs_timestep_{idx}.png')
            # plt.close()
            # # pdb.set_trace()
            # log_likelihoods = torch.tensor(list(likelihood_list.values()))
            # probabilities = torch.nn.functional.softmax(log_likelihoods, dim=0)
            # # print(probabilities)
                
            # # normalize
            
            # for key in likelihood_list.keys():
            #     likelihood_list[key] = probabilities[key].item()

            # # print(likelihood_list)
            # # plot the likelihood
            # plt.bar(likelihood_list.keys(), likelihood_list.values())
            # plt.title(f'likelihoods for timestep {idx}')
            # plt.savefig(f'distribution_viz/normalized_probs_timestep_{idx}.png')
            # plt.close()
            # plt.show()



            


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetLowdimWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
