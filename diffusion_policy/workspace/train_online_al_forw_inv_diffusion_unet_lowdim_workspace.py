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
import matplotlib.pyplot as plt
import pickle
from diffusion_policy.env.push2d.push2d_keypoints_env import Push2dKeypointsEnv
from diffusion_policy.env.push2d.push2d_env import Push2dEnv
import cv2
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
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseLowdimDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseLowdimDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
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

        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # compute loss
                        raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        
                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break
                
                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0:
                    runner_log = env_runner.run(policy)
                    # log all
                    step_log.update(runner_log)

                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss = self.model.compute_loss(batch)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = train_sampling_batch
                        obs_dict = {'obs': batch['obs']}
                        gt_action = batch['action']
                        
                        result = policy.predict_action(obs_dict)
                        if cfg.pred_action_steps_only:
                            pred_action = result['action']
                            start = cfg.n_obs_steps - 1
                            end = start + cfg.n_action_steps
                            gt_action = gt_action[:,start:end]
                        else:
                            pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        # log
                        step_log['train_action_mse_error'] = mse.item()
                        # release RAM
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse
                
                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

    def setup_curiosity_model(self):
        class CuriosityModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.obs_encoding_dim = 36
                self.single_obs_dim = 20
                self.action_dim = 16
                self.obs_encoder = torch.nn.Sequential(
                    torch.nn.Linear(self.single_obs_dim, 64),
                    torch.nn.ReLU(),
                    torch.nn.Linear(64, 32),
                    torch.nn.ReLU(),
                    torch.nn.Linear(32, self.obs_encoding_dim),
                )

                self.forward_model = torch.nn.Sequential(
                    torch.nn.Linear(self.obs_encoding_dim + self.action_dim, 72),
                    torch.nn.ReLU(),
                    torch.nn.Linear(72, 128),
                    torch.nn.ReLU(),
                    torch.nn.Linear(128, 64),
                    torch.nn.ReLU(),
                    torch.nn.Linear(64, self.obs_encoding_dim),
                )

                self.inverse_model = torch.nn.Sequential(
                    torch.nn.Linear(self.obs_encoding_dim * 2, 72),
                    torch.nn.ReLU(),
                    torch.nn.Linear(72, 128),
                    torch.nn.ReLU(),
                    torch.nn.Linear(128, 64),
                    torch.nn.ReLU(),
                    torch.nn.Linear(64, 32),
                    torch.nn.ReLU(),
                    torch.nn.Linear(32, self.action_dim),
                )


            def forward(self, obs_t, obs_t1, action):
                obs_t_embed  = self.obs_encoder(obs_t)
                obs_t1_embed = self.obs_encoder(obs_t1)
                obs_action = torch.cat([obs_t_embed, action], dim=1)
                pred_obs_t1_embed = self.forward_model(obs_action)

                two_obs_embed = torch.cat([obs_t_embed, obs_t1_embed], dim=1)
                pred_action = self.inverse_model(two_obs_embed)

                return pred_action, pred_obs_t1_embed
            
            def forward_dynamics(self, obs_t, action):
                obs_t_embed = self.obs_encoder(obs_t)
                # print shapes
                # print("obs_t_embed", obs_t_embed.shape)
                # print("action", action.shape)
                obs_action = torch.cat([obs_t_embed, action], dim=1)
                pred_obs_t1_embed = self.forward_model(obs_action)
                # print("pred_obs_t1_embed", pred_obs_t1_embed.shape)
                return pred_obs_t1_embed
            
            def inverse_dynamics(self, obs_t, obs_t1):
                obs_t_embed = self.obs_encoder(obs_t)
                obs_t1_embed = self.obs_encoder(obs_t1)
                # print shapes
                # print("obs_t_embed", obs_t_embed.shape)
                # print("obs_t1_embed", obs_t1_embed.shape)
                two_obs_embed = torch.cat([obs_t_embed, obs_t1_embed], dim=1)
                pred_action = self.inverse_model(two_obs_embed)
                # print("pred_action", pred_action.shape)

                return pred_action
            
            def encode_obs(self, obs):
                return self.obs_encoder(obs)

        
        # create model
        curiosity_model = CuriosityModel()
        self.curiosity_model = curiosity_model


    def visualize_top_obs(self, env_runner, top_obs, top_obs_dicts, save_tag_folder):
        
        obs_to_label = {'recover': [], 'avoid': []}
        for obs_idx in range(len(top_obs)):
            # vis_env.reset()
            vis_env = Push2dEnv()

            agent_pos = top_obs_dicts[obs_idx]['pos_agent'][1]
            block_pos = top_obs_dicts[obs_idx]['block_pose'][1]
            block_angle = top_obs_dicts[obs_idx]['block_angle'][1]
            goal_pos1 = [top_obs_dicts[obs_idx]['goal_pose1'][0][0], top_obs_dicts[obs_idx]['goal_pose1'][0][1]]
            goal_pos2 = [top_obs_dicts[obs_idx]['goal_pose2'][0][0], top_obs_dicts[obs_idx]['goal_pose2'][0][1]]

            vis_env.reset_with_input(agent_pos, block_pos, block_angle, goal_pos1, goal_pos2)
            vis_env.step(None)
            # obs_t = top_obs[obs_idx]
            # vis_env.block.position = tuple(obs_t['block_pose'][0])
            # vis_env.block.angle = obs_t['block_angle'][0]
            # vis_env.agent.position = tuple(obs_t['pos_agent'][0])
            # vis_env.agent.velocity = tuple(obs_t['vel_agent'][0])
            # step env
            # vis_env.space.step(1.0 / vis_env.sim_hz)
            img = vis_env._render_frame("human")
            # pdb.set_trace()
            # save image to save_tag_folder
            img_path = save_tag_folder + f'/top_obs_{obs_idx}.png'
            cv2.imwrite(img_path, img)

            # plot image 
            # plt.imshow(img)
            # plt.show()

        
            # close env
            vis_env.close()
        print("visualized top obs")
        # save obs_to_label to folder
        with open(f'{save_tag_folder}/obs_to_label.pkl', 'wb') as f:
            pickle.dump(obs_to_label, f)

        return obs_to_label


    def save_added_trajs_to_video(self, env_runner, top_episode_idxs, save_tag_folder, dataset):
        # create video
        render_size = 96
        kp_kwargs = Push2dKeypointsEnv.genenerate_keypoint_manager_params()
        env = Push2dKeypointsEnv(render_size=render_size, render_action=False, **kp_kwargs)
        for ep_idx in top_episode_idxs:
            
            # get episode
            episode_to_replay = dataset.original_replay_buffer.get_episode(ep_idx)
            # pdb.set_trace()
            # seed = episode_to_replay['seed']
            
            # reset env and get observations (including info and render for recording)
            obs = env.reset()
            info = env._get_info()
            state_to_replay = episode_to_replay['state'][0]
            # remove the last element of the state_to_replay
            state_to_replay = state_to_replay[:-1]
            # remove third to last element of the state_to_replay
            state_to_replay = list(state_to_replay[:-3]) + list(state_to_replay[-2:])
            env._set_state(state_to_replay)
            img = env.render(mode='human')
            list_of_imgs = [img]

            # step-level while loop
            for t in range(1, len(episode_to_replay['action'])):
                
                state_to_replay = episode_to_replay['state'][t]
                # remove the last element of the state_to_replay
                state_to_replay = state_to_replay[:-1]
                # remove third to last element of the state_to_replay
                state_to_replay = list(state_to_replay[:-3]) + list(state_to_replay[-2:])
                env._set_state(state_to_replay)
                img = env.render(mode='human')
                list_of_imgs.append(img)

            # save to video with fps 5
            video_save_path = save_tag_folder + f'/added_trajs_{ep_idx}.mp4'
            # save list of images to video with cv2
            out = cv2.VideoWriter(video_save_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, (render_size, render_size))
            for img in list_of_imgs:
                out.write(img)
            out.release()
            

        env.close()
        print("saved added trajs to video")


    def train_curiosity_model(self, model, dataset, recovery_dataset, rollout_data, cfg, active_learning_round, device, n_epochs_inv_dyn):
        curiosity_train_losses = []
        dataset_x = []
        dataset_y = []
        state_dim = 20
        beta = 0.7

        # Add original dataset to dataset_x and dataset_y
        counter = 0
        n_episodes = dataset.replay_buffer.n_episodes
        for ep in range(n_episodes):
            if ep == 0:
                start_idx = 0
            else:
                start_idx = dataset.replay_buffer.episode_ends[ep-1]
            end_idx = dataset.replay_buffer.episode_ends[ep]-16
            # print("ep", ep, "start_idx", start_idx, "end_idx", end_idx)
            # pdb.set_trace()
            for c in range(start_idx, end_idx):
                t= counter
                # normalize observation
                obs = dataset[t]['obs'][:cfg.n_obs_steps,:]
                obs_normalized = model.normalizer['obs'].normalize(obs)
                obs_t8 = dataset[t+8]['obs'][:cfg.n_obs_steps,:]
                obs_t8_normalized = model.normalizer['obs'].normalize(obs_t8)
                # action normalized
                action_t = dataset[t]['action']
                action_t_normalized = model.normalizer['action'].normalize(action_t)


                # get demo data
                demo_data = obs_normalized[:cfg.n_obs_steps,:][1] # 1, 20
                # next timestep 
                demo_data_t1 = obs_t8_normalized[:cfg.n_obs_steps,:][1] # 1, 20
                # action
                demo_action = action_t_normalized[:cfg.n_action_steps,cfg.n_latency_steps:] # 8, 2
                counter += 1
                

                input_x = torch.cat([demo_data, demo_data_t1], dim=0)
                input_x_flat = input_x.flatten() # 80
                target_y = demo_action.flatten() # 16

                # move both to device
                # demo_data = demo_data.to(device)
                # demo_data_t1 = demo_data_t1.to(device)
                input_x_flat = input_x_flat.to(device)
                target_y = target_y.to(device)

                # add to dataset
                dataset_x.append(input_x_flat)
                dataset_y.append(target_y)

        # Add expert recovery dataset to dataset_x and dataset_y
        if recovery_dataset is not None:
            counter = 0
            n_episodes = recovery_dataset.replay_buffer.n_episodes
            for ep in range(n_episodes):
                if ep == 0:
                    start_idx = 0
                else:
                    start_idx = recovery_dataset.replay_buffer.episode_ends[ep-1]
                end_idx = recovery_dataset.replay_buffer.episode_ends[ep]-16
                # print("ep", ep, "start_idx", start_idx, "end_idx", end_idx)
                # pdb.set_trace()
                for c in range(start_idx, end_idx):
                    t= counter
                    # get demo data
                    obs = recovery_dataset[t]['obs'][:cfg.n_obs_steps,:]
                    obs_normalized = model.normalizer['obs'].normalize(obs)
                    obs_t8 = recovery_dataset[t+8]['obs'][:cfg.n_obs_steps,:]
                    obs_t8_normalized = model.normalizer['obs'].normalize(obs_t8)
                    # action normalized
                    action_t = recovery_dataset[t]['action']
                    action_t_normalized = model.normalizer['action'].normalize(action_t)


                    # get demo data
                    demo_data = obs_normalized[:cfg.n_obs_steps,:][1] # 1, 20
                    # next timestep 
                    demo_data_t1 = obs_t8_normalized[:cfg.n_obs_steps,:][1] # 1, 20
                    # action
                    demo_action = action_t_normalized[:cfg.n_action_steps,cfg.n_latency_steps:] # 8, 2
                    counter += 1
                    # print("counter", counter)

                    input_x = torch.cat([demo_data, demo_data_t1], dim=0)
                    input_x_flat = input_x.flatten() # 80
                    target_y = demo_action.flatten() # 16

                    # move both to device
                    # demo_data = demo_data.to(device)
                    # demo_data_t1 = demo_data_t1.to(device)
                    input_x_flat = input_x_flat.to(device)
                    target_y = target_y.to(device)

                    # add to dataset
                    dataset_x.append(input_x_flat)
                    dataset_y.append(target_y)

        for rollout_data_idx in range(len(rollout_data)):
            obs_t, obs_t1, action = rollout_data[rollout_data_idx]
            # normalize
            obs_t_normalized = model.normalizer['obs'].normalize(obs_t)
            obs_t1_normalized = model.normalizer['obs'].normalize(obs_t1)
            action_normalized = model.normalizer['action'].normalize(action)

            # get demo data
            # demo_data = obs_t_normalized[:cfg.n_obs_steps,:][1] # 1, 20
            # # next timestep
            # demo_data_t1 = obs_t1_normalized[:cfg.n_obs_steps,:][1] # 1, 20
            # # action
            # demo_action = action_normalized[:cfg.n_action_steps,cfg.n_latency_steps:] # 8, 2
            demo_data = obs_t_normalized[0][1]
            demo_data_t1 = obs_t1_normalized[1]
            demo_action = action_normalized[0]

            # unsqueeze obs_t and obs_t1
            # demo_data = demo_data.unsqueeze(0)
            # demo_data_t1 = demo_data_t1.unsqueeze(0)

            # pdb.set_trace()

            input_x = torch.cat([demo_data, demo_data_t1], dim=0)
            input_x_flat = input_x # 80
            target_y = demo_action.flatten() # 16

            # move both to device
            input_x_flat = input_x_flat.to(device)
            target_y = target_y.to(device)

            # add to dataset
            dataset_x.append(input_x_flat)
            dataset_y.append(target_y)
        
        # create inv_dyn dataset
        dataset_x = torch.stack(dataset_x)
        dataset_y = torch.stack(dataset_y)
        inv_dyn_dataset = torch.utils.data.TensorDataset(dataset_x, dataset_y)
        inv_dyn_dataloader = DataLoader(inv_dyn_dataset, batch_size=256, shuffle=True)

        for local_epoch_idx in range(n_epochs_inv_dyn):

            # print("training inverse dynamics model")
            # move model to device
            self.curiosity_model.to(device)
            
            total_val = []
            
            for batch_idx, batch in enumerate(inv_dyn_dataloader):
                # device transfer
                input_x_flat = batch[0].to(device, non_blocking=True)
                target_action = batch[1].to(device, non_blocking=True)

                # split input_x_flat into obs_t and obs_t1
                input_x_obs_t = input_x_flat[:,:state_dim]
                input_x_obs_t1 = input_x_flat[:,state_dim:]
            
                # train model
                self.curiosity_optimizer.zero_grad()
                pred_action, pred_obs_t1_embed = self.curiosity_model(input_x_obs_t, input_x_obs_t1, target_action)
                
                # compute loss
                # encode obs_t1
                obs_t1_embed = self.curiosity_model.encode_obs(input_x_obs_t1)
                mse_on_obs = self.curiosity_loss_fn(pred_obs_t1_embed, obs_t1_embed)

                # use model to unnormalize
                # pred_action = model.normalizer['action'].unnormalize(pred_action)
                # target_action = model.normalizer['action'].unnormalize(target_action)


                mse_on_action = self.curiosity_loss_fn(pred_action, target_action)
                loss = beta * mse_on_action + (1-beta) * mse_on_obs
                # if local_epoch_idx == n_epochs_inv_dyn-1:
                #     pdb.set_trace()

                loss.backward()
                self.curiosity_optimizer.step()
                total_val.append(loss.item())

            # save model
            if local_epoch_idx % 100 == 0:
                print("local_epoch_idx", local_epoch_idx, "loss", np.mean(total_val))
            # print("total_loss", np.mean(total_val))
            curiosity_train_losses.append(np.mean(total_val))

        # plot inv_train_losses
        if n_epochs_inv_dyn > 0:
            save_dir = os.path.join(self.output_dir, 'inv_model')
            plt.plot(curiosity_train_losses)
            plt.savefig(f'{save_dir}/curiosity_train_losses_round{active_learning_round}.png')
            plt.close()
            torch.save(self.curiosity_model.state_dict(), f'{save_dir}/curiosity_model_round{active_learning_round}.pth')

        


    def train_policy_on_full_demos(self, train_losses, human_label_to_obs, train_dataloader, device, lr_scheduler, 
                                   ema, train_sampling_batch, cfg, wandb_run, json_logger):
        with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
            for batch_idx, batch in enumerate(tepoch):
                # device transfer
                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                if train_sampling_batch is None:
                    train_sampling_batch = batch

                # compute loss
                raw_loss = self.model.compute_loss(batch, human_label_to_obs)
                loss = raw_loss / cfg.training.gradient_accumulate_every
                loss.backward()

                # step optimizer
                if self.global_step % cfg.training.gradient_accumulate_every == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    lr_scheduler.step()
                
                # update ema
                if cfg.training.use_ema:
                    ema.step(self.model)

                # logging
                raw_loss_cpu = raw_loss.item()
                tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                train_losses.append(raw_loss_cpu)
                step_log = {
                    'train_loss': raw_loss_cpu,
                    'global_step': self.global_step,
                    'epoch': self.epoch,
                    'lr': lr_scheduler.get_last_lr()[0]
                }

                is_last_batch = (batch_idx == (len(train_dataloader)-1))
                if not is_last_batch:
                    # log of last step is combined with validation and rollout
                    wandb_run.log(step_log, step=self.global_step)
                    json_logger.log(step_log)
                    self.global_step += 1

                if (cfg.training.max_train_steps is not None) \
                    and batch_idx >= (cfg.training.max_train_steps-1):
                    break
        return train_losses, step_log
    
    def train_policy_on_recovery(self, train_losses, human_label_to_obs, train_dataloader, device, lr_scheduler, 
                                 ema, train_sampling_batch, cfg, wandb_run, json_logger, recovery_train_dataloader):
        with tqdm.tqdm(recovery_train_dataloader, desc=f"Training epoch {self.epoch}", 
                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
            for batch_idx, batch in enumerate(tepoch):
                # device transfer
                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                if train_sampling_batch is None:
                    train_sampling_batch = batch

                # compute loss
                raw_loss = self.model.compute_loss(batch, human_label_to_obs)
                loss = raw_loss / cfg.training.gradient_accumulate_every
                loss.backward()

                # step optimizer
                if self.global_step % cfg.training.gradient_accumulate_every == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    lr_scheduler.step()
                
                # update ema
                if cfg.training.use_ema:
                    ema.step(self.model)

                # logging
                raw_loss_cpu = raw_loss.item()
                tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                train_losses.append(raw_loss_cpu)
                step_log = {
                    'train_loss': raw_loss_cpu,
                    'global_step': self.global_step,
                    'epoch': self.epoch,
                    'lr': lr_scheduler.get_last_lr()[0]
                }

                is_last_batch = (batch_idx == (len(train_dataloader)-1))
                if not is_last_batch:
                    # log of last step is combined with validation and rollout
                    wandb_run.log(step_log, step=self.global_step)
                    json_logger.log(step_log)
                    self.global_step += 1

                if (cfg.training.max_train_steps is not None) \
                    and batch_idx >= (cfg.training.max_train_steps-1):
                    break
        return train_losses, step_log


    def run_active_learning_online(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseLowdimDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseLowdimDataset)

        # prune dataset to only include 10 random episodes
        n_episodes=dataset.original_replay_buffer.n_episodes
        start_n_episodes = min(n_episodes, 10)
        episode_idxs = np.random.choice(n_episodes, start_n_episodes, replace=False)
        dataset.set_subset(episode_idxs)


        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
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
        # env_runner: BaseLowdimRunner
        # env_runner = hydra.utils.instantiate(
        #     cfg.task.env_runner,
        #     output_dir=self.output_dir)
        # assert isinstance(env_runner, BaseLowdimRunner)

        # configure env runner
        eval_env_runner: BaseLowdimRunner
        eval_env_runner = hydra.utils.instantiate(
            cfg.task.eval_env_runner,
            output_dir=self.output_dir)
        assert isinstance(eval_env_runner, BaseLowdimRunner)

        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        num_active_learning_rounds = 8
        training_active_interval = 50
        n_epochs_inv_dyn = 1000

        self.setup_curiosity_model()
        self.curiosity_model.to(device)
        self.curiosity_optimizer = torch.optim.Adam(self.curiosity_model.parameters(), lr=1e-3)
        self.curiosity_loss_fn = torch.nn.MSELoss()
        # move optimizer to device
        optimizer_to(self.curiosity_optimizer, device)

        # make directory for saving inverse dynamics model
        if n_epochs_inv_dyn > 0:
            os.makedirs(os.path.join(self.output_dir, 'inv_model'), exist_ok=True
        )
        os.makedirs(os.path.join(self.output_dir, 'results'), exist_ok=True)

    
        obs_to_diff = None
        human_label_to_obs = None
        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        reward_results = {}
        recovery_dataset = None
        rollout_experience_data = []
        with JsonLogger(log_path) as json_logger:

            for active_learning_round in range(num_active_learning_rounds):
                # if active_learning_round > 0:
                #     training_active_interval = 25
                #     n_epochs_inv_dyn = 500
                print(f"Active learning round {active_learning_round}")

                # create save directory for this round
                save_tag = f'al_round_{active_learning_round}'
                output_dir = self.output_dir
                save_tag_folder = output_dir + f'/results/{save_tag}'
                pathlib.Path(save_tag_folder).mkdir(parents=True, exist_ok=True)
                inv_model_save_path = save_tag_folder + '/inv_model'
                pathlib.Path(inv_model_save_path).mkdir(parents=True, exist_ok=True)

                env_runner: BaseLowdimRunner
                env_runner = hydra.utils.instantiate(
                    cfg.task.env_runner,
                    output_dir=self.output_dir)
                env_runner.set_save_tag_folder(output_dir + f'/results')
                env_runner.set_episode_folder(save_tag_folder)

                eval_env_runner.reset_inits(output_dir, save_tag)

                

                # save obs_to_diff to folder
                if obs_to_diff is not None:
                    with open(f'{save_tag_folder}/obs_to_diff.pkl', 'wb') as f:
                        pickle.dump(obs_to_diff, f)
                    # save to txt file as well
                    with open(f'{save_tag_folder}/obs_to_diff.txt', 'w') as f:
                        for item in obs_to_diff:
                            f.write(str(item) + "\n")
                    with open(f'{save_tag_folder}/human_label_to_obs.pkl', 'wb') as f:
                        pickle.dump(human_label_to_obs, f)

                if active_learning_round == 0:
                    self.save_added_trajs_to_video(env_runner, episode_idxs, save_tag_folder, dataset)

                train_dataloader = DataLoader(dataset, **cfg.dataloader)
                # add to dataset the elements from the previous episodes replay bufffer
                prev_recovery_data = output_dir + f'/results/human_recovery.zarr'
                prev_recovery_data_path = pathlib.Path(prev_recovery_data)
                if prev_recovery_data_path.exists():
                    # add to dataset
                    # create new dataset
                    recovery_dataset: BaseLowdimDataset
                    cfg.task.dataset.zarr_path = prev_recovery_data
                    recovery_dataset = hydra.utils.instantiate(cfg.task.dataset)
                    # get all indices
                    recovery_episode_idxs = np.arange(recovery_dataset.original_replay_buffer.n_episodes)
                    recovery_dataset.set_subset(recovery_episode_idxs)
                    # add to trainloader    
                    recovery_train_dataloader = DataLoader(recovery_dataset, **cfg.dataloader)


                self.train_curiosity_model(self.model, dataset, recovery_dataset, rollout_experience_data,
                                           cfg, active_learning_round, device, n_epochs_inv_dyn)

                # move model to device
                self.model.to(device)

                for local_epoch_idx in range(training_active_interval):
                    print("local_epoch_idx", local_epoch_idx)
                    step_log = dict()
                    # ========= train on full demos for this epoch ==========
                    train_losses = list()
                    train_losses, step_log = self.train_policy_on_full_demos(train_losses, human_label_to_obs, train_dataloader, device, lr_scheduler,
                                                    ema, train_sampling_batch, cfg, wandb_run, json_logger)
                    # ========= train on recovery data for this epoch ==========
                    if active_learning_round > 0:
                        train_losses, step_log = self.train_policy_on_recovery(train_losses, human_label_to_obs, train_dataloader, device, lr_scheduler,
                                                    ema, train_sampling_batch, cfg, wandb_run, json_logger, recovery_train_dataloader)
                        
                    # at the end of each epoch
                    # replace train_loss with epoch average
                    train_loss = np.mean(train_losses)
                    step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # step_log = dict()
                # run rollout
                # if (self.epoch % cfg.training.rollout_every) == 0:
                
                
                runner_log, obs_to_diff, new_human_label_to_obs, extended_rollout_data = env_runner.run(policy, self.curiosity_model)
                if human_label_to_obs is None:
                    human_label_to_obs = new_human_label_to_obs
                else:
                    for key in new_human_label_to_obs:
                        human_label_to_obs[key].extend(new_human_label_to_obs[key])

                # add extended rollout data to rollout_experience_data
                rollout_experience_data.extend(extended_rollout_data)

                runner_log, _ = eval_env_runner.run(policy, self.curiosity_model)
                list_of_max_rewards = []
                for key in runner_log:
                    if 'reward' in key:
                        list_of_max_rewards.append(runner_log[key])
                reward_results[save_tag] = list_of_max_rewards
                # runner_log['test_mean_score']= 0
                

                # save reward results to save_tag_folder as txt
                with open(f'{save_tag_folder}/reward_results.txt', 'w') as f:
                    f.write(str(reward_results))
                # save as pickle
                with open(f'{save_tag_folder}/reward_results.pkl', 'wb') as f:
                    pickle.dump(reward_results, f)

                # log all
                step_log.update(runner_log)
                step_log['val_loss'] = 0

                
                # checkpoint
                # if (self.epoch % cfg.training.checkpoint_every) == 0:
                    # checkpointing
                if cfg.checkpoint.save_last_ckpt:
                    self.save_checkpoint()
                if cfg.checkpoint.save_last_snapshot:
                    self.save_snapshot()

                # sanitize metric names
                metric_dict = dict()
                for key, value in step_log.items():
                    new_key = key.replace('/', '_')
                    metric_dict[new_key] = value
                
                # We can't copy the last checkpoint here
                # since save_checkpoint uses threads.
                # therefore at this point the file might have been empty!
                topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                if topk_ckpt_path is not None:
                    self.save_checkpoint(path=topk_ckpt_path)
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

                # self.save_checkpoint(tag='active_learning_round_{}'.format(active_learning_round))
                # self.save_snapshot(tag='active_learning_round_{}'.format(active_learning_round))


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetLowdimWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
