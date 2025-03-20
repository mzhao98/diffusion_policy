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
from diffusion_policy.policy.diffusion_unet_finetune_lowdim_policy_w_langtraj import DiffusionUnetLowdimPolicy
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusers.training_utils import EMAModel
import pdb
import dill
import pickle

OmegaConf.register_new_resolver("eval", eval, replace=True)



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

        # self.freeze_models_except(['lang_context_encoder', 'traj_context_encoder', 'cond_encoder'])
        self.freeze_models_except(['lang_context_encoder','cond_encoder', 'traj_context_encoder'])

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        self.global_step = 0
        self.epoch = 0

    def freeze_models_except(self, leave_alone):
        # example leave_alone = ['lang_context_encoder', 'traj_context_encoder']
        # pdb.set_trace()
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        for name, param in self.model.named_parameters():
            for la in leave_alone:
                if la in name:
                    param.requires_grad = True
                    # break
                    # break
                    # if la == 'cond_encoder':
                    # # #     # Only finetune the first layer of cond_encoder
                    # # #     # name:model.down_modules.0.0.cond_encoder.1.weight
                    # # #     #  name:model.down_modules.0.0.cond_encoder.1.bias
                    #     if name == 'model.up_modules.1.1.cond_encoder.1.weight' or name == 'model.up_modules.1.1.cond_encoder.1.bias':
                    #     # if name == 'model.down_modules.0.0.cond_encoder.0.weight' or name == 'model.down_modules.0.0.cond_encoder.0.bias':
                    #         param.requires_grad = True
                    # #     elif name == 'model.up_modules.1.1.cond_encoder.1.weight' or name == 'model.up_modules.1.1.cond_encoder.1.bias':
                    # #         param.requires_grad = True
                    # #     else:
                    # #         param.requires_grad = False
                    # else:
                    #     param.requires_grad = True
                    #     break
                # else:
                #     param.requires_grad = False
                #     break

        for name, param in self.ema_model.named_parameters():
            param.requires_grad = False


        for name, param in self.ema_model.named_parameters():
            for la in leave_alone:
                if la in name:
                    param.requires_grad = True
                    # break
                    # break
                    # if la == 'cond_encoder':
                    # # #     # Only finetune the first layer of cond_encoder
                    # # #     # name:model.down_modules.0.0.cond_encoder.1.weight
                    # # #     #  name:model.down_modules.0.0.cond_encoder.1.bias
                    #     if name == 'model.up_modules.1.1.cond_encoder.1.weight' or name == 'model.up_modules.1.1.cond_encoder.1.bias':
                    #     # if name == 'model.down_modules.0.0.cond_encoder.0.weight' or name == 'model.down_modules.0.0.cond_encoder.0.bias':
                    #         param.requires_grad = True
                    # #     elif name == 'model.up_modules.1.1.cond_encoder.1.weight' or name == 'model.up_modules.1.1.cond_encoder.1.bias':
                    # #         param.requires_grad = True
                    # #     else:
                    # #         param.requires_grad = False
                    # # #         break
                    # else:
                    #     param.requires_grad = True
                    #     break
                # else:
                #     param.requires_grad = False
                #     break
        
    def get_cluster_centers(self, demos_dataset, cfg):
        # for each trajectory in the dataset, get the actions
        list_of_traj_action_seqs = []
        for traj_idx in range((demos_dataset.replay_buffer.n_episodes)):
            episode = demos_dataset.replay_buffer.get_episode(traj_idx)
            episode_actions = episode['action']
            # episode_actions_flat = episode_actions.flatten()
            list_of_traj_action_seqs.append(episode_actions)

        # max length of the action sequences
        max_len = max([len(x) for x in list_of_traj_action_seqs])
        mean_len = int(np.mean([len(x) for x in list_of_traj_action_seqs]))
        # scale all the action sequences to the same length
        new_list_of_traj_action_seqs = []
        for traj_idx in range(len(list_of_traj_action_seqs)):
            action_seq = list_of_traj_action_seqs[traj_idx]
            # pdb.set_trace()
            indices = np.linspace(0, action_seq.shape[0]-1, mean_len, dtype=int)
            action_seq_subset = action_seq[indices]
            # flatten the action sequence
            action_seq_subset_flat = action_seq_subset.flatten()
            new_list_of_traj_action_seqs.append(action_seq_subset_flat)
        
        # stack all the action sequences
        all_action_seqs = np.stack(new_list_of_traj_action_seqs)

        # cluster the action sequences
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        n_clusters_to_silhouette_score = {}
        for n_clusters in range(2, 100, 5):
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(all_action_seqs)
            # get silhouette score
            silhouette_score_val = silhouette_score(all_action_seqs, kmeans.labels_)
            print("silhouette_score_val", silhouette_score_val)
            n_clusters_to_silhouette_score[n_clusters] = silhouette_score_val
        
        # choose the best number of clusters
        best_n_clusters = max(n_clusters_to_silhouette_score, key=n_clusters_to_silhouette_score.get)
        best_n_clusters = 5
        print("best_n_clusters", best_n_clusters)
        kmeans = KMeans(n_clusters=best_n_clusters, random_state=0).fit(all_action_seqs)
        cluster_centers = kmeans.cluster_centers_
        # get indices of the cluster centers
        cluster_center_indices = [] 
        for cluster_center in cluster_centers:
            cluster_center_idx = np.argmin(np.linalg.norm(all_action_seqs - cluster_center, axis=1))
            cluster_center_indices.append(cluster_center_idx)
        
        print("cluster_center_indices", cluster_center_indices)
        # pdb.set_trace()
        return best_n_clusters, cluster_center_indices




        

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # load trained model
        checkpoint_path = 'data/outputs/2025.02.18/17.48.31_train_diffusion_unet_lowdim_push2d_lowdim/checkpoints/latest.ckpt'
        # checkpoint_path = 'data/outputs/2025.03.07/15.31.52_train_diffusion_unet_lowdim_push2d_lowdim/checkpoints/latest.ckpt'
        
        print(f"Resuming from checkpoint {checkpoint_path}")
        self.load_checkpoint(path=checkpoint_path)

        

        
        device = torch.device(cfg.training.device)

        # configure Training Demos dataset
        print("\n\ncfg.task.dataset", cfg.task.dataset)
        demos_dataset: BaseLowdimDataset
        cfg.task.dataset.use_target = False
        cfg.task.dataset.val_ratio = 0.5
        demos_dataset = hydra.utils.instantiate(cfg.task.dataset)
        print("len(demos_dataset): ", len(demos_dataset))
        # dataset.set_device(device)
        assert isinstance(demos_dataset, BaseLowdimDataset)
        train_dataloader = DataLoader(demos_dataset, **cfg.dataloader)
        train_normalizer = demos_dataset.get_normalizer()

        # configure Target Demos dataset
        
        target_dataset: BaseLowdimDataset
        cfg.task.dataset.use_target = True
        cfg.task.dataset.val_ratio = 0.0

        # n_clusters, cluster_centers = self.get_cluster_centers(demos_dataset, cfg)
        n_clusters = 2
        cluster_centers = ['right', 'left']
        self.model.create_weight_transformer(n_clusters, cluster_centers)
        self.ema_model = copy.deepcopy(self.model)

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

        self.model.set_normalizer(train_normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(train_normalizer)

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
            
        # self.freeze_models_except(['lang_context_encoder', 'traj_context_encoder', 'cond_encoder'])
        # self.freeze_models_except(['weight_transformer'])
        self.freeze_models_except(['cond_encoder', 'lang_context_encoder', 'weight_transformer'])
        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

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

        # Create the results dir
        results_path = pathlib.Path(self.output_dir).joinpath('results')
        # make directory results_path
        os.makedirs(results_path, exist_ok=True)

        # finetune the weights
        self.freeze_models_except(['weight_transformer'])
        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())
        
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(0):
                print(f"Starting epoch {local_epoch_idx}")

                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                with tqdm.tqdm(target_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        obs_input = batch['obs']
                        obs_input = obs_input.view(obs_input.shape[0], -1)
                        
                        # transfer to device 
                        obs_input = obs_input.to(torch.device(cfg.training.device))

                        

                        # compute loss
                        raw_loss = self.model.compute_loss_with_weight_cfg(batch)
                        # raw_loss = self.model.compute_loss_with_cfg(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        # print("loss", loss)
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

        # get performance of the model after finetuning the weights
        # ========= eval for this epoch ==========
        policy = self.model
        if cfg.training.use_ema:
            policy = self.ema_model
        policy.eval()

        # run rollout
        # all_utterances = ['left', 'right']
        # all_utterances.extend(list(set(list(target_utterance_dict.values()))))
        use_weight_transform = False
        
        reward_results = {}

        save_tag = 'weight_adapt'
        output_dir = self.output_dir
        save_tag_folder = output_dir + f'/results/{save_tag}'
        pathlib.Path(save_tag_folder).mkdir(parents=True, exist_ok=True)

        # for utterance in all_utterances:
        #     # print("utterance", utterance)
        #     utterance_folder = output_dir + f'/results/{save_tag}/' + utterance
        #     # print("utterance_folder", utterance_folder)
        #     pathlib.Path(utterance_folder).mkdir(parents=True, exist_ok=True)

        #     env_runner.reset_inits(self.output_dir, utterance, save_tag)
        #     runner_log = env_runner.run_train_eval(policy, utterance, save_tag, use_weight_transform)
        #     list_of_max_rewards = []
        #     for key in runner_log:
        #         if 'reward' in key:
        #             list_of_max_rewards.append(runner_log[key])
        #     reward_results[save_tag][utterance] = list_of_max_rewards
        utterance_to_use = list(target_utterance_dict.values())[0]
        print("utterance_to_use", utterance_to_use)
        env_runner.reset_inits(self.output_dir, save_tag)
        runner_log = env_runner.run_train_eval(policy, utterance_to_use, use_weight_transform)
        list_of_max_rewards = []
        for key in runner_log:
            if 'reward' in key:
                list_of_max_rewards.append(runner_log[key])
        reward_results[save_tag] = list_of_max_rewards



        # finetune the lang_context_encoder and traj_context_encoder
        self.freeze_models_except(['cond_encoder'])
        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())
        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(21):
                print(f"Starting epoch {local_epoch_idx}")

                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                with tqdm.tqdm(target_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        obs_input = batch['obs']
                        obs_input = obs_input.view(obs_input.shape[0], -1)
                        
                        # transfer to device 
                        obs_input = obs_input.to(torch.device(cfg.training.device))

                        # compute loss
                        raw_loss = self.model.compute_loss_with_cfg(batch)
                        # raw_loss = self.model.compute_loss_with_weight_lang_cfg(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        # print("loss", loss)
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

                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        obs_input = batch['obs']
                        obs_input = obs_input.view(obs_input.shape[0], -1)
                        
                        # transfer to device 
                        obs_input = obs_input.to(torch.device(cfg.training.device))

                        # compute loss
                        raw_loss = self.model.compute_loss_with_cfg(batch)
                        # raw_loss = self.model.compute_loss_with_weight_lang_cfg(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        # print("loss", loss)
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
                step_log['test_mean_score'] = 0

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                # if (self.epoch % cfg.training.rollout_every) == 0:
                #     runner_log = env_runner.run(policy)
                #     pdb.set_trace()
                #     # log all
                #     step_log.update(runner_log)
                if (self.epoch % cfg.training.rollout_every) == 0:
                    save_tag = f'finetune_{local_epoch_idx}'
                    output_dir = self.output_dir
                    save_tag_folder = output_dir + f'/results/{save_tag}'
                    pathlib.Path(save_tag_folder).mkdir(parents=True, exist_ok=True)
                    env_runner.reset_inits(self.output_dir, save_tag)
                    runner_log = env_runner.run_train_eval(policy, save_tag, use_weight_transform)
                    list_of_max_rewards = []
                    for key in runner_log:
                        if 'reward' in key:
                            list_of_max_rewards.append(runner_log[key])
                    reward_results[save_tag] = list_of_max_rewards
                    step_log.update(runner_log)

                    # for utterance in all_utterances:
                        
                        # utterance_folder = output_dir + f'/results/{save_tag}/' + utterance
                        # pathlib.Path(utterance_folder).mkdir(parents=True, exist_ok=True)

                        # env_runner.reset_inits(self.output_dir, utterance, save_tag)
                        # runner_log = env_runner.run_train_eval(policy, utterance, save_tag, use_weight_transform)
                        # list_of_max_rewards = []
                        # for key in runner_log:
                        #     if 'reward' in key:
                        #         list_of_max_rewards.append(runner_log[key])
                        # reward_results[save_tag][utterance] = list_of_max_rewards
                        # step_log.update(runner_log)

                # skip running validation
                step_log['val_loss'] = 0.0

                
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
        # save the reward results
        with open(f'{self.output_dir}/reward_results.pkl', 'wb') as f:
            pickle.dump(reward_results, f)

        # plot the reward_results
        self.plot_reward_results(reward_results)

    def plot_reward_results(self, reward_results):
        import matplotlib.pyplot as plt
        import numpy as np
        t_to_mean_list = []
        t_to_std_list = []
        for step_key in reward_results.keys():
            # print("step_key", step_key)
            # print("reward_results[step_key]", reward_results[step_key])
            mean_rewards = np.mean(reward_results[step_key], axis=0)
            std_rewards = np.std(reward_results[step_key], axis=0)
            # print("mean_rewards", mean_rewards)
            # print("std_rewards", std_rewards)
            t_to_mean_list.append(mean_rewards)
            t_to_std_list.append(std_rewards)

        mean_rewards = np.array(t_to_mean_list)
        std_rewards = np.array(t_to_std_list)
            
        # plot the results as time series with error bars
        t = np.arange(0, len(mean_rewards))
        plt.plot(t, mean_rewards, label='Mean Reward')
        plt.fill_between(t, mean_rewards-std_rewards, mean_rewards+std_rewards, alpha=0.2, label='Std Dev')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Mean Reward')
        plt.title('Mean Reward vs Epochs')
        plt.savefig(f'{self.output_dir}/mean_reward_vs_epochs.png')
        plt.show()



@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetLowdimWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
