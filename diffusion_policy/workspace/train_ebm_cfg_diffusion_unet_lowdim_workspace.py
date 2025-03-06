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

OmegaConf.register_new_resolver("eval", eval, replace=True)


class Classifier(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.hidden_dim = 512
        self.hidden_dim2 = 172
        self.hidden_dim3 = 64
        self.hidden_dim4 = 256
        self.hidden_dim5 = 128
        self.hidden_dim6 = 64
        self.hidden_dim7 = 32
        self.fc = torch.nn.Linear(input_dim, self.hidden_dim)
        self.fc2 = torch.nn.Linear(self.hidden_dim, self.hidden_dim2)
        self.fc3 = torch.nn.Linear(self.hidden_dim2, self.hidden_dim3)
        self.fc4 = torch.nn.Linear(self.hidden_dim3, output_dim)
        # self.fc5 = torch.nn.Linear(self.hidden_dim4, self.hidden_dim5)
        # self.fc6 = torch.nn.Linear(self.hidden_dim5, self.hidden_dim6)
        # self.fc7 = torch.nn.Linear(self.hidden_dim6, self.hidden_dim7)
        # self.fc8 = torch.nn.Linear(self.hidden_dim7, output_dim)
        self.softplus = torch.nn.Softplus()

    def forward(self, x, action):
        x = torch.cat((x, action), dim=1)
        x = self.fc(x)
        # x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        # x = torch.nn.functional.relu(x)
        x = self.fc3(x)
        # x = torch.nn.functional.relu(x)
        x = self.fc4(x)
        # x = torch.nn.functional.relu(x)
        # x = self.fc5(x)
        # # x = torch.nn.functional.relu(x)
        # x = self.fc6(x)
        # x = torch.nn.functional.relu(x)
        # x = self.fc7(x)
        # x = torch.nn.functional.relu(x)
        # x = self.fc8(x)

        # x = torch.nn.functional.relu(x)
        # make in the range -1 to 1
        # x = torch.nn.functional.tanh(x) *10
        # scale between 0 and 10
        # x = torch.nn.functional.relu(x) * 10
        # softmax
        # x = torch.nn.functional.softmax(x, dim=1)
        # x = self.softplus(x)

        return x

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

        # load classifier from state dict
        # context_classifier = Classifier(16*20 + 16*2, 2)
        # # load the classifier
        # context_classifier.load_state_dict(torch.load('classifier_ood_2class_each_r0.pth'))
        # self.context_classifier = context_classifier
        # # move to device
        # self.context_classifier.to(torch.device(cfg.training.device))

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            # lastest_ckpt_path = 'data/outputs/2025.02.04/22.51.29_train_diffusion_unet_lowdim_push2d_lowdim/checkpoints/epoch=0030-test_mean_score=0.644.ckpt'
            # lastest_ckpt_path = 'data/outputs/2025.02.04/23.52.42_train_diffusion_unet_lowdim_push2d_lowdim/checkpoints/epoch=0060-test_mean_score=0.827.ckpt'
            # lastest_ckpt_path = 'data/outputs/2025.02.05/01.05.38_train_diffusion_unet_lowdim_push2d_lowdim/checkpoints/epoch=0090-test_mean_score=0.864.ckpt'
            # lastest_ckpt_path = 'data/outputs/2025.02.05/22.00.31_train_diffusion_unet_lowdim_push2d_lowdim/checkpoints/latest.ckpt'
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        device = torch.device(cfg.training.device)
        
        dataset: BaseLowdimDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        # dataset.set_device(device)
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

        global_strategy_mode = 0

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

                        obs_input = batch['obs']
                        obs_input = obs_input.view(obs_input.shape[0], -1)
                        
                        # get new classifier output
                        # transfer to device 
                        obs_input = obs_input.to(torch.device(cfg.training.device))
                        # pred_class = self.context_classifier(obs_input, batch['action'].view(batch['action'].shape[0], -1))
                        # take argmax
                        # pred_class = torch.argmax(pred_class, dim=1)*9

                        # batch['idx'] = pred_class
                        # pdb.set_trace()
                        

                        # compute loss
                        raw_loss = self.model.compute_loss_with_cfg(batch)
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
                    env_runner.set_mode(global_strategy_mode)
                    global_strategy_mode = (global_strategy_mode + 1) % 2
                    runner_log = env_runner.run(policy)
                    # log all
                    step_log.update(runner_log)

                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    # with torch.no_grad():
                    val_losses = list()
                    with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                            leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                            # pdb.set_trace()
                            loss = self.model.compute_loss_with_cfg(batch)
                            val_losses.append(loss)
                            if (cfg.training.max_val_steps is not None) \
                                and batch_idx >= (cfg.training.max_val_steps-1):
                                break
                    if len(val_losses) > 0:
                        val_loss = torch.mean(torch.tensor(val_losses)).item()
                        # log epoch average validation loss
                        step_log['val_loss'] = val_loss

                # run diffusion sampling on a training batch
                # if (self.epoch % cfg.training.sample_every) == 0:
                #     with torch.no_grad():
                #         # sample trajectory from training set, and evaluate difference
                #         batch = train_sampling_batch
                #         obs_dict = {'obs': batch['obs']}
                #         gt_action = batch['action']
                #         obs_dict['idx'] = batch['idx']
                #         # squeeze the dimension at 1
                #         obs_dict['obs'] = obs_dict['obs'].squeeze(1)
                #         # pdb.set_trace()
                        
                #         result = policy.predict_action(obs_dict)
                #         if cfg.pred_action_steps_only:
                #             pred_action = result['action']
                #             start = cfg.n_obs_steps - 1
                #             end = start + cfg.n_action_steps
                #             gt_action = gt_action[:,start:end]
                #         else:
                #             pred_action = result['action_pred']
                #         mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                #         # log
                #         step_log['train_action_mse_error'] = mse.item()
                #         # release RAM
                #         del batch
                #         del obs_dict
                #         del gt_action
                #         del result
                #         del pred_action
                #         del mse
                
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

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetLowdimWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
