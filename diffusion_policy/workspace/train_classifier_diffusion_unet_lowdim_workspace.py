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


# build classifier to take in the observation and output a class
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
        # make classifier to predict the class
        self.context_classifier = Classifier(16*20 + 16*2,2)
        self.context_classifier.to(torch.device(cfg.training.device))
        
        # load the classifier
        # classifier.load_state_dict(torch.load('classifier_ood.pth'))
        

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.training.resume:
            # lastest_ckpt_path = self.get_checkpoint_path()
            # lastest_ckpt_path = 'data/outputs/2025.02.04/22.51.29_train_diffusion_unet_lowdim_push2d_lowdim/checkpoints/epoch=0030-test_mean_score=0.644.ckpt'
            # lastest_ckpt_path = 'data/outputs/2025.02.04/23.52.42_train_diffusion_unet_lowdim_push2d_lowdim/checkpoints/epoch=0060-test_mean_score=0.827.ckpt'
            lastest_ckpt_path = 'data/outputs/2025.02.05/22.00.31_train_diffusion_unet_lowdim_push2d_lowdim/checkpoints/latest.ckpt'
                        # if lastest_ckpt_path.is_file():
            print(f"Resuming from checkpoint {lastest_ckpt_path}")
            self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        # pdb.set_trace()
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
        # device = torch.device('cpu')
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)


        single_batch = dataset[0]
        obs_dict = {'obs': single_batch['obs'].unsqueeze(0)} # torch.Size([1, 16, 21])
        gt_action = single_batch['action'].unsqueeze(0) 

        optimizer = torch.optim.Adam(self.context_classifier.parameters(), lr=1e-5)
        # criterion = torch.nn.functional.mse_loss
        criterion = torch.nn.CrossEntropyLoss()
        self.context_classifier.load_state_dict(torch.load('classifier_ood_2class_each_r0.pth'))

        # sample k datapoints
        K = len(dataset)
        total_original_mse = 0
        total_classifier_mse = 0
        for k in range(65,len(dataset),1):
            print(f"K: {k} of {len(dataset)}")
            # k = random.randint(0, len(dataset)-1)
            single_batch = dataset[k]
            obs_dict = {'obs': single_batch['obs'].unsqueeze(0),
                        'idx': single_batch['idx']}
            gt_action = single_batch['action']
            obs_dict = dict_apply(obs_dict, lambda x: x.to(device))
            obs = obs_dict['obs']
            obs = obs.view(obs.shape[0], -1)
            gt_action = gt_action.to(device)
            
            # get new classifier output
            pred_class = self.context_classifier(obs, gt_action.unsqueeze(0).view(gt_action.unsqueeze(0).shape[0], -1))
            print("pred_class", pred_class)
            # pred_class = pred_class
            # pred class takes the index of the max value
            pred_class = torch.argmax(pred_class, dim=1)*9
            
            # make a copy of the original obs_dict
            original_obs_dict = obs_dict.copy()
            original_obs_dict['idx'] = (1-pred_class/9)*9

            # modified obs_dict
            obs_dict['idx'] = pred_class

            # move data to device
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
            print(f"new mse: {mse}")
            # get the original prediction
            original_result = self.model.predict_action(original_obs_dict)
            if cfg.pred_action_steps_only:
                original_pred_action = original_result['action']
                start = cfg.n_obs_steps - 1
                end = start + cfg.n_action_steps
                gt_action = gt_action[:,start:end]
            else:
                original_pred_action = original_result['action_pred']
            original_mse = torch.nn.functional.mse_loss(original_pred_action, gt_action)
            print(f"original_mse: {original_mse}")
            total_original_mse += original_mse
            total_classifier_mse += mse

            # predicted class to cpu
            pred_class = pred_class.detach().cpu().numpy()[0]
            original_mse = original_mse.detach().cpu().numpy()
            mse = mse.detach().cpu().numpy()
            # plot the old and new predictions
            plt.figure(figsize=(8, 6))
            plt.xlim(0,450)
            plt.ylim(0,450)
            for t in range(pred_action.shape[1]):
                pred_act_t = pred_action[0,t]
                original_pred_act_t = original_pred_action[0,t]
                gt_action_t = gt_action[t]
                # flatten data
                flatten_pred_action = pred_act_t.view(-1).detach().cpu().numpy()
                flatten_original_pred_action = original_pred_act_t.view(-1).detach().cpu().numpy()
                flatten_gt_action = gt_action_t.view(-1).detach().cpu().numpy()
                plt.scatter(flatten_pred_action[0], flatten_pred_action[1], s=50, alpha=0.5, c='red', label='new')
                plt.scatter(flatten_original_pred_action[0], 
                            flatten_original_pred_action[1], s=50, alpha=0.5, c='blue', label='orig')
                plt.scatter(flatten_gt_action[0], flatten_gt_action[1], s=100, c='black', marker='*', label='gt')
            # plt.legend()
            plt.title(f"Predictions: class:{pred_class}, orig mse: {np.round(original_mse,1)}, new mse: {np.round(mse,1)}")
            plt.xlabel("X-axis")
            plt.ylabel("Y-axis")
            plt.savefig(f'distribution_viz/predictions_k={k}.png')
            plt.close()
        print(f"Total Original MSE: {total_original_mse}")
        print(f"Total Classifier MSE: {total_classifier_mse}")

        return


        self.model.eval()   
        init_original_mse = 0
        final_classifier_mse = 0
        train_losses = []
        correct_class_dict = {}
        
        max_iters = 10
        for iter in range(max_iters):
            print(f"Iteration: {iter}")
            current_classifier_mse = 0
            with tqdm.tqdm(train_dataloader, desc=f"Training epoch {iter}", 
                        leave=False) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        if batch_idx not in correct_class_dict:
                            correct_class_dict[batch_idx] = None
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

                
                        obs_dict = {'obs': batch['obs'],
                                    'idx': batch['idx']}
                        gt_action = batch['action']
                        obs_dict = dict_apply(obs_dict, lambda x: x.to(device))
                        obs = obs_dict['obs']
                        obs = obs.view(obs.shape[0], -1)
                        
                        # get new classifier output
                        # pdb.set_trace()
                        pred_class = self.context_classifier(obs, gt_action.view(gt_action.shape[0], -1))
                        # pred_class = pred_class
                        # print("pred_class", pred_class)
                        # clip pred_class to be between 0 and 9
                        # pred_class = torch.clamp(pred_class, 0, 9)
                        # print("pred_class", pred_class)
                        # pred class takes the index of the max value
                        pred_class_indices = torch.argmax(pred_class, dim=1)
                        print("pred_class_indices", pred_class_indices)
                        
                        # make a copy of the original obs_dict
                        # original_obs_dict = obs_dict.copy()

                        # modified obs_dict
                        obs_dict['idx'] = pred_class

                        # get correct class by choosing 0 or 1 that gives the best mse
                        candidate_0_obs_dict = obs_dict.copy()
                        candidate_0_obs_dict['idx'] = torch.zeros_like(pred_class_indices)
                        candidate_1_obs_dict = obs_dict.copy()
                        candidate_1_obs_dict['idx'] = torch.ones_like(pred_class_indices)*9

                        N_samples = 1
                        # get average pred_0_action and pred_1_action
                        avg_pred_0_action = None
                        avg_pred_1_action = None
                        for n in range(N_samples):
                            pred_0_action = self.model.predict_action(candidate_0_obs_dict)['action_pred']
                            pred_1_action = self.model.predict_action(candidate_1_obs_dict)['action_pred']
                            if avg_pred_0_action is None:
                                avg_pred_0_action = pred_0_action
                                avg_pred_1_action = pred_1_action
                            else:
                                avg_pred_0_action += pred_0_action
                                avg_pred_1_action += pred_1_action
                        avg_pred_0_action /= N_samples
                        avg_pred_1_action /= N_samples
                        pred_0_action = avg_pred_0_action
                        pred_1_action = avg_pred_1_action

                        # pred_0_action = self.model.predict_action(candidate_0_obs_dict)['action_pred']
                        # pred_1_action = self.model.predict_action(candidate_1_obs_dict)['action_pred']
                        gt_action = gt_action.to(device)
                        mse_0 = torch.nn.functional.mse_loss(pred_0_action, gt_action, reduction='none')
                        mse_1 = torch.nn.functional.mse_loss(pred_1_action, gt_action, reduction='none')
                        mse_0 = torch.sum(mse_0, dim=1)
                        mse_1 = torch.sum(mse_1, dim=1)
                        mse_0 = torch.sum(mse_0, dim=1)
                        mse_1 = torch.sum(mse_1, dim=1)
                        # print("mse_0", mse_0)
                        # print("mse_1", mse_1)
                        # pdb.set_trace()
                        # choose the class that gives the best mse
                        correct_class = torch.where(mse_0 < mse_1, torch.zeros_like(pred_class_indices), torch.ones_like(pred_class_indices))
                        if correct_class_dict[batch_idx] is None:
                            correct_class_dict[batch_idx] = correct_class
                        else:
                            # add
                            correct_class_dict[batch_idx] += correct_class
                        
                        # average over number of adds
                        correct_class = correct_class_dict[batch_idx] / (iter+1)
                        # round to 0 if less than or equal to 0.5, 1 if greater than 0.5
                        correct_class = torch.where(correct_class <= 0.5, torch.zeros_like(pred_class_indices), torch.ones_like(pred_class_indices))
                        # convert to long
                        correct_class = correct_class.long()
                        # # correct_class = torch.where(mse_0 < mse_1, torch.zeros_like(pred_class_indices), torch.ones_like(pred_class_indices))
                        print("correct_class", correct_class)
                        # pdb.set_trace()
                        # move data to device
                        # num_preds = 3
                        # average_pred_action = None
                        # for p in range(num_preds):
                        #     result = self.model.predict_action(obs_dict)
                        #     if cfg.pred_action_steps_only:
                        #         pred_action = result['action']
                        #         start = cfg.n_obs_steps - 1
                        #         end = start + cfg.n_action_steps
                        #         gt_action = gt_action[:,start:end]
                        #         # add to average    
                        #         if average_pred_action is None:
                        #             average_pred_action = pred_action
                        #         else:
                        #             average_pred_action += pred_action
                        #     else:
                        #         pred_action = result['action_pred']
                        #         # add to average
                        #         if average_pred_action is None:
                        #             average_pred_action = pred_action
                        #         else:
                        #             average_pred_action += pred_action
                        # # average the predictions
                        # average_pred_action /= num_preds
                        # pred_action = average_pred_action

                        # move both to device
                        # pred_action = pred_action.to(device)
                        # gt_action = gt_action.to(device)    
                        loss = criterion(pred_class, correct_class)     
                        print("loss", loss)       
                        # loss = criterion(pred_action, gt_action)
                        if iter == 0:
                            init_original_mse += loss.item()
                        if iter == max_iters - 1:
                            final_classifier_mse += loss.item()
                        current_classifier_mse += loss.item()
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
            print(f"Initial Original MSE: {init_original_mse}")
            print(f"Current Classifier MSE: {current_classifier_mse}")
            train_losses.append(current_classifier_mse)
            print(f"Final Classifier MSE: {final_classifier_mse}")
            torch.save(self.context_classifier.state_dict(), 'classifier_ood_2class_each_r0.pth')
            print(f"Classifier saved")
                
        # plot the training losses
        plt.figure(figsize=(8, 6))
        plt.plot(train_losses)
        plt.title(f"Training Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(f'distribution_viz/training_losses.png')
        plt.show()


                

                # print("loss total: ", loss_total)
            # print(f"Total Original MSE: {total_original_mse}")
            # print(f"Total Classifier MSE: {total_classifier_mse}")





        # for index from 0 to demo_end_idx
        # get the batch from the dataset
        # get the observation from the batch
        # get the action from the batch
        # get the prediction from the model
        # get the mse between the prediction and the action
        # self.model.eval()   
        # for i in range(demo_end_idx):
        #     single_batch = dataset[i]
        #     # pdb.set_trace()
        #     obs_dict = {'obs': single_batch['obs'].unsqueeze(0)}
        #     gt_action = single_batch['action'].unsqueeze(0)

        #     # move data to device
        #     obs_dict = dict_apply(obs_dict, lambda x: x.to(device))
        #     gt_action = gt_action.to(device)
            
        #     # Sample K= 30 predictions
        #     K = 30
        #     sampled_predictions = {}
        #     for k in range(K):
        #         result = self.model.predict_action(obs_dict)
        #         pred_action = result['action'][0]
        #         # flatten
        #         for t in range(pred_action.shape[0]):
        #             pred_act_t = pred_action[t]
        #             if t not in sampled_predictions:
        #                 sampled_predictions[t] = []
        #         # flatten data
        #         # flatten_pred_action = pred_action.view(-1)
        #             flatten_pred_action = pred_act_t.view(-1)
        #             sampled_predictions[t].append(flatten_pred_action.detach().cpu().numpy())

        #     # plot the sampled predictions
        #     # plot the ground truth action
        #     plt.figure(figsize=(8, 6))
        #     plt.xlim(0,450)
        #     plt.ylim(0,450)
        #     for t in sampled_predictions:
        #         sampled_predictions[t] = np.array(sampled_predictions[t])
        #         for j in range(sampled_predictions[t].shape[0]):
        #             plt.scatter(sampled_predictions[t][j][0], sampled_predictions[t][j][1], s=50, alpha=0.5)
        #         # plot gt action with a black star
        #         gt_action_t = gt_action[0,t]
        #         gt_action_t = gt_action_t.view(-1).detach().cpu().numpy()
        #         plt.scatter(gt_action_t[0], gt_action_t[1], s=100, c='black', marker='*')
        #         plt.legend()
        #         plt.title(f"Sampled Predictions and Ground Truth Action:")
        #         plt.xlabel("X-axis")
        #         plt.ylabel("Y-axis")
        #         plt.savefig(f'distribution_viz/sample_predictions_{i}_subt_{t}.png')
        #         plt.show()
        #     pdb.set_trace()
            

    


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

