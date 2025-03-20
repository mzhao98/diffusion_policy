import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import wandb.sdk.data_types.video as wv
from diffusion_policy.env.push2d.push2d_keypoints_env import Push2dKeypointsEnv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
# from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
import pdb
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel
# import pygame
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
from diffusion_policy.common.replay_buffer import ReplayBuffer
import pickle
import time

def plot_interactive(imgs_over_episode, state_uncertainty_over_episode, action_uncertainty_over_episode):
    # Global dictionary to store labels: time step index -> label ("Good", "Erroneous", "Avoid")
    labels = {}
    result = {"labels": None, "mode": None}
    # Create the main popup window
    root = tk.Tk()
    root.title("Episode Explorer")

    # --- Time Slider and Image Display ---
    def update_image(val):
        idx = int(slider.get())
        pil_img = Image.fromarray(imgs_over_episode[idx])
        tk_img = ImageTk.PhotoImage(pil_img)
        image_label.config(image=tk_img)
        image_label.image = tk_img  # keep a reference

    slider = tk.Scale(root, from_=0, to=len(imgs_over_episode)-1,
                      orient=tk.HORIZONTAL, command=update_image, label="Time Step")
    slider.pack(fill="x", padx=10, pady=5)

    image_label = tk.Label(root)
    image_label.pack(padx=10, pady=5)

    # --- Uncertainty Plots: Two side-by-side subplots ---
    time_steps = list(range(len(action_uncertainty_over_episode)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))
    
    # Base plot for action uncertainty.
    ax1.plot(time_steps, action_uncertainty_over_episode, '-o', label="Action Uncertainty")
    high_action = [(t, u) for t, u in zip(time_steps, action_uncertainty_over_episode) if u > 1000]
    if high_action:
        rp_time, rp_unc = zip(*high_action)
        ax1.scatter(rp_time, rp_unc, color="red", zorder=3, label="High Uncertainty (>1000)")
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Action Uncertainty")
    ax1.legend()
    
    # Base plot for state uncertainty.
    ax2.plot(time_steps, state_uncertainty_over_episode, '-o', label="State Uncertainty")
    high_state = [(t, u) for t, u in zip(time_steps, state_uncertainty_over_episode) if u > 1000]
    if high_state:
        rp_time2, rp_unc2 = zip(*high_state)
        ax2.scatter(rp_time2, rp_unc2, color="red", zorder=3, label="High Uncertainty (>1000)")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("State Uncertainty")
    ax2.legend()

    

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(padx=10, pady=5)

    def update_plot():
        # Clear and re-draw both axes.
        ax1.clear()
        ax2.clear()
        # Re-plot action uncertainty.
        ax1.plot(time_steps, action_uncertainty_over_episode, '-o', label="Action Uncertainty")
        high_action = [(t, u) for t, u in zip(time_steps, action_uncertainty_over_episode) if u > 1000]
        if high_action:
            rp_time, rp_unc = zip(*high_action)
            ax1.scatter(rp_time, rp_unc, color="red", zorder=3, label="High Uncertainty (>1000)")
        ax1.set_xlabel("Time Step")
        ax1.set_ylabel("Action Uncertainty")
        ax1.legend()
        
        # Re-plot state uncertainty.
        ax2.plot(time_steps, state_uncertainty_over_episode, '-o', label="State Uncertainty")
        high_state = [(t, u) for t, u in zip(time_steps, state_uncertainty_over_episode) if u > 1000]
        if high_state:
            rp_time2, rp_unc2 = zip(*high_state)
            ax2.scatter(rp_time2, rp_unc2, color="red", zorder=3, label="High Uncertainty (>1000)")
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("State Uncertainty")
        ax2.legend()
        
        # Add labelling markers for each labelled time step.
        for idx, lab in labels.items():
            color = label_colors.get(lab, "black")
            ax1.scatter(idx, action_uncertainty_over_episode[idx], color=color, s=100,
                        edgecolors="black", zorder=4)
            ax2.scatter(idx, state_uncertainty_over_episode[idx], color=color, s=100,
                        edgecolors="black", zorder=4)
        canvas.draw()

    # --- Time Bars for Labeling ---
    # Configuration for time bars
    rect_width = 20
    canvas_height = 30
    num_steps = len(time_steps)

    # Colors for each label category.
    label_colors = {
        "Good": "green",
        "Erroneous": "orange",
        "Avoid": "red"
    }

    # Dictionary to store the canvases for each time bar.
    timebar_canvases = {}

    def redraw_timebars():
        for cat, cv in timebar_canvases.items():
            cv.delete("all")
            for t in range(num_steps):
                x0 = t * rect_width
                y0 = 0
                x1 = x0 + rect_width
                y1 = canvas_height
                # Fill the rectangle if this time step is labeled with the corresponding category.
                fill_color = label_colors[cat] if labels.get(t) == cat else "white"
                cv.create_rectangle(x0, y0, x1, y1, fill=fill_color, outline="black")

    def on_timebar_click(event, category):
        t = int(event.x // rect_width)
        if t < 0 or t >= num_steps:
            return
        # Toggle: if already labeled with this category, remove it; otherwise assign this category.
        if labels.get(t) == category:
            del labels[t]
        else:
            labels[t] = category
        redraw_timebars()
        update_plot()

    # Create a frame to hold the three time bars.
    timebars_frame = tk.Frame(root)
    timebars_frame.pack(padx=10, pady=10)

    # For each category, create a label and an interactive canvas.
    for cat in ["Good", "Erroneous", "Avoid"]:
        frame = tk.Frame(timebars_frame)
        frame.pack(fill="x", pady=2)
        tk.Label(frame, text=cat, width=10).pack(side=tk.LEFT)
        cv = tk.Canvas(frame, width=num_steps * rect_width, height=canvas_height, bg="white")
        cv.pack(side=tk.LEFT)
        cv.bind("<Button-1>", lambda event, cat=cat: on_timebar_click(event, cat))
        timebar_canvases[cat] = cv

    redraw_timebars()

    # --- Finished Labelling Buttons ---
    finish_frame = tk.Frame(root)
    finish_frame.pack(fill="x", pady=10)


    def finish_labelling(mode):
        result["mode"] = mode
        result["labels"] = labels.copy()  # make a copy to return
        print(f"Finished labelling with mode: {mode}")
        print("Labels:", labels)
        # root.quit()
        root.destroy()

    finish_btn_avoid = tk.Button(finish_frame, text="Finished Labelling, Avoid",
                                command=lambda: finish_labelling("avoid"))
    finish_btn_avoid.pack(side=tk.LEFT, padx=5)

    finish_btn_recover = tk.Button(finish_frame, text="Finished Labelling, Recover",
                                command=lambda: finish_labelling("recover"))
    finish_btn_recover.pack(side=tk.LEFT, padx=5)

    # Initialize by displaying the first image.
    update_image(0)

    root.mainloop()
    plt.close()

    # close tkinter
    

    return result["labels"], result["mode"]


class Push2dKeypointsRunner(BaseLowdimRunner):
    def __init__(self,
            output_dir,
            keypoint_visible_rate=1.0,
            n_train=10,
            n_train_vis=3,
            train_start_seed=0,
            n_test=22,
            n_test_vis=6,
            legacy_test=False,
            test_start_seed=10000,
            max_steps=200,
            n_obs_steps=8,
            n_action_steps=8,
            n_latency_steps=0,
            fps=10,
            crf=22,
            agent_keypoints=False,
            past_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None
        ):
        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test



        self.savetag = None
        # handle latency step
        # to mimic latency, we request n_latency_steps additional steps 
        # of past observations, and the discard the last n_latency_steps
        self.env_n_obs_steps = n_obs_steps + n_latency_steps
        self.env_n_action_steps = n_action_steps

        
        self.n_train = n_train
        self.n_train_vis = n_train_vis
        self.train_start_seed = train_start_seed
        self.n_test = n_test
        self.n_test_vis = n_test_vis
        self.legacy_test = legacy_test
        self.test_start_seed = test_start_seed
        self.output_dir = output_dir

        self.fps = fps
        self.crf = crf
        self.agent_keypoints = agent_keypoints
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.n_latency_steps = n_latency_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        
    def set_save_tag_folder(self, savetag):
        self.savetag_folder = savetag

    def set_episode_folder(self, epi_savetag):
        self.episode_save_folder = epi_savetag
    
    def run(self, policy: BaseLowdimPolicy, curiosity_model):
        device = policy.device
        dtype = policy.dtype

        # plan for rollout
        n_envs = 10
        max_steps = 400


        # pdb.set_trace()
        obs_to_diff = []
        ep_to_rewards = {}
        label_to_obs = {}
        extended_rollout_data = []

        for rollout_ep in range(n_envs):
            env = MultiStepWrapper(
                Push2dKeypointsEnv(
                keypoint_visible_rate=1.0, 
                agent_keypoints=self.agent_keypoints,
                render_size=96,
                render_action=False
            ),
            n_obs_steps=self.env_n_obs_steps,
            n_action_steps=self.env_n_action_steps,
            max_episode_steps=max_steps
            )

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()
            # move policy to device
            policy.to(device)

            log_data = []

            done = False
            rewards = []
            action_mode = 'robot'
            agent = None

            action_diffs_over_episode = []
            state_diffs_over_episode = []
            diffs_over_episode = []
            imgs_over_episode = []
            timestep = -1
            timestep_to_rollout_obs = {}
            queried_uncertainty = False
            diff = 0
            state_diff = 0
            action_diff = 0

            while not done:
                timestep += 1
                Do = obs.shape[-1] // 2
                # print("obs.shape", obs.shape)   
                # pdb.set_trace()

                obs = np.expand_dims(obs, axis=0)


                # create obs dict
                np_obs_dict = {
                    # handle n_latency_steps by discarding the last n_latency_steps
                    'obs': obs[...,:self.n_obs_steps,:Do].astype(np.float32),
                    'obs_mask': obs[...,:self.n_obs_steps,Do:] > 0.5,
                }
                first_obs = np_obs_dict['obs']


                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))
                # pdb.set_trace()
                
                if action_mode == 'robot':
                    # run policy
                    with torch.no_grad():
                        # pdb.set_trace()
                        action_dict = policy.predict_action(obs_dict)

                    # device_transfer
                    np_action_dict = dict_apply(action_dict,
                        lambda x: x.detach().to('cpu').numpy())

                    # handle latency_steps, we discard the first n_latency_steps actions
                    # to simulate latency
                    action = np_action_dict['action'][:,self.n_latency_steps:]
                    action = action[0]
                    obs, reward, done, info = env.step(action)
                    # visualize
                    img = env.render(mode='human')

                else:
                    action = agent.act(obs)
                    obs, reward, done, info = env.step_with_one_action(action)
                    # visualize
                    img = env.render(mode='human')
                    if not action is None:
                        # teleop started
                        # state dim 2+3
                        # state = np.concatenate([info['pos_agent'], info['block_pose'], info['goal_pose']])
                        # pdb.set_trace()
                        state = np.concatenate([info['pos_agent'][1], info['block_pose'][1], 
                                                info['block_angle'][1],info['goal_pose1'][1], info['goal_pose2'][1]])
                        # state = np.concatenate([info['pos_agent']])
                        # discard unused information such as visibility mask and agent pos
                        # for compatibility
                        keypoint = obs[1].reshape(2, -1)[0].reshape(-1, 2)[:9]
                        # import pdb; pdb.set_trace()
                        # print("act", act)
                        data = {
                            'img': img,
                            'state': np.float32(state),
                            'keypoint': np.float32(keypoint),
                            'action': np.float32(action),
                            'n_contacts': np.float32([info['n_contacts']])
                        }
                        recovery_episode.append(data)

                    # pause time for control_hz
                    time.sleep(1.0 / 10)
                    # clock.tick(control_hz)

                    if done:
                        # save episode buffer to replay buffer (on disk)
                        data_dict = dict()
                        if len(recovery_episode) > 0:
                            for key in recovery_episode[0].keys():
                                data_dict[key] = np.stack(
                                    [x[key] for x in recovery_episode])
                            recovery_replay_buffer.add_episode(data_dict, compressors='disk')
                            print(f'saved rollout recovery for epiosde {rollout_ep}')
                    


                
                rewards.append(reward)

                

                # convert img to numpy rgb image
                img = np.array(img)

                imgs_over_episode.append(img)
                
                second_obs = obs[...,:self.n_obs_steps,:Do].astype(np.float32)

                timestep_to_rollout_obs[timestep] = (first_obs, second_obs, np_action_dict['action'])


                correct_action = np_action_dict['action'][:,self.n_latency_steps:]

                extended_rollout_data.append((first_obs, second_obs, correct_action))   

                # normalize observations
                first_obs = policy.normalizer['obs'].normalize(first_obs)
                second_obs = policy.normalizer['obs'].normalize(second_obs)
                correct_action = policy.normalizer['action'].normalize(correct_action)


                # convert all to torch
                # first_obs = torch.from_numpy(first_obs).to(device)
                # second_obs = torch.from_numpy(second_obs).to(device)
                # correct_action = torch.from_numpy(correct_action).to(device)
                first_obs = first_obs.to(device)
                second_obs = second_obs.to(device)
                correct_action = correct_action.to(device)

                batch_idx = 0
                first_obs_batch = first_obs[batch_idx][1]
                second_obs_batch = second_obs[1]
                input_x = torch.cat([first_obs_batch, second_obs_batch], dim=0)
                input_x_flat = input_x.flatten() # 80
                target_y = correct_action[batch_idx].flatten() # 16

                # move both to device
                # first_obs_batch = first_obs_batch.to(device)
                # second_obs_batch = second_obs_batch.to(device)
                input_x_flat = input_x_flat.to(device)
                target_y = target_y.to(device)

                
                if action_mode == 'robot':
                    # split input_x_flat to first_obs and second_obs
                    first_obs_flat = input_x_flat[:20]
                    second_obs_flat = input_x_flat[20:]

                    # unsqueeze to add batch dim
                    first_obs_flat = first_obs_flat.unsqueeze(0)
                    second_obs_flat = second_obs_flat.unsqueeze(0)


                    # predict model
                    pred_y = curiosity_model.inverse_dynamics(first_obs_flat, second_obs_flat)

                    # unnormalize pred_y and target_y
                    pred_y = policy.normalizer['action'].unnormalize(pred_y)[0]
                    target_y = policy.normalizer['action'].unnormalize(target_y)
                    # pdb.set_trace()
                    action_diff = np.linalg.norm(pred_y.cpu().detach().numpy() - target_y.cpu().detach().numpy())
                    # pdb.set_trace()
                    # unsqueeze target_y
                    unnorm_target_y = correct_action[batch_idx].flatten()
                    unnorm_target_y = unnorm_target_y.unsqueeze(0)
                    second_obs_embed = curiosity_model.encode_obs(second_obs_flat)[0]
                    pred_obs_t1_embed = curiosity_model.forward_dynamics(first_obs_flat, unnorm_target_y)[0]
                    # pdb.set_trace()
                    state_diff = np.linalg.norm(pred_obs_t1_embed.cpu().detach().numpy() - second_obs_embed.cpu().detach().numpy())
                    diff = action_diff + state_diff
                    action_diffs_over_episode.append(action_diff)
                    state_diffs_over_episode.append(state_diff)


                    # plot image with 512x512
                    img_plot = Image.fromarray(img)
                    img_plot = img_plot.resize((512, 512))
                    img_plot = np.array(img_plot)

                    # overlay with opacity previous image
                    if len(imgs_over_episode) > 2:
                        prev_img = imgs_over_episode[-2]
                        prev_img = Image.fromarray(prev_img)
                        prev_img = prev_img.resize((512, 512))
                        prev_img = np.array(prev_img)
                        img_plot = (img_plot * 0.5 + prev_img * 0.5).astype(np.uint8)

                    # plot on image
                    plt.imshow(img_plot)

                    # plt.imshow(img_plot)
                    # reshape to 8, 2
                    pred_action = pred_y.cpu().detach().numpy().reshape(-1, 2)
                    target_action = target_y.cpu().detach().numpy().reshape(-1, 2)
                    # plot on image
                    plt.plot(pred_action[:,0], pred_action[:,1], 'ro')
                    plt.plot(target_action[:,0], target_action[:,1], 'go')
                    plt.legend(['pred', 'target'])
                    plt.title(f"diff: {diff}")
                    plt.savefig(pathlib.Path(self.episode_save_folder) / f"ep_{rollout_ep}_diff_{timestep}.png")
                    plt.close()

                    diffs_over_episode.append(diff)
                    print("diff", diff)
                    # pdb.set_trace()
                    obs_to_diff.append((first_obs_batch, diff, info))

                if (state_diff > 1 or action_diff > 150) and queried_uncertainty is False and timestep > 1:
                    queried_uncertainty = True
                    time_labels, response_mode = plot_interactive(imgs_over_episode, state_diffs_over_episode, action_diffs_over_episode)
                    print("time_labels", time_labels)
                    print("response_mode", response_mode)
                    
                    for t, label in time_labels.items():
                        if label.lower() not in label_to_obs:
                            label_to_obs[label.lower()] = []
                        label_to_obs[label.lower()].append(timestep_to_rollout_obs[t])

                    if response_mode == 'avoid':
                        break

                    if response_mode == 'recover':
                        # create a zarr file in savetag folder
                        output = pathlib.Path(self.savetag_folder) / f"human_recovery.zarr"
                        recovery_replay_buffer = ReplayBuffer.create_from_path(output, mode='a')
                        recovery_episode = list()
                        action_mode = 'human'
                        agent = env.teleop_agent()
                        control_hz = 10
                        # clock = pygame.time.Clock()

                

                done = np.all(done)
                past_action = action
            
            ep_to_rewards[f'reward_{rollout_ep}'] = max(rewards)
            if queried_uncertainty is False:
                time_labels, response_mode = plot_interactive(imgs_over_episode, diffs_over_episode)
                for t, label in time_labels.items():
                    if label.lower() not in label_to_obs:
                        label_to_obs[label.lower()] = []
                    label_to_obs[label.lower()].append(timestep_to_rollout_obs[t])
            env.close()

            

        log_data = ep_to_rewards

        return log_data, obs_to_diff, label_to_obs, extended_rollout_data
