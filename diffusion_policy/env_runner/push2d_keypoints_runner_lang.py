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
# from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
import pdb
import matplotlib.pyplot as plt
import cv2
from scipy.stats import gaussian_kde
from scipy.interpolate import griddata
import pickle
import itertools
from transformers import BertTokenizer, BertModel

class Push2dKeypointsDistributionLangRunner(BaseLowdimRunner):
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

        # handle latency step
        # to mimic latency, we request n_latency_steps additional steps 
        # of past observations, and the discard the last n_latency_steps
        env_n_obs_steps = n_obs_steps + n_latency_steps
        env_n_action_steps = n_action_steps

        # assert n_obs_steps <= n_action_steps
        kp_kwargs = Push2dKeypointsEnv.genenerate_keypoint_manager_params()

        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    Push2dKeypointsEnv(
                        legacy=legacy_test,
                        keypoint_visible_rate=keypoint_visible_rate,
                        agent_keypoints=agent_keypoints,
                        **kp_kwargs
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                ),
                n_obs_steps=env_n_obs_steps,
                n_action_steps=env_n_action_steps,
                max_episode_steps=max_steps
            )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()
        # train
        for i in range(n_train):
            seed = train_start_seed + i
            enable_render = i < n_train_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)
            
            env_seeds.append(seed)
            env_prefixs.append('train/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = SyncVectorEnv(env_fns)

        # test env
        # env.reset(seed=env_seeds)
        # x = env.step(env.action_space.sample())
        # imgs = env.call('render')
        # import pdb; pdb.set_trace()

        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.agent_keypoints = agent_keypoints
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.n_latency_steps = n_latency_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.strategy_mode = 0
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.context_encoder = BertModel.from_pretrained("bert-base-uncased",output_hidden_states = True)
        # get dimension of the hidden states
        self.dbert = 768
        self.set_up_context_encoder()

    def set_replay_buffer(self, dataset, train_dataloader):
        self.dataset = dataset
        self.replay_buffer = dataset.replay_buffer
        self.train_dataloader = train_dataloader

    def get_dataloader_item(self, idx):
        sample_at = idx
        k = int(np.floor(sample_at))
        my_sample = next(itertools.islice(self.train_dataloader, k, None))
        return my_sample

    def set_up_context_encoder(self):
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

    def get_utterance_embedding(self, utterance):
        tokenized_inputs = self.tokenizer(utterance, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad(): 
            outputs = self.context_encoder(**tokenized_inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
        # store the embeddings
        self.context_lookup[utterance] = embeddings[0].cpu().numpy()
        return embeddings[0].cpu().numpy()

    def set_mode(self, mode):
        self.strategy_mode = mode

    def render_obs(self, env, obs, action):
        block_position = env.envs[0].env.env.block.position
        block_x = block_position[0]
        block_y = block_position[1]
        agent_position = env.envs[0].env.env.agent.position
        agent_x = agent_position[0]
        agent_y = agent_position[1]
        goal_pose1 = env.envs[0].env.env.goal_pose1
        goal_pose2 = env.envs[0].env.env.goal_pose2
        goal_pose1_x = goal_pose1[0]
        goal_pose1_y = goal_pose1[1]
        goal_pose2_x = goal_pose2[0]
        goal_pose2_y = goal_pose2[1]
        plt.figure(figsize=(5,5))
        plt.xlim(0,450)
        plt.ylim(0,450)
        # draw a square for the block
        plt.plot(block_x, block_y, 's', color='gray')
        # draw a circle for the agent
        plt.plot(agent_x, agent_y, 'o', color='red')
        # draw a green square for the goal pose 1
        plt.plot(goal_pose1_x, goal_pose1_y, 's', color='green')
        # draw a green square for the goal pose 2
        plt.plot(goal_pose2_x, goal_pose2_y, 's', color='green')
        plt.show()

    def render_obs_with_samples(self, env, obs, action, list_of_sampled_actions, iter_idx, list_of_sampled_likelihoods):
        # Compute likelihoods
        # plot likelihoods
        # take softmax of the log likelihoods in list_of_sampled_likelihoods
        list_of_sampled_likelihoods = np.array(list_of_sampled_likelihoods)
        # list_of_sampled_likelihoods = np.exp(list_of_sampled_likelihoods)
        # normalize
        # list_of_sampled_likelihoods = list_of_sampled_likelihoods / np.sum(list_of_sampled_likelihoods, axis=0)
        # compute variance of sampled actions
        std_action_preds = np.std(list_of_sampled_actions, axis=0)[0]
        # pdb.set_trace()

        block_position = env.envs[0].env.env.block.position
        block_x = block_position[0]
        block_y = block_position[1]
        agent_position = env.envs[0].env.env.agent.position
        agent_x = agent_position[0]
        agent_y = agent_position[1]
        goal_pose1 = env.envs[0].env.env.goal_pose1
        goal_pose2 = env.envs[0].env.env.goal_pose2
        goal_pose1_x = goal_pose1[0]
        goal_pose1_y = goal_pose1[1]
        goal_pose2_x = goal_pose2[0]
        goal_pose2_y = goal_pose2[1]

        # flip over the x axis
        block_y = 450 - block_y
        agent_y = 450 - agent_y
        goal_pose1_y = 450 - goal_pose1_y
        goal_pose2_y = 450 - goal_pose2_y


        plt.figure(figsize=(5,5))
        plt.xlim(0,450)
        plt.ylim(0,450)
        # draw a green square for the goal pose 1
        plt.plot(goal_pose1_x, goal_pose1_y, 's', color='green', markersize=30)
        # draw a green square for the goal pose 2
        plt.plot(goal_pose2_x, goal_pose2_y, 's', color='green', markersize=30)

        # draw a square for the block
        plt.plot(block_x, block_y, 's', color='gray', markersize=30)
        # draw a circle for the agent
        plt.plot(agent_x, agent_y, 'o', color='red', markersize=20)

        # for the correct action, plot in red, with thick line
        points_x = []
        points_y = []
        for i in range(action.shape[1]):
            # flip over y = 225 horizontal line
            action_x = action[0][i,0]
            action_y = 450 - action[0][i,1]
            points_x.append(action_x)
            points_y.append(action_y)
        plt.plot(points_x, points_y, color='red', linewidth=10, alpha=0.3)
        
        # setup the colors
        sample_idx_to_color = {}
        sample_idx_to_color[len(list_of_sampled_actions)] = 'red'
        # sample 6 other colors via 'hex
        colors = ['yellow', 'blue', 'purple', 'orange', 'pink', 'green']


        # for action sample in list_of_sampled_actions, draw a line between the points
        
        for sample_idx in range(len(list_of_sampled_actions)):
            action_sample = list_of_sampled_actions[sample_idx]
            points_x = []
            points_y = []
            # for i in range(action_sample.shape[1]):
            for i in range(0,8):
                # flip over y = 225 horizontal line
                action_sample_x = action_sample[0][i,0]
                action_sample_y = 450 - action_sample[0][i,1]

                # points_x.append(action_sample[0][i,0])
                # points_y.append(action_sample[0][i,1])
                points_x.append(action_sample_x)
                points_y.append(action_sample_y)
            # draw a line between the points, randomize colors
            rand_color = colors[sample_idx]
            plt.plot(points_x, points_y, color=rand_color, marker='o')
            sample_idx_to_color[sample_idx] = rand_color

        # for the first action in each sample, add to a list
        points_xy_samples = {}
        for i in range(list_of_sampled_actions[0].shape[1]):
            points_xy_samples[i] = []
            for action_sample in list_of_sampled_actions:
                # pdb.set_trace()
                # flip over x-axis
                action_sample_x = action_sample[0][i,0]
                action_sample_y = 450 - action_sample[0][i,1]
                points_xy_samples[i].append([action_sample_x, action_sample_y])
                
                # points_xy_samples[i].append([action_sample[0][i,0], action_sample[0][i,1]])

        plt.savefig(f'distribution_viz/gameboard_{iter_idx}.png')
        plt.show()

        # plot likelihoods with color of bar representing the likelihood
        plt.figure(figsize=(5,5))
        plt.bar(range(len(list_of_sampled_likelihoods)), list_of_sampled_likelihoods, color=[sample_idx_to_color[i] 
                                                                                             for i in range(len(list_of_sampled_likelihoods))])
        plt.title(f"Likelihoods of sampled actions: t={iter_idx}")
        plt.xlabel("Sampled action")
        plt.ylabel("Likelihood")
        plt.savefig(f'distribution_viz/likelihoods_{iter_idx}.png')
        plt.show()        

        # plot the variance of the actions
        # std_action_preds is 8, 2
        plt.figure(figsize=(5,5))
        # because the variance is 2d, plot as heatmap
        plt.imshow(std_action_preds.T, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title(f"Variance of sampled actions: t={iter_idx}")
        plt.xlabel("Timestep")
        plt.ylabel("Action dimension")
        plt.savefig(f'distribution_viz/variance_{iter_idx}.png')
        plt.show()

        # subtimestep_to_kde = {}
        # # plot density manifold
        # for key in points_xy_samples:
        #     points_xy = points_xy_samples[key]
        #     kde = plot_density_manifold(points_xy, iter_idx, key)
        #     subtimestep_to_kde[key] = kde

        return


    def run_with_target(self, policy: BaseLowdimPolicy):
        device = policy.device
        dtype = policy.dtype

        env = self.env

        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        # pdb.set_trace()
        observation_to_kde = {}

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            
            past_action = None
            policy.reset()

            # pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval Push2dKeypointsRunner {chunk_idx+1}/{n_chunks}", 
            #     leave=False, mininterval=self.tqdm_interval_sec)
            done = False
            iter_idx = 0

            utterance = input("Enter the utterance: ")
            self.get_utterance_embedding(utterance)

            # for buffer_idx in range(replay_buffer.n_episodes):
            buffer_idx = 0
            episode_to_replay = self.replay_buffer.get_episode(buffer_idx)
            # get buffer idx start and end
            if buffer_idx == 0:
                buffer_idx_start = 0
            else:
                buffer_idx_start = self.replay_buffer.episode_ends[buffer_idx-1]
            buffer_idx_end = self.replay_buffer.episode_ends[buffer_idx]

            # get seeds
            # pdb.set_trace()
            # seed = episode_to_replay['seed']
            obs = env.seed(buffer_idx)
            obs = env.reset()
            img = env.render(mode='human')
            # pdb.set_trace()

            for t in range(buffer_idx_start, buffer_idx_end, 8):
                
                obs_to_replay = self.dataset[t]['obs'][:self.n_obs_steps,:]
                obs_to_replay = obs_to_replay.unsqueeze(0) # 1, 16, 20
                # print("obs_to_replay", obs_to_replay.shape)
                # obs_to_replay = self.get_dataloader_item(t)['obs'][:,:self.n_obs_steps,:]
                # pdb.set_trace()
                # env._set_state(state_to_replay)
                img = env.render(mode='human')
                # obs = env._get_obs()
                # pdb.set_trace()

                iter_idx += 1
                # render obs
                # self.render_obs(env, obs, past_action)
                Do = obs.shape[-1] // 2
                # create obs dict
                np_obs_dict = {
                    # handle n_latency_steps by discarding the last n_latency_steps
                    'obs': obs[...,:self.n_obs_steps,:Do].astype(np.float32),
                    'obs_mask': obs[...,:self.n_obs_steps,Do:] > 0.5,
                    'idx': np.ones((obs[...,:self.n_obs_steps,:Do].shape[0], 1), dtype=np.float32) * self.strategy_mode,
                }
                # assert torch.sum(obs_to_replay-obs[...,:self.n_obs_steps,:Do]) < 0.001
                # diff = torch.sum(obs_to_replay-obs[...,:self.n_obs_steps,:Do])
                # pdb.set_trace()
                # print("diff",diff)
                np_obs_dict['obs'] = obs_to_replay.numpy().astype(np.float32)
                # print(" obs[...,:self.n_obs_steps,:Do]",  obs[...,:self.n_obs_steps,:Do].shape)
                conditioning_context = np.zeros((obs.shape[0], self.dbert))
                for i in range(obs.shape[0]):
                    conditioning_context[i] = self.context_lookup[utterance]
                np_obs_dict['idx'] = conditioning_context.astype(np.float32)

                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))

                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                # sample K actions, plot them as lines on a plot
                K = 6
                list_of_sampled_actions = []
                list_of_sampled_likelihoods = []
                for k in range(K):
                    action_sample_dict = policy.predict_action(obs_dict)
                    obs_dict['action'] = action_sample_dict['action']
                    # drop obs mask
                    if 'obs_mask' in obs_dict:
                        del obs_dict['obs_mask']
                    action_likelihood = policy.get_log_likelihood(obs_dict)
                    # get action likelihood of first item
                    action_likelihood = action_likelihood[0][0].detach().to('cpu').numpy()
                    action_likelihood = np.sum(action_likelihood, axis=0)
                    # print("action_likelihood", action_likelihood)
                    list_of_sampled_likelihoods.append(action_likelihood)

                    np_action_sample_dict = dict_apply(action_sample_dict,lambda x: x.detach().to('cpu').numpy())
                    action_sample = np_action_sample_dict['action'][:,self.n_latency_steps:]
                    list_of_sampled_actions.append(action_sample)

                


                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                # handle latency_steps, we discard the first n_latency_steps actions
                # to simulate latency
                action = np_action_dict['action'][:,self.n_latency_steps:]
                current_obs = tuple(obs_dict['obs'][:,-1].cpu().numpy()[0])

                # ground truth action
                gt_action = self.dataset[t]['action'][:self.n_action_steps,self.n_latency_steps:] # 1, 16, 2
                # gt_action = self.get_dataloader_item(t)['action'][:,self.n_latency_steps:,:]
                gt_action = np.expand_dims(gt_action, axis=0)
                print("action", action.shape)
                print("gt_action", gt_action.shape)
                action = gt_action
                # get likelihood of gt action
                obs_dict['action'] = torch.from_numpy(action).to(device=device)
                action_likelihood = policy.get_log_likelihood(obs_dict)
                # get action likelihood of first item
                action_likelihood = action_likelihood[0][0].detach().to('cpu').numpy()
                action_likelihood = np.sum(action_likelihood, axis=0)
                # print("action_likelihood", action_likelihood)
                list_of_sampled_likelihoods.append(action_likelihood)


                # compute divergence between the sampled actions and the ground truth action
                mean_sampled_action = np.mean(list_of_sampled_actions, axis=0)
                divergence = np.mean(np.sqrt((mean_sampled_action - gt_action)**2))
                print("Divergence between sampled actions and ground truth action: ", divergence)
                if divergence > 50:
                    print("High divergence detected, check the distribution of actions")
                    # utterance = input("Enter a new utterance: ")
                    # check if 'right' would result in lower divergence
                    candidate_utterance = 'right'
                    candidate_embedding = self.context_lookup[candidate_utterance]
                    np_obs_dict['idx'] = candidate_embedding.astype(np.float32)
                    # unsqueeze to add batch dimension
                    # to torch
                    # obs_dict['idx'] = torch.from_numpy(obs_dict['idx']).to(device=device)
                    # unsqueeze to add batch dimension
                    # obs_dict['idx'] = obs_dict['idx'].unsqueeze(0)
                    # add dimension
                    np_obs_dict['idx'] = np.expand_dims(np_obs_dict['idx'], axis=0)
                    obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))
                    # pdb.set_trace()
                    candidate_pred_action = policy.predict_action(obs_dict)
                    candidate_pred_action = candidate_pred_action['action'][:,self.n_latency_steps:].detach().to('cpu').numpy()
                    candidate_divergence_right = np.mean(np.sqrt((candidate_pred_action - gt_action)**2))
                    print("candidate_divergence_right", candidate_divergence_right)

                    # check if 'left' would result in lower divergence
                    candidate_utterance = 'left'
                    candidate_embedding = self.context_lookup[candidate_utterance]
                    np_obs_dict['idx'] = candidate_embedding.astype(np.float32)
                    # obs_dict['idx'] = torch.from_numpy(obs_dict['idx']).to(device=device)
                    # obs_dict['idx'] = obs_dict['idx'].unsqueeze(0)
                    np_obs_dict['idx'] = np.expand_dims(np_obs_dict['idx'], axis=0)
                    obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))
                    candidate_pred_action = policy.predict_action(obs_dict)
                    candidate_pred_action = candidate_pred_action['action'][:,self.n_latency_steps:].detach().to('cpu').numpy()
                    candidate_divergence_left = np.mean(np.sqrt((candidate_pred_action - gt_action)**2))
                    print("candidate_divergence_left", candidate_divergence_left)

                    candidates_to_present = []
                    if candidate_divergence_right < divergence:
                        candidates_to_present.append('right')
                    if candidate_divergence_left < divergence:
                        candidates_to_present.append('left')

                    print("Candidates to present: ", candidates_to_present)
                    utterance = input(f"These candidates would result in lower divergence: {candidates_to_present}. Enter a new utterance: ")


                    self.get_utterance_embedding(utterance)

                # step env
                # obs, reward, done, info = env.step(action)
                
                # img = env.envs[0].env.env._render_frame(mode='human')
                # pdb.set_trace()
                subtimestep_to_kde = self.render_obs_with_samples(env, obs, action, 
                                                                  list_of_sampled_actions, iter_idx, 
                                                                  list_of_sampled_likelihoods)
                obs, reward, done, info = env.step(action)
                img = env.envs[0].env.env._render_frame(mode='human')
                
                
                # compute variance of list of sampled likelihoods
                # variance = np.std(list_of_sampled_likelihoods)
                # print("Variance of likelihoods: ", variance)
                # if variance > 0.1:
                #     print("High variance detected, check the distribution of actions")
                #     utterance = input("Enter a new utterance: ")
                #     self.get_utterance_embedding(utterance)
                # observation_to_kde[current_obs] = subtimestep_to_kde
                # cv2.imshow('image', img)
                done = np.all(done)
                past_action = action
                print("done", done)
                if done:
                    img = env.envs[0].env.env._render_frame(mode='human')
                    break

                # update pbar
            #     pbar.update(action.shape[1])
            # pbar.close()

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]



        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data

    
    def run(self, policy: BaseLowdimPolicy):
        device = policy.device
        dtype = policy.dtype

        env = self.env

        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        # pdb.set_trace()
        observation_to_kde = {}

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()

            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval Push2dKeypointsRunner {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            done = False
            iter_idx = 0

            utterance = input("Enter the utterance: ")
            self.get_utterance_embedding(utterance)

            while not done:
                iter_idx += 1
                # render obs
                # self.render_obs(env, obs, past_action)
                Do = obs.shape[-1] // 2
                # create obs dict
                np_obs_dict = {
                    # handle n_latency_steps by discarding the last n_latency_steps
                    'obs': obs[...,:self.n_obs_steps,:Do].astype(np.float32),
                    'obs_mask': obs[...,:self.n_obs_steps,Do:] > 0.5,
                    'idx': np.ones((obs[...,:self.n_obs_steps,:Do].shape[0], 1), dtype=np.float32) * self.strategy_mode,
                }

                conditioning_context = np.zeros((obs.shape[0], self.dbert))
                for i in range(obs.shape[0]):
                    conditioning_context[i] = self.context_lookup[utterance]
                np_obs_dict['idx'] = conditioning_context.astype(np.float32)

                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))

                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                # sample K actions, plot them as lines on a plot
                K = 2
                list_of_sampled_actions = []
                list_of_sampled_likelihoods = []
                for k in range(K):
                    action_sample_dict = policy.predict_action(obs_dict)
                    obs_dict['action'] = action_sample_dict['action']
                    # drop obs mask
                    if 'obs_mask' in obs_dict:
                        del obs_dict['obs_mask']
                    action_likelihood = policy.get_log_likelihood(obs_dict)
                    # get action likelihood of first item
                    action_likelihood = action_likelihood[0][0].detach().to('cpu').numpy()
                    action_likelihood = np.sum(action_likelihood, axis=0)
                    # print("action_likelihood", action_likelihood)
                    list_of_sampled_likelihoods.append(action_likelihood)

                    np_action_sample_dict = dict_apply(action_sample_dict,lambda x: x.detach().to('cpu').numpy())
                    action_sample = np_action_sample_dict['action'][:,self.n_latency_steps:]
                    list_of_sampled_actions.append(action_sample)


                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                # handle latency_steps, we discard the first n_latency_steps actions
                # to simulate latency
                action = np_action_dict['action'][:,self.n_latency_steps:]
                current_obs = tuple(obs_dict['obs'][:,-1].cpu().numpy()[0])
                # pdb.set_trace()

                # step env
                # obs, reward, done, info = env.step(action)
                
                # img = env.envs[0].env.env._render_frame(mode='human')
                subtimestep_to_kde = self.render_obs_with_samples(env, obs, action, list_of_sampled_actions, iter_idx, list_of_sampled_likelihoods)
                obs, reward, done, info = env.step(action)
                
                # compute variance of list of sampled likelihoods
                variance = np.std(list_of_sampled_likelihoods)
                print("Variance of likelihoods: ", variance)
                if variance > 0.1:
                    print("High variance detected, check the distribution of actions")
                    utterance = input("Enter a new utterance: ")
                    self.get_utterance_embedding(utterance)
                # observation_to_kde[current_obs] = subtimestep_to_kde
                # cv2.imshow('image', img)
                done = np.all(done)
                past_action = action

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]



        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data


def plot_density_manifold(data, iter_idx, subtimestep, grid_size=100, cmap='coolwarm'):
    
    """
    Plots a density manifold from a given Nx2 list of points.

    Parameters:
        data (list of list): An Nx2 list of points (x, y).
        grid_size (int): The resolution of the density grid. Default is 100.
        cmap (str): The colormap for the plot. Default is 'coolwarm'.
    """
    # Convert the list into numpy arrays for easier handling
    points = np.array(data)
    # pdb.set_trace()
    x = points[:, 0]
    y = points[:, 1]

    # Perform kernel density estimation for the data
    kde = gaussian_kde(points.T)
    # pdb.set_trace()

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
    plt.scatter(x, y, color='black', label='Original Points')
    plt.legend()
    plt.title(f"Density Manifold Plot: t={iter_idx}_subt={subtimestep}")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.savefig(f'distribution_viz/density_manifold_{iter_idx}_subt_{subtimestep}.png')
    plt.close()
    return kde

