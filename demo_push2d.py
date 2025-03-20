import numpy as np
import click
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.env.push2d.push2d_keypoints_env import Push2dKeypointsEnv
import pygame
import pdb

@click.command()
@click.option('-o', '--output', required=True)
@click.option('-rs', '--render_size', default=96, type=int)
@click.option('-hz', '--control_hz', default=10, type=int)
def main(output, render_size, control_hz):
    """
    Collect demonstration for the Push-T task.
    
    Usage: python demo_pusht.py -o data/pusht_demo.zarr
    
    This script is compatible with both Linux and MacOS.
    Hover mouse close to the blue circle to start.
    Push the T block into the green area. 
    The episode will automatically terminate if the task is succeeded.
    Press "Q" to exit.
    Press "R" to retry.
    Hold "Space" to pause.
    """
    
    # create replay buffer in read-write mode
    replay_buffer = ReplayBuffer.create_from_path(output, mode='a')

    # create PushT env with keypoints
    kp_kwargs = Push2dKeypointsEnv.genenerate_keypoint_manager_params()
    env = Push2dKeypointsEnv(render_size=render_size, render_action=False, **kp_kwargs)
    agent = env.teleop_agent()
    clock = pygame.time.Clock()

    buffer_idx = int(input("Enter buffer index: "))
    
    print("num eps", replay_buffer.n_episodes)
    # episode-level while loop
    # for buffer_idx in (range(replay_buffer.n_episodes)):
    for buffer_idx in range(buffer_idx, buffer_idx+1):
        # record in seed order, starting with 0
        # seed = replay_buffer.n_episodes
        # buffer_idx = 0
        # print(f'starting seed {seed}')

        # set seed for env
        # env.seed(seed)
        # get seed for env from replay buffer at index buffer_idx
        episode_to_replay = replay_buffer.get_episode(buffer_idx)
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
        
        # loop state
        retry = False
        pause = False
        done = False
        # plan_idx = 0
        pygame.display.set_caption(f'plan_idx:{buffer_idx}')
        # step-level while loop
        for t in range(1, len(episode_to_replay['action'])):
            
            state_to_replay = episode_to_replay['state'][t]
            # remove the last element of the state_to_replay
            state_to_replay = state_to_replay[:-1]
            # remove third to last element of the state_to_replay
            state_to_replay = list(state_to_replay[:-3]) + list(state_to_replay[-2:])
            env._set_state(state_to_replay)
            img = env.render(mode='human')
            # regulate control frequency
            clock.tick(control_hz)


if __name__ == "__main__":
    main()
