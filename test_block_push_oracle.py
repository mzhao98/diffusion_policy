import numpy as np
import click
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.env.block_pushing.block_pushing import BlockPush
import pygame
from diffusion_policy.env.block_pushing.oracles.oriented_push_oracle import OrientedPushOracle
import pdb
import time
from tf_agents.environments import suite_gym
from diffusion_policy.env import block_pushing
# from diffusion_policy.env import block_pushing_multimodal
import gym
from diffusion_policy.common.replay_buffer import ReplayBuffer

def main():
    replay_buffer_file_to_save = 'blockpush_simple.zarr'


    # Create an instance of your registered environment.
    env = gym.make("BlockPush-v0")

    # Wrap the environment for TF‑Agents to automatically set up specs.
    tf_env = suite_gym.wrap_env(env)

    # You can now access the time_step_spec and action_spec.
    print("Time step spec:", tf_env.time_step_spec())
    print("Action spec:", tf_env.action_spec())

    # Create oracle
    time_step = tf_env.reset()
    oracle = OrientedPushOracle(tf_env)
    policy_state = oracle.get_initial_state()
    done = False
    tf_env.render(mode='human')
    while not done:
        # pdb.set_trace()
        policy_step = oracle._action(time_step, policy_state)
        time_step = tf_env.step(policy_step.action)
        tf_env.render(mode='human')
        done = time_step.is_last()
        time.sleep(0.1)

    


if __name__ == "__main__":
    main()