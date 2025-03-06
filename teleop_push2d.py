import numpy as np
import click
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.env.push2d.push2d_keypoints_env import Push2dKeypointsEnv
import pygame


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

    # create PushT environments with keypoints
    kp_kwargs = Push2dKeypointsEnv.genenerate_keypoint_manager_params()
    env = Push2dKeypointsEnv(render_size=render_size, render_action=False, **kp_kwargs)
    agent = env.teleop_agent()
    clock = pygame.time.Clock()
    output_for_lang = output.split('.zarr')[0] + '_lang.pkl'

    # episode-level while loop
    lang_commands = {}
    while True:
        episode = list()
        # record in seed order, starting with 0
        seed = replay_buffer.n_episodes
        # seed = 0
        print(f'starting seed {seed}')
        utterance = input("Please input the command: ")
        lang_commands[seed] = utterance

        # set seed for environments
        env.seed(seed)

        # reset environments and get observations (including info and render for recording)
        obs = env.reset()
        info = env._get_info()
        img = env.render(mode='human')

        # loop state
        retry = False
        pause = False
        done = False
        plan_idx = 0
        pygame.display.set_caption(f'plan_idx:{plan_idx}')
        # step-level while loop
        while not done:
            # process keypress events
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        # hold Space to pause
                        plan_idx += 1
                        pygame.display.set_caption(f'plan_idx:{plan_idx}')
                        pause = True
                    elif event.key == pygame.K_r:
                        # press "R" to retry
                        retry = True
                    elif event.key == pygame.K_q:
                        # press "Q" to exit
                        exit(0)
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_SPACE:
                        pause = False

            # handle control flow
            if retry:
                break
            if pause:
                continue

            # get action from mouse
            # None if mouse is not close to the agent
            act = agent.act(obs)
            if not act is None:
                # teleop started
                # state dim 2+3
                # state = np.concatenate([info['pos_agent'], info['block_pose'], info['goal_pose']])
                state = np.concatenate([info['pos_agent'], info['block_pose'], info['block_angle'],
                                        info['goal_pose1'], info['goal_pose2']])
                # state = np.concatenate([info['pos_agent']])
                # discard unused information such as visibility mask and agent pos
                # for compatibility
                keypoint = obs.reshape(2, -1)[0].reshape(-1, 2)[:9]
                # import pdb; pdb.set_trace()
                # print("act", act)
                data = {
                    'img': img,
                    'state': np.float32(state),
                    'keypoint': np.float32(keypoint),
                    'action': np.float32(act),
                    'n_contacts': np.float32([info['n_contacts']])
                }
                episode.append(data)

            # step environments and render
            obs, reward, done, info = env.step(act)
            img = env.render(mode='human')

            # regulate control frequency
            clock.tick(control_hz)
        if not retry:
            # save episode buffer to replay buffer (on disk)
            data_dict = dict()
            for key in episode[0].keys():
                data_dict[key] = np.stack(
                    [x[key] for x in episode])
            replay_buffer.add_episode(data_dict, compressors='disk')
            print(f'saved seed {seed}')
            # save language commands
            import pickle
            with open(output_for_lang, 'wb') as f:
                pickle.dump(lang_commands, f)
            print(f'saved lang commands {output_for_lang}')
        else:
            print(f'retry seed {seed}')


if __name__ == "__main__":
    main()