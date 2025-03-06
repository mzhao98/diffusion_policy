from gym.envs.registration import register
import diffusion_policy.env.push2d

register(
    id='push2d-keypoints-v0',
    entry_point='envs.push2d.push2d_keypoints_env:Push2dKeypointsEnv',
    max_episode_steps=200,
    reward_threshold=1.0
)