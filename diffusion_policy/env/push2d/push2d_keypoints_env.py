from typing import Dict, Sequence, Union, Optional
from gym import spaces
from diffusion_policy.env.push2d.push2d_env import Push2dEnv
from diffusion_policy.env.push2d.pymunk_keypoint_manager import PymunkKeypointManager
import numpy as np
import pdb

class Push2dKeypointsEnv(Push2dEnv):
    def __init__(self,
            legacy=False,
            block_cog=None, 
            damping=None,
            render_size=96,
            keypoint_visible_rate=1.0, 
            agent_keypoints=False,
            draw_keypoints=False,
            reset_to_state=None,
            render_action=True,
            local_keypoint_map: Dict[str, np.ndarray]=None, 
            color_map: Optional[Dict[str, np.ndarray]]=None):
        super().__init__(
            legacy=legacy, 
            block_cog=block_cog,
            damping=damping,
            render_size=render_size,
            reset_to_state=reset_to_state,
            render_action=render_action)
        ws = self.window_size

        if local_keypoint_map is None:
            # create default keypoint definition
            kp_kwargs = self.genenerate_keypoint_manager_params()
            local_keypoint_map = kp_kwargs['local_keypoint_map']
            color_map = kp_kwargs['color_map']

        # create observation spaces
        Dblockkps = np.prod(local_keypoint_map['block'].shape)
        Dagentkps = np.prod(local_keypoint_map['agent'].shape)
        Dagentpos = 2

        Do = Dblockkps
        if agent_keypoints:
            # blockkp + agnet_pos
            Do += Dagentkps
        else:
            # blockkp + agnet_kp
            Do += Dagentpos
        # obs + obs_mask
        Dobs = Do * 2

        low = np.zeros((Dobs,), dtype=np.float64)
        high = np.full_like(low, ws)
        # mask range 0-1
        high[Do:] = 1.

        # (block_kps+agent_kps, xy+confidence)
        self.observation_space = spaces.Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=np.float64
        )

        self.keypoint_visible_rate = keypoint_visible_rate
        self.agent_keypoints = agent_keypoints
        self.draw_keypoints = draw_keypoints
        self.kp_manager = PymunkKeypointManager(
            local_keypoint_map=local_keypoint_map,
            color_map=color_map)
        self.draw_kp_map = None

    @classmethod
    def genenerate_keypoint_manager_params(cls):
        env = Push2dEnv()
        kp_manager = PymunkKeypointManager.create_from_pusht_env(env)
        kp_kwargs = kp_manager.kwargs
        return kp_kwargs

    def _get_obs(self):
        # get keypoints
        obj_map = {
            'block': self.block
        }
        if self.agent_keypoints:
            obj_map['agent'] = self.agent

        kp_map = self.kp_manager.get_keypoints_global(pose_map=obj_map, is_obj=True)
        # python dict guerentee order of keys and values
        kps = np.concatenate(list(kp_map.values()), axis=0)

        # select keypoints to drop
        n_kps = kps.shape[0]
        visible_kps = self.np_random.random(size=(n_kps,)) < self.keypoint_visible_rate
        kps_mask = np.repeat(visible_kps[:,None], 2, axis=1)

        # save keypoints for rendering
        vis_kps = kps.copy()
        vis_kps[~visible_kps] = 0
        draw_kp_map = {
            'block': vis_kps[:len(kp_map['block'])]
        }
        if self.agent_keypoints:
            draw_kp_map['agent'] = vis_kps[len(kp_map['block']):]
        self.draw_kp_map = draw_kp_map
        
        # construct obs
        obs = kps.flatten()
        obs_mask = kps_mask.flatten()
        if not self.agent_keypoints:
            # passing agent position when keypoints are not available
            agent_pos = np.array(self.agent.position)
            obs = np.concatenate([
                obs, agent_pos
            ])
            obs_mask = np.concatenate([
                obs_mask, np.ones((2,), dtype=bool)
            ])

        # obs, obs_mask
        obs = np.concatenate([
            obs, obs_mask.astype(obs.dtype)
        ], axis=0)
        return obs
    

    # def _render_obs(self, kp_obs):
    #     # obs, obs_mask

    #     self.reset() # obs is (40,)
    #     obs_val = kp_obs[:-2]
    #     # reshape obs_val to 9,2
    #     obs_val = obs_val.reshape(-1, 2).cpu().numpy()
    #     obs_pos = kp_obs[-2:]
    #     draw_kp_map = {'block': obs_val}
    #     img = np.zeros((self.window_size, self.window_size, 3), dtype=np.uint8)
    #     scale = np.array(img.shape[:2]) / np.array([512,512])
    #     obj_map = {
    #         'block': self.block
    #     }
    #     # self.kp_manager.convert_keypoints_to_obj_pos(obj_map, self.block)
    #     obj = self.block
    #     af_transfrom_from_obj_to_keypts = self.kp_manager.get_tf_img_obj(obj)
    #     kp_local = self.kp_manager.local_keypoint_map['block']
    #     block_keypts = af_transfrom_from_obj_to_keypts(kp_local)
    #     # invert the transformation
    #     af_transfrom_from_keypts_to_obj = af_transfrom_from_obj_to_keypts.inverse
    #     # convert the keypoints to object position
    #     obj_keypts = af_transfrom_from_keypts_to_obj(obs_val)
    #     # get matrix from affine transform
    #     # matrix = af_transfrom_from_obj_to_keypts.params
    #     # # convert rotation matrix in upper left 2x2 to angle
    #     # rot_mat = matrix[:2,:2]
    #     # angle = np.arctan2(rot_mat[1,0], rot_mat[0,0])
    #     # # get the position of the object
    #     # pos = matrix[:2,2]
    #     # angle = angle * 180 / np.pi

    #     # obj_keypts = af_transfrom_from_keypts_to_obj(block_keypts)
    #     # pose_map = {'block': obj_keypts, 'agent': obs_pos}
    #     # self.kp_manager.draw_keypoints_pose(img, pose_map, is_obj=True)
    #     kp_global = self.kp_manager.get_keypoints_global(draw_kp_map, is_obj=False)


    #     pdb.set_trace()
    #     Do = kp_obs.shape[0] // 2
    #     obs = kp_obs[:Do]
    #     obs_mask = kp_obs[Do:]
    #     obs = obs.reshape(-1, 2)
    #     obs_mask = obs_mask.reshape(-1, 2)
    #     img = np.zeros((self.window_size, self.window_size, 3), dtype=np.uint8)
    #     img = self.kp_manager.draw_keypoints(img, draw_kp_map, radius=int(img.shape[0]/96))
    #     return img
        
    
    
    def _render_frame(self, mode):
        img = super()._render_frame(mode)
        if self.draw_keypoints:
            self.kp_manager.draw_keypoints(
                img, self.draw_kp_map, radius=int(img.shape[0]/96))
        return img
