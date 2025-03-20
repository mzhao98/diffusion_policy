import gym
from gym import spaces

import collections
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d
import shapely.geometry as sg
import cv2
import skimage.transform as st
from diffusion_policy.env.pusht.pymunk_override import DrawOptions
import pdb

def pymunk_to_shapely(body, shapes):
    geoms = list()
    for shape in shapes:
        if isinstance(shape, pymunk.shapes.Poly):
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            verts += [verts[0]]
            geoms.append(sg.Polygon(verts))
        else:
            raise RuntimeError(f'Unsupported shape type {type(shape)}')
    geom = sg.MultiPolygon(geoms)
    return geom

class Push2dEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}
    reward_range = (0., 1.)

    def __init__(self,
            legacy=False, 
            block_cog=None, damping=None,
            render_action=True,
            render_size=96,
            reset_to_state=None
        ):
        self._seed = None
        self.seed()
        self.window_size = ws = 512  # The size of the PyGame window
        self.render_size = render_size
        self.sim_hz = 100
        # Local controller params.
        # self.k_p, self.k_v = 100, 20    # PD control.z
        self.k_p, self.k_v = 100, 20  # PD control.z
        self.control_hz = self.metadata['video.frames_per_second']
        # legcay set_state for data compatibility
        self.legacy = legacy

        # agent_pos, block_pos, block_angle
        self.observation_space = spaces.Box(
            low=np.array([0,0,0,0,0,0,0,0], dtype=np.float64),
            high=np.array([ws,ws,ws,ws,ws, ws, ws, ws], dtype=np.float64),
            shape=(8,),
            dtype=np.float64
        )

        # positional goal for agent
        self.action_space = spaces.Box(
            low=np.array([0,0], dtype=np.float64),
            high=np.array([ws,ws], dtype=np.float64),
            shape=(2,),
            dtype=np.float64
        )


        self.render_action = render_action

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self.screen = None

        self.space = None
        self.teleop = None
        self.render_buffer = None
        self.latest_action = None
        self.reset_to_state = reset_to_state
        self.agent_copies = []
        self.agent_copies_lines = []
        self.colors = []
    
    def reset(self):
        seed = self._seed
        self._setup()
        
        # use legacy RandomState for compatibility
        state = self.reset_to_state
        if state is None:
            rs = np.random.RandomState(seed=seed)
            state = np.array([
                rs.randint(250, 252), rs.randint(50, 52), # agent pos
                # rs.randint(100, 400), rs.randint(100, 400), # block pos
                # rs.randint(40, 45), rs.randint(40, 45),
                # rs.randint(250, 262), rs.randint(250, 262), # block pos
                rs.randint(40,452), rs.randint(40, 452), # block pos
                # np.clip(rs.normal(250, 50), 50, 400),  # block pos normal around 250
                # np.clip(rs.normal(250, 50), 50, 400),  # block pos normal around 250
                rs.randn() * 2 * np.pi - np.pi,  # block angle
                # 0,  # block angle
                rs.randint(50, 52), rs.randint(260, 262),  # goal pos 1
                rs.randint(400, 402), rs.randint(260, 262)  # goal pos 2
                ])
        self._set_state(state)

        observation = self._get_obs()
        return observation
    
    def reset_with_input(self, agent_pos, block_pos, block_angle, goal_pos1, goal_pos2):
        seed = self._seed
        self._setup()
        
        # use legacy RandomState for compatibility
        state = self.reset_to_state
        if state is None:
            rs = np.random.RandomState(seed=seed)
            state = np.array([
                agent_pos[0], agent_pos[1], # agent pos
                block_pos[0], block_pos[1], # block pos
                block_angle,  # block angle
                # 0,  # block angle
                goal_pos1[0], goal_pos1[1],  # goal pos 1
                goal_pos2[0], goal_pos2[1]  # goal pos 2
                ])
        self._set_state(state)

        observation = self._get_obs()
        return observation

    def step(self, action):
        dt = 1.0 / self.sim_hz
        self.n_contact_points = 0
        n_steps = self.sim_hz // self.control_hz
        # print("nsteps", n_steps)
        if action is not None:
            self.latest_action = action
            # print("action", action)
            for i in range(n_steps):
                # Step PD control.
                # self.agent.velocity = self.k_p * (act - self.agent.position)    # P control works too.
                acceleration = self.k_p * (action - self.agent.position) + self.k_v * (Vec2d(0, 0) - self.agent.velocity)
                self.agent.velocity += acceleration * dt

                # Step physics.
                self.space.step(dt)
                # print("agent position", self.agent.position)
                # self.render('human')

        # compute reward
        goal_body = self._get_goal_pose_body(self.goal_pose1)
        goal_geom = pymunk_to_shapely(goal_body, self.block.shapes)
        block_geom = pymunk_to_shapely(self.block, self.block.shapes)

        intersection_area = goal_geom.intersection(block_geom).area
        goal_area = goal_geom.area
        coverage1 = intersection_area / goal_area

        # check distance to second goal
        goal_body = self._get_goal_pose_body(self.goal_pose2)
        goal_geom = pymunk_to_shapely(goal_body, self.block.shapes)
        block_geom = pymunk_to_shapely(self.block, self.block.shapes)
        intersection_area = goal_geom.intersection(block_geom).area
        goal_area = goal_geom.area
        coverage2 = intersection_area / goal_area

        coverage = max(coverage1, coverage2)
        # print("coverage", coverage)

        reward = np.clip(coverage / self.success_threshold, 0, 1)
        # reward is distance of agent to goal
        # reward = np.linalg.norm(self.agent.position - self.goal_pose) / 512
        done = coverage > self.success_threshold
        # done = reward < 0.05

        observation = self._get_obs()
        info = self._get_info()
        # print("output step", observation, reward, done, info)
        # print("obs shape", observation.shape)

        return observation, reward, done, info
    


    def step_with_samples(self, action, list_of_sampled_actions):
        # action = action[0]
        # print("input step with samples", action, list_of_sampled_actions)
        dt = 1.0 / self.sim_hz
        self.n_contact_points = 0
        n_steps = self.sim_hz // self.control_hz
        initial_agent_velocity = self.agent.velocity
        initial_agent_position_x = self.agent.position.x
        initial_agent_position_y = self.agent.position.y
        if action is not None:
            self.latest_action = action
            for i in range(n_steps):
                # print("action", action)
                # Step PD control.
                # self.agent.velocity = self.k_p * (act - self.agent.position)    # P control works too.
                acceleration = self.k_p * (action - self.agent.position) + self.k_v * (Vec2d(0, 0) - self.agent.velocity)
                self.agent.velocity += acceleration * dt

                # Step physics.
                self.space.step(dt)
                # print("stepped physics")
        current_velocity = self.agent.velocity
        self.agent.velocity -= current_velocity
        current_block_position = self.block.position
        current_block_angle = self.block.angle
        # print("made it here")

        # randomly sample colors for len(list_of_sampled_actions)
        self.colors = []
        for i in range(len(list_of_sampled_actions)):
            self.colors.append((np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))

        # pdb.set_trace()
        # display the sampled action trajectories
        for sample_action_idx in range(len(list_of_sampled_actions)):
            act_seq = list_of_sampled_actions[sample_action_idx][0]
            agent_copy = self.add_circle((initial_agent_position_x, initial_agent_position_y), 5, self.colors[sample_action_idx])
            agent_copy.velocity = initial_agent_velocity
            self.agent_copies.append(agent_copy)
            previous_agent_position = agent_copy.position
            
            for future_t in range(0, len(act_seq)):
                act = act_seq[future_t]
                # print("act", act)
                
                for i in range(n_steps):
                    # act = action_sample[i]
                    # print("act", act)
                    # Step PD control.
                    
                    # self.agent.velocity = self.k_p * (act - self.agent.position)    # P control works too.
                    acceleration = self.k_p * (act - agent_copy.position) + self.k_v * (Vec2d(0, 0) - agent_copy.velocity)
                    # make a copy of the agent
                    
                    # agent_copy.velocity = self.agent.velocity
                    agent_copy.velocity += acceleration * dt

                    # Step physics.
                    self.space.step(dt)
                    # draw a colored line between the current agent position and the previous_agent_position
                line = (previous_agent_position, agent_copy.position, self.colors[sample_action_idx])
                self.agent_copies_lines.append(line)
                previous_agent_position = agent_copy.position

                agent_copy.velocity -= agent_copy.velocity

        self.agent.velocity += current_velocity
        self.block.position = current_block_position
        self.block.angle = current_block_angle
        # print("made it here 2")

        # compute reward
        goal_body = self._get_goal_pose_body(self.goal_pose1)
        goal_geom = pymunk_to_shapely(goal_body, self.block.shapes)
        block_geom = pymunk_to_shapely(self.block, self.block.shapes)

        intersection_area = goal_geom.intersection(block_geom).area
        goal_area = goal_geom.area
        coverage1 = intersection_area / goal_area

        # check distance to second goal
        goal_body = self._get_goal_pose_body(self.goal_pose2)
        goal_geom = pymunk_to_shapely(goal_body, self.block.shapes)
        block_geom = pymunk_to_shapely(self.block, self.block.shapes)
        intersection_area = goal_geom.intersection(block_geom).area
        goal_area = goal_geom.area
        coverage2 = intersection_area / goal_area

        coverage = max(coverage1, coverage2)

        reward = np.clip(coverage / self.success_threshold, 0, 1)
        # reward is distance of agent to goal
        # reward = np.linalg.norm(self.agent.position - self.goal_pose) / 512
        done = coverage > self.success_threshold
        # done = reward < 0.05

        observation = self._get_obs()
        info = self._get_info()
        print("output step with samples", observation, reward, done, info)
        print("obs shape", observation.shape)
        print("info", info)
        return observation, reward, done, info

    def render(self, mode):
        return self._render_frame(mode)

    def teleop_agent(self):
        TeleopAgent = collections.namedtuple('TeleopAgent', ['act'])
        def act(obs):
            act = None
            mouse_position = pymunk.pygame_util.from_pygame(Vec2d(*pygame.mouse.get_pos()), self.screen)
            if self.teleop or (mouse_position - self.agent.position).length < 30:
                self.teleop = True
                act = mouse_position
            return act
        return TeleopAgent(act)

    def _get_obs(self):
        obs = np.array(
            tuple(self.agent.position) \
            + tuple(self.block.position) \
            + tuple([self.block.angle]) \
            + tuple(self.goal_pose1) \
            + tuple(self.goal_pose2)
            # + tuple([1])
        )
        return obs

    def _get_goal_pose_body(self, pose):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (50, 100))
        body = pymunk.Body(mass, inertia)

        # body = pymunk.Body(body_type=pymunk.Body.STATIC)
        # preserving the legacy assignment order for compatibility
        # the order here doesn't matter somehow, maybe because CoM is aligned with body origin
        body.position = pose[:2]
        body.angle = pose[2]
        return body
    
    def _get_info(self):
        n_steps = self.sim_hz // self.control_hz
        n_contact_points_per_step = int(np.ceil(self.n_contact_points / n_steps))
        info = {
            'pos_agent': np.array(self.agent.position),
            'vel_agent': np.array(self.agent.velocity),
            'block_pose': np.array(list(self.block.position)),
            'block_angle': np.array([self.block.angle]),
            'goal_pose1': np.array(self.goal_pose1),
            'goal_pose2': np.array(self.goal_pose2),
            'n_contacts': n_contact_points_per_step}
        return info

    def _render_frame(self, mode):
        # print("starting render")
        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        self.screen = canvas

        draw_options = DrawOptions(canvas)

        # Draw goal pose.
        # Draw goal pose.
        goal_body = self._get_goal_pose_body(self.goal_pose1)
        for shape in self.block.shapes:
            goal_points = [pymunk.pygame_util.to_pygame(goal_body.local_to_world(v), draw_options.surface) for v in
                           shape.get_vertices()]
            goal_points += [goal_points[0]]
            pygame.draw.polygon(canvas, self.goal_color, goal_points)

        goal_body = self._get_goal_pose_body(self.goal_pose2)
        for shape in self.block.shapes:
            goal_points = [pymunk.pygame_util.to_pygame(goal_body.local_to_world(v), draw_options.surface) for v in
                           shape.get_vertices()]
            goal_points += [goal_points[0]]
            pygame.draw.polygon(canvas, self.goal_color, goal_points)

        

        # Draw agent and block.
        self.space.debug_draw(draw_options)

        # draw lines
        for line in self.agent_copies_lines:
            pygame.draw.line(canvas, line[2], line[0], line[1], 2)
        # self.agent_copies_lines = []


        if mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            
                

            # the clock is already ticked during in step for "human"


        img = np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        img = cv2.resize(img, (self.render_size, self.render_size))
        if self.render_action:
            if self.render_action and (self.latest_action is not None):
                action = np.array(self.latest_action)
                coord = (action / 512 * 96).astype(np.int32)
                marker_size = int(8/96*self.render_size)
                thickness = int(1/96*self.render_size)
                cv2.drawMarker(img, coord,
                    color=(255,0,0), markerType=cv2.MARKER_CROSS,
                    markerSize=marker_size, thickness=thickness)
        
        # if self.agent_copy is not empty, draw them and remove them
        for agent_copy in self.agent_copies:
            agent_pos = agent_copy.position
            agent_pos = np.array([agent_pos[0], agent_pos[1]])
            coord = (agent_pos / 512 * 96).astype(np.int32)
            marker_size = int(8/96*self.render_size)
            thickness = int(1/96*self.render_size)
            # pygame.draw.circle(canvas, (0, 0, 0), agent_pos, 15, 'LightGray')
            cv2.drawMarker(img, coord,
                    color=(50,50,50), markerType=cv2.MARKER_STAR,
                    markerSize=marker_size, thickness=thickness)
        #     # self.space.debug_draw(draw_options)
            self.agent_copies.remove(agent_copy)
            self.space.remove(agent_copy, *agent_copy.shapes)
            # remove lines
            for line in self.agent_copies_lines:
                # delete from pygame
                pygame.draw.line(canvas, (0, 0, 0), line[0], line[1], 5)
            # self.agent_copies_lines = []

        # print("rendered frame")
        # convert current frame to RGB
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
    
    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0,25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)

    def _handle_collision(self, arbiter, space, data):
        self.n_contact_points += len(arbiter.contact_point_set.points)

    def _set_state(self, state):
        if isinstance(state, np.ndarray):
            state = state.tolist()
        pos_agent = state[:2]
        pos_block = state[2:4]
        pos_goal1 = state[5:7]
        pos_goal2 = state[7:9]
        self.agent.position = pos_agent
        # setting angle rotates with respect to center of mass
        # therefore will modify the geometric position
        # if not the same as CoM
        # therefore should be modified first.
        self.block.position = pos_block
        self.block.angle = state[4]
        pos_goal1.append(0)
        pos_goal2.append(0)
        self.goal_pose1 = pos_goal1
        self.goal_pose2 = pos_goal2

        # Run physics to take effect
        self.space.step(1.0 / self.sim_hz)
    
    def _set_state_local(self, state_local):
        agent_pos_local = state_local[:2]
        block_pose_local = state_local[2:]
        tf_img_obj = st.AffineTransform(
            translation=self.goal_pose[:2], 
            rotation=self.goal_pose[2])
        tf_obj_new = st.AffineTransform(
            translation=block_pose_local[:2],
            rotation=block_pose_local[2]
        )
        tf_img_new = st.AffineTransform(
            matrix=tf_img_obj.params @ tf_obj_new.params
        )
        agent_pos_new = tf_img_new(agent_pos_local)
        new_state = np.array(
            list(agent_pos_new[0]) + list(tf_img_new.translation) \
                + [tf_img_new.rotation])
        self._set_state(new_state)
        return new_state

    def _setup(self):
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = 0
        self.teleop = False
        self.render_buffer = list()
        
        # Add walls.
        walls = [
            self._add_segment((5, 506), (5, 5), 2),
            self._add_segment((5, 5), (506, 5), 2),
            self._add_segment((506, 5), (506, 506), 2),
            self._add_segment((5, 506), (506, 506), 2)
        ]
        self.space.add(*walls)

        # Add agent, block, and goal zone.
        self.agent = self.add_circle((256, 400), 15)
        self.block = self.add_box((256, 300), 55, 55, 0)
        self.goal_color = pygame.Color('LightGreen')
        self.goal_pose1 = np.array([50,256])  # x, y
        self.goal_pose2 = np.array([375,256])  # x, y

        # Add collision handling
        self.collision_handeler = self.space.add_collision_handler(0, 0)
        self.collision_handeler.post_solve = self._handle_collision
        self.n_contact_points = 0

        self.max_score = 50 * 100
        self.success_threshold = 0.95    # 95% coverage.

    def _add_segment(self, a, b, radius):
        shape = pymunk.Segment(self.space.static_body, a, b, radius)
        shape.color = pygame.Color('LightGray')    # https://htmlcolorcodes.com/color-names
        return shape

    def add_circle(self, position, radius, color='RoyalBlue'):
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        body.friction = 1
        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color(color)
        self.space.add(body, shape)
        return body

    def add_box(self, position, height, width, angle):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (height, width))
        body = pymunk.Body(mass, inertia)
        body.position = position
        body.angle = angle
        shape = pymunk.Poly.create_box(body, (height, width))
        shape.color = pygame.Color('LightSlateGray')
        self.space.add(body, shape)
        return body

    def add_tee(self, position, angle, scale=30, color='LightSlateGray', mask=pymunk.ShapeFilter.ALL_MASKS()):
        mass = 1
        length = 4
        vertices1 = [(-length*scale/2, scale),
                                 ( length*scale/2, scale),
                                 ( length*scale/2, 0),
                                 (-length*scale/2, 0)]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        vertices2 = [(-scale/2, scale),
                                 (-scale/2, length*scale),
                                 ( scale/2, length*scale),
                                 ( scale/2, scale)]
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices1)
        body = pymunk.Body(mass, inertia1 + inertia2)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = (shape1.center_of_gravity + shape2.center_of_gravity) / 2
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, shape1, shape2)
        return body
