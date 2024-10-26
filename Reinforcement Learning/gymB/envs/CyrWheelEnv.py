import numpy as np
import os
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
import mujoco
from gymnasium.spaces import Box

class CyrWheel(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 25,
    }

    def __init__(self, **kwargs):
        # Initating arrays for us logging them over time
        self.actions_log = []
        self.rewards_log = []
        self.time_steps = []
        self.time_step = 0

        utils.EzPickle.__init__(self, **kwargs)
        # Establishing Path to XML file
        xml_path = os.path.join(os.path.dirname(__file__), "cyr_wheel.xml") 
        # Setting observation space
        
        observation_space = Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float64)

        # Setting action space 
        # action_space = Box(low=np.array([-1.0]), high=np.array([1.0]), shape=(1,), dtype=np.float32)

        frame_skip = 20 # this allows us to control how many frames are "skipped" -> fine control 
        super().__init__(
            xml_path, 
            frame_skip=frame_skip, 
            observation_space=observation_space, 
            default_camera_config={
                "trackbodyid": 0, 
                "distance": 20.0,
            },
            **kwargs)
        self.action_space = Box(low=np.array([-1.0]), high=np.array([3.0]), shape=(1,), dtype=np.float32)
        self.max_steps = 1000  # Set maximum number of steps per episode
        self.step_count = 0  # Initialize step count
        self.old_reward = 0
        self.init_qpos = np.zeros(self.model.nq)  
        self.init_qvel = np.zeros(self.model.nv)
        
        self.reset_model()  # Reset the model to the initial state
        # utils.EzPickle.__init__(self, **kwargs)
        # xml_path = os.path.join(os.path.dirname(__file__), "cyr_wheel.xml")
        # # start with just xyz observation space. Can you define the bounds sseperate
        # # observation_space = Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float64)
        # # Starting with the simple space of (x,y,z)- coordinates of the observation space.
        # observation_space = Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float64)
        # # self.action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        # duration = 20 # sec
        # frame_skip = duration
        # MujocoEnv.__init__(
        #     self,
        #     xml_path,
        #     frame_skip=frame_skip,
        #     observation_space=observation_space,
        #     default_camera_config={
        #         "trackbodyid": 0,
        #         "distance": 10.0,
        #     },
        #     **kwargs,
        # ) # THIS INITIALIZES THE SUPER CLASS MUJOCO ENV

        # MujocoEnv._initialize_simulation(self)

        # self.init_qpos = np.zeros(self.model.nq)  # position is 0 0 0
        # self.init_qvel = np.zeros(self.model.nv)  # Example to initialize with zeros

        # print(f"THIS IS THE ACTION SPACE: {self.action_space}")
        # self.reset_model()

        # self.desired_path = self.f()


    def f(self):
        # straight line
        self.init_pos_x = self.init_qpos[0]
        return self.init_pos_x

    # def _get_obs(self):
    #     try:
    #         median_contact = np.median(self.data.contact.pos, axis=0)
    #         if np.isnan(median_contact).any():
    #             median_contact = np.zeros(3)  # Default to zero if NaN
    #     except ValueError:
    #         median_contact = np.zeros(3)  # Default to zero if no contacts

    #     obs = np.concatenate([median_contact, self.data.qpos[:5]])  # Simplified example
    #     if np.isnan(obs).any():
    #         raise ValueError(f"NaN in observations: {obs}")

    #     return obs
    
    def step(self, action):
        self.time_step += 1

        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        reward = self.reward(obs)
        # self.step_count += 1  # Increment step count

        # Check if the episode is done
        done = self._check_done(obs)
        truncated = self.step_count >= self.max_steps  # End the episode if max steps reached

        info = {'Control': self.data.ctrl, "Action": action}

        # Log the action and reward
        self.actions_log.append(action)
        self.rewards_log.append(reward)
        self.time_steps.append(self.time_step)

        return obs, reward, done, truncated, info
    
    # def reward(self, actual):
    #     # The desired path is a specific x-coordinate
    #     desired_x = self.init_qpos[0]  # Assuming the initial x position is the desired path

    #     # Calculate the distance from the desired path
    #     x_position = actual[0]  # Assuming the x position is the first element of the observation
    #     y_position = actual[1]  # Assuming the y position is the second element of the observation

    #     # Define the desired path; here we assume an arc path for illustration
    #     radius = 4.0
    #     distance_to_path = np.abs(radius - np.sqrt(x_position**2 + y_position**2))

    #     # Penalize large deviations from the path
    #     path_reward = -distance_to_path

    #     # Encourage smooth movements by penalizing high velocities
    #     smoothness_penalty = np.linalg.norm(self.data.qvel)  # Penalize high velocities

    #     # Reward forward progress; assuming forward progress is measured by change in x position
    #     forward_progress = x_position - self.init_qpos[0]

    #     # Combine rewards and penalties
    #     reward = path_reward - 0.1 * smoothness_penalty + forward_progress

    #     return reward

    
    def reward(self, actual, path_type='waltz'):
        # Get the agent's position and velocity
        x_position = actual[0]
        y_position = actual[1]
        angular_velocity = self.data.qvel[2]  # Assuming angular velocity around the z-axis

        if path_type == 'waltz':
            # Expected periodic path for the x and y positions based on the "waltz" pattern
            period = 10  # How many timesteps for a full oscillation (tune this)
            amplitude = 6.0  # Amplitude of the waltz movement
            desired_x_position = amplitude * np.sin(2 * np.pi * self.time_step / period)  # Sinusoidal waltz in x
            desired_y_position = amplitude * np.cos(2 * np.pi * self.time_step / period)  # Cosine in y for circular motion

            # Reward for following the periodic path (penalize deviation)
            distance_to_path = np.linalg.norm([x_position - desired_x_position, y_position - desired_y_position])

            path_reward = -distance_to_path  # Negative reward for distance from the waltz path

            # Encourage a steady angular velocity (e.g., spinning smoothly)
            desired_angular_velocity = 1.0  # Tune this to control the speed of the spin
            angular_velocity_penalty = np.abs(angular_velocity - desired_angular_velocity)

            # Penalty for high deviation from the desired angular velocity
            smoothness_penalty = angular_velocity_penalty

            # Combine the path reward and smoothness penalty
            reward = path_reward - 0.1 * smoothness_penalty

        else:
            raise ValueError(f"Unsupported path_type: {path_type}")

        return reward

        # path_reward = -distance_to_path
        # smoothness_penalty = np.linalg.norm(self.data.qvel)

        # reward = path_reward - 0.1 * smoothness_penalty

        # # Apply tanh to compress the range to [-1, 1]
        # transformed_reward = np.tanh(reward)

        # return transformed_reward


    # def reward(self, actual):
    #     desired_x = self.init_qpos[0]  # Assuming the initial x position is the desired path
    #     x_position = actual[0]  # Assuming the x position is the first element of the observation
    #     y_position = actual[1]  # Assuming the y position is the second element of the observation

    #     # ARC LINE  
    #     distance_to_path = np.abs(4.0 - np.sqrt((x_position**2 + y_position**2)))
        
    #     # Penalize large deviations from the path
    #     path_reward = -distance_to_path
        
    #     # Encourage smooth movements by penalizing high velocities
    #     smoothness_penalty = np.linalg.norm(self.data.qvel)  # Penalize high velocities
        
    #     # Calculate the total reward
    #     reward = path_reward - 0.5 * smoothness_penalty
        
    #     return reward


    def _get_obs(self):
        if self.data.ncon > 0:
            # Access the contact positions correctly
            contact_positions = np.array([self.data.contact[i].pos for i in range(self.data.ncon)])
            median_contact = np.median(contact_positions, axis=0)[:3]
            if np.isnan(median_contact).any():
                median_contact = np.zeros(3)  # Default to zero if NaN
        else:
            median_contact = np.zeros(3)  # Default to zero if no contacts
        
        obs = np.concatenate([median_contact, self.data.qpos[:5]])  # Simplified example
        if np.isnan(obs).any():
            raise ValueError(f"NaN in observations: {obs}")

        return obs

    def reset_model(self):
        # Reset the simulation to a known state
        # qpos gives: x,y,z
        qpos = np.array([0, 0, 1.1, 0, 0, 0, 0, 0])
        qvel = np.array([0, 3.5, 0, 0, 0, 0, 0.04145])
        self.set_state(qpos, qvel)
        self.step_count = 0  # Reset step count on each new episode
        obs = self._get_obs()
        if np.isnan(obs).any():
            raise ValueError(f"NaN in initial observation: {obs}")
        return obs
        # return self._get_obs()

    def _check_done(self, obs):
        # Custom termination condition

        return np.linalg.norm(obs[:2]) > 10 # Example: Done if position norm is too large

    # def _calculate_path_vector(self):
    #     # Calculate vector to the path from current position
    #     return np.array([0, 0])  # Placeholder, needs actual implementation based on path logic

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

