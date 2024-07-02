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

        utils.EzPickle.__init__(self, **kwargs)
        xml_path = os.path.join(os.path.dirname(__file__), "cyr_wheel.xml")
        observation_space = Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float64)
        duration = 20  # sec, assuming this means 20 timesteps per action?
        frame_skip = duration
        super().__init__(
            xml_path, 
            frame_skip=frame_skip, 
            observation_space=observation_space, 
            default_camera_config={
                "trackbodyid": 0,
                "distance": 10.0,
            },
            **kwargs)
        
        self.max_steps = 1000  # Set maximum number of steps per episode
        self.step_count = 0  # Initialize step count
        
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
    
    # This is about moving this in the action space
    # def action(self):
    #     pass

    def _get_obs(self):
        try:
            median_contact = np.median(self.data.contact.pos, axis=0)
            if np.isnan(median_contact).any():
                median_contact = np.zeros(3)  # Default to zero if NaN
        except ValueError:
            median_contact = np.zeros(3)  # Default to zero if no contacts

        obs = np.concatenate([median_contact, self.data.qpos[:5]])  # Simplified example
        if np.isnan(obs).any():
            raise ValueError(f"NaN in observations: {obs}")

        return obs


    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        reward = self.reward(obs)
        done = self._check_done(obs)
        self.step_count += 1  # Increment step count
        if self.step_count >= self.max_steps:
            done = True  # End the episode if max steps reached
        return obs, reward, done, {}
    
    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        reward = self.reward(obs)
        self.step_count += 1  # Increment step count

        # Check if the episode is done
        done = self._check_done(obs)
        truncated = self.step_count >= self.max_steps  # End the episode if max steps reached

        return obs, reward, done, truncated, {}

    
    def reward(self, actual):
        # This is the function that will caculate the reward based on how much off the Cyr wheel is from the path
        # What is the data structure that we hve here? 
        # We want x2-x1, y2-y1. endpoint also has to be considered. 
        # will have to update the model for this beforehand.
        # print(f"COORDINATES: {actual, self.desired_path}")
        # reward = -np.mean((self.desired_path-actual)**2)  # using MSE for the reward for now.
        # return reward

        desired_path = self.init_qpos[0]
        reward = -np.mean((desired_path - actual)**2)
        return reward

    # def _get_obs(self):
    #     # note that our observation data is the contact point (x, y, z) of the Cyr Wheel at the given time. 
    #     contact_position = self.data.contact.pos.flatten()[:8] # making sure that we have (n, ) vector
    #     # if not contact_position:
    #     #     contact_position = np.zeros((1,3)) 
    #     com_sphere = self.data.body("sphere").xpos.flatten() # this gets us the COM of the sphere attached to the Cyr Wheel 
    #     com_wheel = self.data.body("wheel_and_axle").xpos.flatten() # this gets us the COM of the wheel 
    #     # position = self.data.qpos[:2]  # Assuming qpos contains [x, y] for the wheel's position
    #     # velocity = self.data.qvel[:2]  # Assuming qvel contains [vx, vy] for the wheel's velocity
    #     # orientation = self.data.qpos[2]  # Assuming the third qpos is the wheel's orientation
    #     # angular_velocity = self.data.qvel[2]  # Assuming the third qvel is the wheel's angular velocity
    #     # path_vector = self._calculate_path_vector()
    #     # return np.concatenate([position, velocity, [orientation, angular_velocity], path_vector])
    #     # return np.concatenate(contact_position, com_sphere, com_wheel)
    #     print(f"CONTACT SHAPE {contact_position.shape}")

    #     return contact_position 

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


    # def _get_obs(self):
    #     contact_position = self.data.contact
    #     # print(contact_position)
    #     expected_size = 8

    #     try:
    #         median_contact = np.median(self.data.contact.pos, axis = 0) #median value
    #     except ValueError:
    #         median_contact = [np.nan,np.nan,np.nan]


    #     print(median_contact)
    #     # actual_size = contact_position.size
    #     # if actual_size != expected_size:
    #     #     # Pad with zeros if not enough contact points\
    #     #     print(contact_position.shape)
    #     #     print("THE SIZES OF THE ARRAYS DON'T WORK OUT. ")
                
    #     #     # contact_position = np.pad(contact_position, (0, expected_size - actual_size), mode='constant')
    #     # print(f"CONTACT  {contact_position}")
    #     return contact_position

    def reset_model(self):
        # Reset the simulation to a known state
        # qpos gives: x,y,z
        qpos = np.array([0, 0, 1.6, 0, 0, 0, 0, 0])
        qvel = np.array([0, 3, 0, 0, 0, 0, 0.04145])
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

