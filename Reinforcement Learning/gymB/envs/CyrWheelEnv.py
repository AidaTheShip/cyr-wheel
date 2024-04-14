import numpy as np
import os
from gym import utils, error, spaces
from gym.envs.mujoco import mujoco_env
from mujoco_py import MjViewer, functions

from gym.spaces import Box # the space class is used to define observation and action spaces.
DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 2.0)),
    "elevation": -20.0,
}

# Note: We are using the MujocoEnv superclass which is used for ALL Mujcoco environments
"""
Timestep refers to a single unit of time progression in the simulation or environment.
During each timestep, several things occur: 
- action execusion => agent performs an action (e.g. moving forweard, turning adjusting, etc.)
- environment response => updating the environment based on the action taken by the agent and other internal dynamics 
- observation update => providing new observation to the agent
- reward calculation => alongside with new observation, a reward is often calculated based on the action taken

"""

"""
### Description
The Cyr Wheel is a simulation of a torus-shaped object that the agent can control within a 3D space. The goal is to navigate the Cyr Wheel along a certain path to understand the control mechanism better. 

### Observation Space
The Observation space in RL defines the set of all possible states an agent can observe from the environment. It encapsulates the information the agent receives at each timestep to make decisions.
In our case, the observations include the position, orientation, velocity, and angular velocity of the Cyr wheel within the env? 

| Num | Observation            | Min  | Max  | Name (in corresponding XML file) | Joint | Unit               |
    |-----|------------------------|------|------|----------------------------------|-------|--------------------|
    | 0   | x-coordinate of the wheel | -Inf | Inf | wheel_x                         | slide | position (m)       |
    | 1   | y-coordinate of the wheel | -Inf | Inf | wheel_y                         | slide | position (m)       |
    | 2   | velocity of the wheel  | -Inf | Inf | wheel_vx                        | slide | velocity (m/s)     |
    | 3   | velocity of the wheel  | -Inf | Inf | wheel_vy                        | slide | velocity (m/s)     |
    | 4   | orientation of the wheel   | -Inf | Inf | wheel_orientation              | hinge | angle (rad)        |
    | 5   | angular velocity of the wheel | -Inf | Inf | wheel_av                     | hinge | angular velocity (rad/s) |
    | 6   | x-component of path vector | -Inf | Inf | path_vector_x                 | NA    | position (m)       |
    | 7   | y-component of path vector | -Inf | Inf | path_vector_y                 | NA    | position (m)       |


### Rewards
The reward is computed based on the negative square of the distance to the path to minimize the deviation and a reward for maintaining a target velocity along the path.


### Starting State
Cyr Wheel always start at the same position

### Episode End
Episode ends when the cyr wheel envirobment has either completed the full circle or went off too hard from the path


"""

class CyrWheel(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        xml_path = "cyr_wheel.xml" # Setting the path for the model
        dirname = os.path.dirname(__file__)
        abspath = os.path.join(dirname + "/" + xml_path) # making sure that it aligns with our directory
        xml_path = abspath
        
        frame_skip = 5
        observation_space = 0
        mujoco_env.MujocoEnv.__init__(self, xml_path, frame_skip, observation_space=observation_space)
        
    def step(self, action): 
        """ 
        Purpose of the function: advance env by one timestep using the action a
        - a: array/list representing action to be taken in the environment
        - returns: commonly returns the next state of the environment, the reward obtained after taking the action, boolean indicating whether episode has ended, dictionary with extra information
        
        """
        self.do_simulation(action, self.frame_skip)
        observation = self._get_obs()
        distance_to_path = self.deviation()
        velocity_reward = self.calculate_vel_reward()
        reward = -distance_to_path**2 + velocity_reward
        done = self.check_done(distance_to_path)
        return observation, reward, done, {}
    
    def viewer_setup(self):
        return super().viewer_setup()
    
    def reset_model(self):
        """
        Purpose: reset the env to an initial state to start a new episode
        returns: innitial observation of the env after reset
        """
        # return super().reset_model() # this basically just resets the data right now
        pass
    
    def _get_obs(self):
        """
        purpose: extract / calculate the current observation of the env that will be provided to the agent
        returns: returns an array or a list representing the current state of the evn from the perspective of the agent
        """
        position = self.get_body_com("torus")
        velocity = self.sim.data.qvel # We are getting the velocity based on the mujoco simulation we are running / have instantiated
        orientation = self.sim.data.qpos[2] 
        angular_vel = self.sim.data.qvel[2] # similar to above, we are filtering out the angular velocity
        path_vec = self.calculate_path_vector(position) # this indicates the direction and distance from the current position fot he cyr wheel to the nearest point on a predefined path.
        
        return np.concatenate([position, velocity, [orientation, angular_vel], path_vec])
    
