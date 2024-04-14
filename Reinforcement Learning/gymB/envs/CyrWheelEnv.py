import numpy as np
import os
from gym import utils, error, spaces
from gym.envs.mujoco import mujoco_env
from mujoco_py import MjViewer, functions

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

class CyrWheel(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        xml_path = "cyr_wheel.xml" # Setting the path for the model
        dirname = os.path.dirname(__file__)
        abspath = os.path.join(dirname + "/" + xml_path) # making sure that it aligns with our directory
        xml_path = abspath
        
        frame_skip = 5
        mujoco_env.MujocoEnv.__init__(self, xml_path, frame_skip)
        
    def step(self, a): 
        """ 
        Purpose of the function: advance env by one timestep using the action a
        - a: array/list representing action to be taken in the environment
        - returns: commonly returns the next state of the environment, the reward obtained after taking the action, boolean indicating whether episode has ended, dictionary with extra information
        
        """
        pass
    
    def viewer_setup(self):
        return super().viewer_setup()
    
    def reset_model(self):
        """
        Purpose: reset the env to an initial state to start a new episode
        returns: innitial observation of the env after reset
        """
        return super().reset_model()
    
    def _get_obs(self):
        """
        purpose: extract / calculate the current observation of the env that will be provided to the agent
        returns: returns an array or a list representing the current state of the evn from the perspective of the agent
        """
        pass
    
