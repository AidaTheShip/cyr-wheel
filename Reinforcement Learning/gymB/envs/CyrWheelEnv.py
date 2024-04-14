import numpy as np
import os
from gym import utils, error, spaces
from gym.envs.mujoco import mujoco_env
from mujoco_py import MjViewer, functions

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
        pass
    
    def viewer_setup(self):
        return super().viewer_setup()
    
    def reset_model(self):
        return super().reset_model()
    
    def _get_obs(self):
        pass
    
        