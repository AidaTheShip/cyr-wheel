import gym
from mujoco_py import load_model_from_path, MjSim, MjViewer

# MIGHT MERGE THIS WITH SIMULATION.py - more overview like this though

xml_path = "cyr_wheel.xml"

class CustomMuJoCoEnv(gym.Env):
    def __init__(self, xml_file_path):
        model = load_model_from_path(xml_file_path)
        self.sim = MjSim(model)
        self.viewer = MjViewer(self.sim)
        # Initialize observation and action spaces here

    def step(self, action):
        # Implement this method to update the environment's state
        pass

    def reset(self):
        # Rsetting the state of the environment
        pass

    def render(self, mode='human', close=False):
        # Either we put the render function here or use the glfw code that I wrote in the simulation file.
        if mode == 'human':
            self.viewer.render()
