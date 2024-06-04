import numpy as np
import os
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

class CyrWheel(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 100,
    }

    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        xml_path = os.path.join(os.path.dirname(__file__), "cyr_wheel.xml")
        observation_space = Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float64)
        action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        MujocoEnv.__init__(
            self,
            xml_path,
            frame_skip=5,
            observation_space=observation_space,
            default_camera_config={
                "trackbodyid": 0,
                "distance": 10.0,
            },
            **kwargs,
        )

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        reward = self._calculate_reward(obs)
        done = self._check_done(obs)
        return obs, reward, done, False, {}

    def _get_obs(self):
        position = self.data.qpos[:2]  # Assuming qpos contains [x, y] for the wheel's position
        velocity = self.data.qvel[:2]  # Assuming qvel contains [vx, vy] for the wheel's velocity
        orientation = self.data.qpos[2]  # Assuming the third qpos is the wheel's orientation
        angular_velocity = self.data.qvel[2]  # Assuming the third qvel is the wheel's angular velocity
        path_vector = self._calculate_path_vector()
        return np.concatenate([position, velocity, [orientation, angular_velocity], path_vector])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _calculate_reward(self, obs):
        # Custom reward function based on position, velocity, or other criteria
        return -np.sum(np.square(obs[:2]))  # Example: Negative sum of squares of positions

    def _check_done(self, obs):
        # Custom termination condition
        return np.linalg.norm(obs[:2]) > 10  # Example: Done if position norm is too large

    def _calculate_path_vector(self):
        # Calculate vector to the path from current position
        return np.array([0, 0])  # Placeholder, needs actual implementation based on path logic

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

