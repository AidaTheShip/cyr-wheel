from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym 
import swing
# from swing.envs.swing import SwingEnv
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
import imageio

# Initialize the environment
env = gym.make("swing-v0", render_mode='human')  # Set render_mode to 'human' for visualization

# Alternatively, you can vectorize the environment if you're using stable-baselines3
# env = make_vec_env(SwingEnv, n_envs=1)

# Initialize the PPO model
model = PPO('MlpPolicy', env, verbose=1)

# Train the model for a certain number of timesteps
model.learn(total_timesteps=10000)

# Visualize the trained model
obs = env.reset()

for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, truncated, info = env.step(action)
    env.render()

    if done or truncated:
        obs = env.reset()

env.close()