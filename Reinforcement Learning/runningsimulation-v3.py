
"""
This is the general algorithm that we are looking at. 

for n times
    while Goal is not achieved
        take_action()
        take_step()
    end of while
end of for

"""

# from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import make_vec_env
import numpy
import gymnasium as gym
import gymB
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

# Define the Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, state_space, action_space):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_space, 16, dtype=np.float64)
        self.fc2 = nn.Linear(16, action_space, dtype=np.float62)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)

# print(env.observation_space.shape[0])
# print(env.action_space.shape[0])

env_id = 'CyrWheel-v0'
env = gym.make('CyrWheel-v0', render_mode='human')

state_dim = env.observation_space.shape[0] 
action_dim = env.action_space.shape[0]

vec_env = make_vec_env(env_id, n_envs=1)

# policy_net = PolicyNetwork(state_dim, action_dim)

# model = PPO('MlpPolicy', env, verbose=1)
model = PPO('MlpPolicy', vec_env, verbose=1)

model.learn(total_timesteps=int(2e5), progress_bar=True)

model.save("DQN_cyrwheel")

model = PPO.load("DQN_cyrwheel", env=env)

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# vec_env = model.get_env()
obs = env.reset()

for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    vec_env.render("human")

# model.learn(total_timesteps=10000)

#observation = env.reset()
#n_timesteps = 1000