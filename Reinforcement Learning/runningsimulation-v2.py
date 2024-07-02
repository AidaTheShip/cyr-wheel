
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

env = gym.make('CyrWheel-v0', render_mode='human')
observation = env.reset()
n_timesteps = 1000

# model = PPO('MlpPolicy', env, verbose=1)
# model.learn(total_timesteps=n_timesteps)
# model.save("ppo_cyrwheel")

# obs = env.reset()
# for _ in range(n_timesteps):
#     action = env.action_space.sample()  # Random action
#     observation, reward, done, info = env.step(action)
#     env.render()
#     if done:
#         print("Episode finished after {} timesteps".format(_+1))
#         observation = env.reset()  # Reset the environment
# env.close()


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

# Function to compute discounted rewards
# def compute_discounted_rewards(rewards, gamma): # DO WE NEED THIS? 
#     discounted_rewards = []
#     R = 0
#     for r in rewards[::-1]:
#         R = r + gamma * R
#         discounted_rewards.insert(0, R)
#     return discounted_rewards

def select_action(policy_net, state):
    # state = torch.from_numpy(state).float()
    print("STATE", state)
    if state[0] is not None:
        state = torch.tensor(state[0])
    else: 
        state = torch.tensor(numpy.zeros(shape=(8,)))

    probs = policy_net(state)
    action = np.random.choice(len(probs.detach().numpy()), p=probs.detach().numpy())
    return action

# Main Training Loop
def train(env, policy_net, optimizer, num_episodes, epsilon=0.01):
    for episode in range(num_episodes):
        state = env.reset()
        log_probs = []
        rewards = []
        done = False
        choice = np.choice(0,1)
        while not done:
            if epsilon < choice:
                action = env.action_space.sample()  # Random action
            action, log_prob = select_action(policy_net, state)
            print(action.shape, state)
            # observation, reward, done, info = env.step(action)
            next_state, reward, done, _ = env.step([action])
            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state

            env.render()
            if done:
                policy_gradient = []
                for log_prob, reward in zip(log_probs, rewards):
                    policy_gradient.append(-log_prob * reward)

                optimizer.zero_grad()
                policy_gradient = torch.stack(policy_gradient).sum()
                policy_gradient.backward()
                optimizer.step()

                print(f"Episode {episode+1}: Total Reward = {sum(rewards)}")
                
                observation = env.reset()  # Reset the environment
                env.close()
                # discounted_rewards = compute_discounted_rewards(rewards, gamma)
                # discounted_rewards = torch.tensor(discounted_rewards)
                # policy_gradient = []


if __name__ == "__main__":
    # env = gym.make('CyrWheel')  # Replace with your environment
    state_space = env.observation_space.shape[0]
    action_space = 1

    print(state_space, action_space)

    policy_net = PolicyNetwork(state_space, action_space)
    optimizer = optim.Adam(policy_net.parameters(), lr=0.01)
    num_episodes = 1000
    gamma = 0.99

    train(env, policy_net, optimizer, num_episodes, gamma)

    # Evaluate the trained policy
    state = env.reset()
    done = False
    while not done:
        env.render()
        action, _ = select_action(policy_net, state)
        state, reward, done, _ = env.step([action])
    env.close()
