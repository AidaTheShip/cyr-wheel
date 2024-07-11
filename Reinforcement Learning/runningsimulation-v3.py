from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym 
import gymB
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt

env_id = 'CyrWheel-v0'
vec_env = make_vec_env(env_id, n_envs=1)

# Initialize the PPO model
model = PPO('MlpPolicy', vec_env, verbose=1)

# Train the model
model.learn(total_timesteps=int(2e6), progress_bar=True)

# Save the model
model.save("PPO_cyrwheel")

eval_env = make_vec_env(env_id, n_envs=1)


# Load the model
model = PPO.load("PPO_cyrwheel", env=eval_env)

# Evaluate the policy
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=1000)

# Reset the environment
obs = eval_env.reset()

timesteps = []
contact_points_x = []
contact_points_y = []
contact_points_z = []

# Run the trained model
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, infos = eval_env.step(action)
    # print(obs[0][1]) # Testing whether this works.
    
    timesteps.append(i)
    contact_points_x.append(obs[0][0])
    contact_points_y.append(obs[0][1])
    contact_points_z.append(obs[0][2])
    # contact_points = infos[0].get('contact_points', None)
    # if contact_points:
    #     overall_contact_points.append(contact_points)
    #     print(f"Contact Points at step {i}: {contact_points}")
    # else:
    #     overall_contact_points.append([0, 0, 0]) 
    
    eval_env.render("human")


print(f"MEAN REWARD: {mean_reward}\nSTD_reward: {std_reward}")
# Ensure proper closing of the environments
eval_env.close()
vec_env.close()


# Plot the contact points over time
# plt.figure(figsize=(12, 6))
# plt.plot(timesteps, contact_points_x, label='Contact Point X')
# plt.plot(timesteps, contact_points_y, label='Contact Point Y')
# plt.plot(timesteps, contact_points_z, label='Contact Point Z')
# plt.xlabel('Time Steps')
# plt.ylabel('Contact Points')
# plt.title('Contact Points Over Time')
# plt.legend()
# plt.grid(True)
# plt.show()

plt.figure(figsize=(12, 6))
plt.plot(contact_points_x, contact_points_y, label='Contact Point X vs. Contact Point Y')
# plt.plot(timesteps, contact_points_y, label='Contact Point Y')
# plt.plot(timesteps, contact_points_z, label='Contact Point Z')
plt.xlabel('Contact Points X')
plt.ylabel('Contact Points Y')
plt.title('Contact Points X vs. Y')
plt.legend()
plt.grid(True)
plt.show()