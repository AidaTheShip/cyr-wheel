from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym 
import gymB
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
import imageio

env_id = 'CyrWheel-v0'
vec_env = make_vec_env(env_id, n_envs=1)

# Initialize the PPO model
model = PPO('MlpPolicy', vec_env, n_epochs=20, verbose=1, n_steps=10000, batch_size=10000)

# Train the model
model.learn(total_timesteps=int(1e5), progress_bar=True)

# Save the model
model.save("PPO_cyrwheel")

eval_env = make_vec_env(env_id, n_envs=1)


# Load the model
# model = PPO.load("PPO_cyrwheel", env=eval_env)

# Try DQN?
model = PPO.load("PPO_cyrwheel", env=eval_env)

# Evaluate the policy
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100)

# Reset the environment
obs = eval_env.reset()

timesteps = []
control_data = []
contact_points_x = []
contact_points_y = []
contact_points_z = []
action_data = []
frames = []
# writer = imageio.get_writer('cyr_wheel_simulation.mp4', fps=25)

n_eval_episodes = 1
# Run the trained model
for _ in range(n_eval_episodes):
    obs = eval_env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = eval_env.step(action)
        # print(obs[0][1]) # Testing whether this works.
        control_data.append(infos[0]['Control'])  # Append control info
        action_data.append(infos[0]['Action'])
        timesteps.append(i)
        contact_points_x.append(obs[0][0])
        contact_points_y.append(obs[0][1])
        contact_points_z.append(obs[0][2])
        # if dones:
        #     obs = eval_env.reset()
        # contact_points = infos[0].get('contact_points', None)
        # if contact_points:
        #     overall_contact_points.append(contact_points)
        #     print(f"Contact Points at step {i}: {contact_points}")
        # else:
        #     overall_contact_points.append([0, 0, 0]) 
        
        # Renders right after simulation
        # eval_env.render("human")
        frame = eval_env.render("rgb_array")
        frames.append(frame)
        # writer.append_data(frame)

with imageio.get_writer("cyr_wheel_simulation.mp4", fps=25) as writer:
        for frame in frames:
            writer.append_data(frame)

print(f"MEAN REWARD: {mean_reward}\nSTD_reward: {std_reward}")
# Ensure proper closing of the environments
eval_env.close()
vec_env.close()
writer.close()

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

# print(f"CONTACT POINTS x: {contact_points_x}, CONTACT POINTS y: {contact_points_y}, CONTACT POINTS z: {contact_points_z}")
print(f"CONTROL: {control_data}, \n ACTION: {action_data}")

plt.figure(figsize=(12, 6))
plt.plot(contact_points_x, contact_points_y, label='Contact Point X vs. Contact Point Y')
# plt.plot(timesteps, contact_points_y, label='Contact Point Y')
# plt.plot(timesteps, contact_points_z, label='Contact Point Z')
plt.xlabel('Contact Points X')
plt.ylabel('Contact Points Y')
plt.title('Contact Points X vs. Y')
plt.legend()
# plt.grid(True)
plt.show()


plt.figure(figsize=(12, 6))
timesteps = list(range(1000))
plt.plot(timesteps, control_data, label='Control Data')
plt.xlabel('Time Steps')
plt.ylabel('Control Values')
plt.title('Control Values Over Time')
plt.legend()
# plt.grid(True)
plt.show()

# plt.figure(figsize=(12, 6))
# timesteps = list(range(1000))
# plt.plot(timesteps, action_data, label='Action Data')
# plt.xlabel('Time Steps')
# plt.ylabel('Action Values')
# plt.title('Action Values Over Time')
# plt.legend()
# # plt.grid(True)
# plt.show()