import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import gymB
import matplotlib.pyplot as plt
import imageio



env = gym.make("CyrWheel-v0") # making an instance of our gym 
# Set up a checkpoint callback to save the model periodically during training
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./models/',
                                         name_prefix='ppo_cyrwheel')

# Set up evaluation callback to monitor the agent's performance during training
eval_callback = EvalCallback(env, best_model_save_path='./models/best_model/',
                             log_path='./logs/', eval_freq=5000,
                             deterministic=True, render=False)

# Instantiate the PPO agent
# You can adjust the PPO parameters depending on the complexity of the environment
model = PPO(
    "MlpPolicy",  # Multi-layer perceptron policy network
    env,  # The CyrWheel environment
    verbose=1,  # Print detailed training output
    tensorboard_log="./ppo_cyrwheel_tensorboard/",  # Tensorboard for monitoring training progress
    learning_rate=0.001,  # You can tweak the learning rate as needed
    n_steps=2048,  # Number of steps to collect before performing a learning update
    batch_size=64,  # Batch size for learning updates
    n_epochs=50,  # Number of times to iterate over a batch of experience
    gamma=0.99,  # Discount factor for rewards
    gae_lambda=0.95,  # GAE (Generalized Advantage Estimation) parameter
    clip_range=0.2,  # Clipping parameter for PPO
)

# Start training the agent
# Total time steps is the total number of interactions with the environment
# This may need tuning depending on the complexity of the task
total_timesteps = 1000000  # 1 million steps (can be adjusted as needed)

model.learn(
    total_timesteps=total_timesteps,  # Total training steps
    callback=[checkpoint_callback, eval_callback],  # Save models and evaluate during training
    tb_log_name="PPO_CyrWheel"
)

# Save the final trained model
model.save("ppo_cyrwheel_final_model")

eval_env = gym.make("CyrWheel-v0", render_mode='rgb_array')

# Initialize the list to store frames
frames = []
n_eval_episodes = 10  # Number of episodes to evaluate

# Evaluate the model and collect frames
for _ in range(n_eval_episodes):
    obs, _ = eval_env.reset()
    for i in range(1000):  # Assuming max steps per episode is 1000
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, _, infos = eval_env.step(action)

        # Capture the frame as a NumPy array (rgb_array mode)
        frame = eval_env.render()
        if frame is not None:
            frames.append(frame)

# Save frames as a video using imageio
with imageio.get_writer("cyr_wheel_simulation.mp4", fps=25) as writer:
    for frame in frames:
        writer.append_data(frame)

# obs, _ = eval_env.reset()
# for _ in range(1000):  # Run for 1000 steps to test the agent
#     action, _states = model.predict(obs, deterministic=True)  # Only pass the observation (obs)
#     obs, rewards, dones, _, info = eval_env.step(action)
#     eval_env.render()  # Optional: render the environment to visualize the agent's performance
#     if dones:
#         obs, _ = eval_env.reset()  # Reset again after episode finishes
    



# Plotting actions and rewards over time
plt.figure(figsize=(14, 6))

# Plot the actions
plt.subplot(2, 1, 1)
plt.plot(env.time_steps, env.actions_log, label='Actions (Control Inputs)', color='b')
plt.title('Actions Over Time')
plt.xlabel('Time Step')
plt.ylabel('Action Value')
plt.legend()

# Plot the rewards
plt.subplot(2, 1, 2)
plt.plot(env.time_steps, env.rewards_log, label='Rewards', color='g')
plt.title('Rewards Over Time')
plt.xlabel('Time Step')
plt.ylabel('Reward Value')
plt.legend()

plt.tight_layout()
plt.show()