from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym 
import gymB
from stable_baselines3.common.evaluation import evaluate_policy


# Define your custom environment class (CyrWheel)
# ...

# Register the environment (if not using custom registration, directly use CyrWheel class)
# gym.envs.register(
#     id='CyrWheel-v0',
#     entry_point='path.to.your.module:CyrWheel',
# )

# Create the environment
env_id = 'CyrWheel-v0'
vec_env = make_vec_env(env_id, n_envs=1)

# Initialize the PPO model
model = PPO('MlpPolicy', vec_env, verbose=1)

# Train the model
model.learn(total_timesteps=int(10000), progress_bar=True)

# Save the model
model.save("PPO_cyrwheel")

# Load the model
model = PPO.load("PPO_cyrwheel", env=vec_env)

# Evaluate the policy
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Reset the environment
# obs = vec_env.reset()
print(f"MEAN REWARD: {mean_reward} \n Std_reward: {std_reward}")
# # Run the trained model
# for i in range(1000):
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = vec_env.step(action)
#     vec_env.render("human")
