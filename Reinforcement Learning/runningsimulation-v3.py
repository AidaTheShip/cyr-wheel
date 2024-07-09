from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym 
import gymB
from stable_baselines3.common.evaluation import evaluate_policy

env_id = 'CyrWheel-v0'
vec_env = make_vec_env(env_id, n_envs=1)

# Initialize the PPO model
model = PPO('MlpPolicy', vec_env, verbose=1)

# Train the model
model.learn(total_timesteps=int(2e4), progress_bar=True)

# Save the model
model.save("PPO_cyrwheel")

eval_env = make_vec_env(env_id, n_envs=2)


# Load the model
model = PPO.load("PPO_cyrwheel", env=eval_env)

# Evaluate the policy
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=5000)

# Reset the environment
obs = eval_env.reset()

# Run the trained model
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = eval_env.step(action)
    eval_env.render("human")

print(f"MEAN REWARD: {mean_reward}\nSTD_reward: {std_reward}")
# Ensure proper closing of the environments
eval_env.close()
vec_env.close()