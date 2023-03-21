from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from custom_env import SnakeEnv

env = SnakeEnv(mode='human', grid_size=12, initial_size=4)

checkpoint_callback = CheckpointCallback(
  save_freq=int(1e6),
  save_path="./ppo_logs/",
  name_prefix="rl_model",
  save_replay_buffer=True,
  save_vecnormalize=True,
)

model = PPO('CnnPolicy', env, verbose=1, \
                        tensorboard_log="./tensorboard/", \
                        batch_size=256, n_steps=128, n_epochs=4, \
                        learning_rate=2.5e-4, clip_range=0.1, vf_coef=0.5, \
                        ent_coef=0.01)

# Train the agent and display a progress bar
model.learn(total_timesteps=int(1e7), progress_bar=True, callback=checkpoint_callback)
# Save the agent
model.save("ppo_snake")

model = PPO.load("ppo_snake", env=env)

# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(10000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render()