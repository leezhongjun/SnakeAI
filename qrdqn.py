from sb3_contrib import QRDQN
from stable_baselines3.common.callbacks import CheckpointCallback

from custom_env import SnakeEnv

env = SnakeEnv(mode='human', grid_size=12, initial_size=4)

checkpoint_callback = CheckpointCallback(
  save_freq=int(1e6),
  save_path="./qrdqn_logs/",
  name_prefix="rl_model",
  save_replay_buffer=True,
  save_vecnormalize=True,
)

model = QRDQN('CnnPolicy', env, verbose=1, \
                tensorboard_log="./tensorboard/", \
                optimize_memory_usage=False, exploration_fraction=0.025)

# # Train the agent and display a progress bar
model.learn(total_timesteps=int(1e7), progress_bar=True, callback=checkpoint_callback)
# Save the agent
model.save("qrdqn_snake")

model = QRDQN.load("qrdqn_snake", env=env)

# Enjoy trained agent
obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()