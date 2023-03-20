from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

from custom_env import SnakeEnv

env = SnakeEnv(mode='human', grid_size=12, initial_size=4)

checkpoint_callback = CheckpointCallback(
  save_freq=int(1e6),
  save_path="./dqn_logs/",
  name_prefix="rl_model",
  save_replay_buffer=True,
  save_vecnormalize=True,
)

model = DQN('CnnPolicy', env, verbose=1, \
                tensorboard_log="./tensorboard/", \
                buffer_size=100000, learning_rate=1e-4, \
                batch_size=32, learning_starts=100000, target_update_interval=1000, \
                train_freq=4, gradient_steps=1, exploration_fraction=0.1, exploration_final_eps=0.01, \
                optimize_memory_usage=False)

# Train the agent and display a progress bar
model.learn(total_timesteps=int(1e7), progress_bar=True, callback=checkpoint_callback)
# Save the agent
model.save("dqn_snake")

model = DQN.load("dqn_snake", env=env)

# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render()