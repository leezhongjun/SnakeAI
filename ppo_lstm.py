from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback
import numpy as np

from custom_env import SnakeEnv

env = SnakeEnv(mode='human', grid_size=12, initial_size=4)

checkpoint_callback = CheckpointCallback(
  save_freq=int(1e6),
  save_path="./ppo_lstm_logs/",
  name_prefix="rl_model",
  save_replay_buffer=True,
  save_vecnormalize=True,
)

model = RecurrentPPO('CnnLstmPolicy', env, verbose=1, \
                        tensorboard_log="./tensorboard/", \
                        batch_size=256, n_steps=128, n_epochs=4, \
                        learning_rate=2.5e-4, clip_range=0.1, vf_coef=0.5, \
                        ent_coef=0.01, \
                        policy_kwargs={'enable_critic_lstm': False, 'lstm_hidden_size': 128})

# Train the agent and display a progress bar
model.learn(total_timesteps=int(1e7), progress_bar=True, callback=checkpoint_callback)
# Save the agent
model.save("ppo_lstm_snake")

model = RecurrentPPO.load("ppo_lstm_snake", env=env)

# Enjoy trained agent
env = model.get_env()
obs = env.reset()
# cell and hidden state of the LSTM
lstm_states = None
num_envs = 1
# Episode start signals are used to reset the lstm states
episode_starts = np.ones((num_envs,), dtype=bool)
while True:
    action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    episode_starts = dones
    env.render()