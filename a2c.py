from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike

from custom_env import SnakeEnv


env = SnakeEnv(mode='human', grid_size=12, initial_size=4)

checkpoint_callback = CheckpointCallback(
  save_freq=int(1e6),
  save_path="./a2c_logs/",
  name_prefix="rl_model",
  save_replay_buffer=True,
  save_vecnormalize=True,
)

model = A2C('CnnPolicy', env, verbose=1, \
                tensorboard_log="./tensorboard/", \
                ent_coef=0.01, \
                vf_coef=0.25, \
                policy_kwargs={'optimizer_class': RMSpropTFLike, 'optimizer_kwargs': {'eps':1e-5}})

# Train the agent and display a progress bar
model.learn(total_timesteps=int(1e7), progress_bar=True, callback=checkpoint_callback)
# Save the agent
model.save("a2c_snake")

model = A2C.load("a2c_snake", env=env)

# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render()