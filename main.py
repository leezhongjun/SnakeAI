import time, random
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from tqdm import tqdm

from custom_env import SnakeEnv

env = SnakeEnv(mode='human', grid_size=9, initial_size=4)

# check_env(env, skip_render_check=True)

# ep = 10000
# pbar = tqdm(range(ep))
# total_r = 0
# for x in pbar:
#     env.reset()
#     while True:
#         action = int(input("Enter action: "))
#         _, r, done, _ = env.step(action)
#         env.render(mode='human')
#         # time.sleep(0.5)
#         if done:
#             break
#     total_r += r
#     pbar.set_description(f"Episode {x+1}/{ep} | Reward: {r} | Avg Reward: {total_r/(x+1)}")



# Instantiate the agent
# model = DQN(
#     'MlpPolicy', env, verbose=1, exploration_fraction=0.5, exploration_final_eps=0.01
#     )
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="./dqn_tensorboard/")

# # Train the agent and display a progress bar
model.learn(total_timesteps=int(3_000_000), progress_bar=True)
# # Save the agent
model.save("ppo_snake")
# model.save("dqn_snake")
# del model  # delete trained model to demonstrate loading

# Load the trained agent
# NOTE: if you have loading issue, you can pass `print_system_info=True`
# to compare the system on which the model was trained vs the current one
# model = DQN.load("dqn_lunar", env=env, print_system_info=True)
model = PPO.load("ppo_snake", env=env)
# model = DQN.load("dqn_snake", env=env)

# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(10000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render()