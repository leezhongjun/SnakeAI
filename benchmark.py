import time, argparse, requests, os
from stable_baselines3 import DQN, A2C, PPO
from tqdm import tqdm

from custom_env import SnakeEnv
import search

DQN_URL = "https://www.dropbox.com/s/mt1y5xh6z4s6pn4/dqn_snake.zip?raw=1"
A2C_URL = "https://www.dropbox.com/s/jcwxplwtkfffsgr/a2c_snake.zip?raw=1"

algos = {'greedy': search.greedy_search, 'random': search.random_search, 'bfs': search.bfs_search, 'dfs': search.dfs_search, 'ham': search.hamiltonian_path_search, 'op_ham': search.optimised_hamiltonian_path_search, 'dqn': None, 'a2c': None}

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--algo', type=str, default='greedy', choices=list(algos.keys()), help='Algorithm to use')
parser.add_argument('--grid_size', type=int, default=12, help='Grid size')
parser.add_argument('--initial_size', type=int, default=4, help='Initial snake size')
parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes to run')
parser.add_argument('--show_render', type=bool, default=False, help='Render the game or not')
parser.add_argument('--delay', type=float, default=0, help='Add a delay between frames rendered')
parser.add_argument('--save_gif', type=str, default='', help='Enter a filename to save the gif as (e.g. snake_video). Leave blank to not save gif.')

args = parser.parse_args()
print(args)

ep = int(args.episodes)

filename = args.save_gif

if args.algo == 'dqn':
    if not os.path.exists('dqn_snake.zip'):
        resp = requests.get(DQN_URL)
        with open('dqn_snake.zip', "wb") as f:
            f.write(resp.content)
        
    env = SnakeEnv(mode='human', grid_size=int(args.grid_size), initial_size=int(args.initial_size), save_video=bool(filename))
    model = DQN.load("dqn_snake", env=env)
    rl = True

elif args.algo == 'a2c':
    if not os.path.exists('a2c_snake.zip'):
        resp = requests.get(A2C_URL)
        with open('a2c_snake.zip', "wb") as f:
            f.write(resp.content)
        
    env = SnakeEnv(mode='human', grid_size=int(args.grid_size), initial_size=int(args.initial_size), save_video=bool(filename))
    model = A2C.load("a2c_snake", env=env)
    rl = True

else:
    algo = algos[args.algo]
    env = SnakeEnv(mode='human', grid_size=int(args.grid_size), initial_size=int(args.initial_size), save_video=bool(filename))
    rl = False

pbar = tqdm(range(ep))
total_r = total_s = total_r_per_s = 0

for x in pbar:
    obs = env.reset()
    raw_obs = env.get_raw_obs()
    steps = 0
    while True:
        if rl:
            action, _states = model.predict(obs, deterministic=True)
            obs, r, done, info = env.step(int(action))
        else:
            action = algo(raw_obs)
            raw_obs, r, done = env.step_for_algo(action)

        if args.delay:
            time.sleep(args.delay)

        if args.show_render:
            env.render()

        steps += 1

        if done:
            break
        
    if filename:
        env.save_video_func(f"{filename}_{x}.gif")
    ep_r = env.get_score()
    total_s += steps
    total_r_per_s += ep_r/steps
    total_r += ep_r
    pbar.set_description(f"Episode {x+1}/{ep} | Episode score, steps, score per step: {ep_r:.2f}, {steps:.2f}, {ep_r/steps:.2f} | Avg score, steps, score per step: : {total_r/(x+1):.2f}, {total_s/(x+1):.2f}, {total_r_per_s/(x+1):.2f}")
