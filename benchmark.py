import time, argparse, requests, os
from stable_baselines3 import DQN
from tqdm import tqdm

from custom_env import SnakeEnv
import search

DQN_URL = ""

algos = {'greedy': search.greedy_search, 'random': search.random_search, 'dqn': None}

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--algo', type=str, default='greedy', choices=list(algos.keys()), help='Algorithm to use')
parser.add_argument('--grid_size', type=int, default=12, help='Grid size')
parser.add_argument('--initial_size', type=int, default=4, help='Initial snake size')
parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes to run')
parser.add_argument('--show_render', type=bool, default=False, help='Render the game or not')
parser.add_argument('--delay', type=float, default=0, help='Add a delay per frame rendered')

args = parser.parse_args()
print(args)

ep = int(args.episodes)

if args.algo == 'dqn':
    if not os.path.exists('dqn_snake.zip'):
        resp = requests.get(DQN_URL)
        with open('dqn_snake.zip', "wb") as f:
            f.write(resp.content)
        
    env = SnakeEnv(mode='human', grid_size=int(args.grid_size), initial_size=int(args.initial_size))
    model = DQN.load("dqn_snake", env=env)
    rl = True
else:
    algo = algos[args.algo]
    env = SnakeEnv(mode='human', grid_size=int(args.grid_size), initial_size=int(args.initial_size))
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
            obs, r, done, info = env.step(action)
        else:
            action = algo(raw_obs)
            raw_obs, r, done = env.step_for_algo(action)

        time.sleep(args.delay)

        if args.show_render:
            env.render()

        steps += 1

        if done:
            break
        
    ep_r = env.get_score()
    total_s += steps
    total_r_per_s += ep_r/steps
    total_r += ep_r
    pbar.set_description(f"Episode {x+1}/{ep} | Episode score, steps, score per step: {ep_r:.2f}, {steps:.2f}, {ep_r/steps:.2f} | Avg score, steps, score per step: : {total_r/(x+1):.2f}, {total_s/(x+1):.2f}, {total_r_per_s/(x+1):.2f}")