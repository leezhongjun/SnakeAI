# SnakeAI
Single agent [Snake](https://en.wikipedia.org/wiki/Snake_(video_game)) AI using RL and search algorithms

Uses [Gymnasium](https://gymnasium.farama.org/) and [StableBaselines3](https://stable-baselines3.readthedocs.io/en/master/)

This uses a 12x12 grid, with the snake having initial size of 4

RL algorithms implemented: DQN, QR-DQN, PPO, Recurrent PPO, A2C

## Algorithms
Each algorithm is tested for 1000 episodes
| Algorithm | Demo | Average score *(max: 140)* | Average steps | Average score per step |
| --- | --- | --- | --- | --- |
| Random | ![random_vid](/vid_saves/random_vid_0.gif) | 2.45 | 424.1 | 0.00578 |
| Greedy | ![greedy_vid](/vid_saves/greedy_vid_0.gif) | 19.90 | 194.9 | 0.110 |
| DFS | ![dfs_vid](/vid_saves/dfs_vid_0.gif) | 20.13 | 531.3 | 0.0379 |
| BFS | ![bfs_vid](/vid_saves/bfs_vid_0.gif) | 31.35 | 323.6 | 0.0969 |
| Hamiltonian | ![ham_vid](/vid_saves/ham_vid_0.gif) | 140 | 5016.7 | 0.0279 |
| Optimised Hamiltonian | ![op_ham_vid](/vid_saves/op_ham_vid_0.gif) | 140 | 4585.5 | 0.0305 |
| DQN | ![dqn_vid](/vid_saves/dqn_vid_0.gif) | 33.36 | 392.9 | 0.0850 |
| QR-DQN | ![qrdqn_vid](/vid_saves/qrdqn_vid_0.gif) | 33.39 | 379.7| 0.0914 |
| A2C | ![a2c_vid](/vid_saves/a2c_vid_0.gif) | 19.18 | 178.6 | 0.107 |

### Random search
 - Randomly choose a direction
 - If there is something blocking, move in another direction

### Greedy search
 - One move horizon
 - Move in the direction of the food
 - If there is something blocking, move in another direction

### DFS
 - Depth-first search to find a complete path to the food
 - If there is no path to the food, use greedy search

### BFS
 - Breadth-first search to find a complete (and shortest) path to the food
 - If there is no path to the food, use greedy search

### Hamiltonian path
 - Follows a Hamiltonian path (a path that visits every node once)
 - Guarantees that the snake will never die
 - Calculated via longest path
 - Only works with even-sized grids

### Optimised Hamiltonian path
 - Uses Hamiltonian path but it takes shortcuts
 
 ![take_shortcuts](https://user-images.githubusercontent.com/80515759/226577899-467443c3-5982-4e40-bbf1-b6302377f951.png)

### Reinforcement learning
 - [StableBaselines3](https://stable-baselines3.readthedocs.io/en/master/) models
 - All models trained for 10,000,000 episodes
 - Hyperparameters based on those used on Atari games
 - Script downloads trained models automatically
### Training graphs
 - ![#425066](https://placehold.co/15x15/425066/425066.png) - DQN
 - ![#9334e6](https://placehold.co/15x15/9334e6/9334e6.png) - QR-DQN
 - ![#12b5cb](https://placehold.co/15x15/12b5cb/12b5cb.png) - A2C

Mean Episode Reward per Episode

![mean_reward](https://user-images.githubusercontent.com/80515759/226867371-3eb0bd7c-3856-4741-80e8-9108dd501691.png)

Mean Episode Length per Episode

![mean_length](https://user-images.githubusercontent.com/80515759/226867800-06953fa1-0410-43df-91e3-bb8518a4f38d.png)

## Gym environment
Uses a custom [Gymnasium](https://gymnasium.farama.org/) environment for snake game

Obsevation space: 
 - An RGB image of the game of shape (84, 84, 3)

Action space: 
 - A discrete space of 4 actions (up, left, down, right)

Rewards (modifiable):
 - +1 for eating food
 - -1 for dying
 - -0.001 for everything else

Actions are made with tensor operations in [PyTorch](https://pytorch.org/), inspired by [this Medium article](https://medium.com/@oknagg/learning-to-play-snake-at-1-million-fps-4aae8d36d2f1)

## Conclusion
Current RL approaches to snake are not very effective compared to algorithmic approaches. Successful RL approaches often have [heavy reward shaping](https://www.reddit.com/r/reinforcementlearning/comments/zfvyq1/ai_beats_snake_game_with_deep_qlearning/) (e.g. [distance to food](https://openreview.net/pdf?id=iu2XOJ45cxo)) or use observations besides the pure RGB display (e.g. [direction to food](https://ieeexplore.ieee.org/document/9480232)).

## To use
1. Install requirements
```
pip install -r requirements.txt
```
2. Run `benchmark.py` and pass optional arguments (defaults to greedy search without rendering)
```
python benchmark.py [-h] [--algo {greedy,random,bfs,dfs,ham,op_ham,dqn,qrdqn,a2c}] [--grid_size GRID_SIZE] [--initial_size INITIAL_SIZE] [--episodes EPISODES] [--show_render SHOW_RENDER] [--delay DELAY] [--save_gif SAVE_GIF]
```
3. Run respective `.py` file to train RL model (e.g. `python dqn.py` for training DQN model)

4. View tensorboard logs
```
tensorboard --logdir ./tensorboard/
```

### References
 - [Automated Snake Game Solvers via AI Search Algorithms (pdf)](https://bpb-us-e2.wpmucdn.com/sites.uci.edu/dist/5/1894/files/2016/12/AutomatedSnakeGameSolvers.pdf)
 - [chuyangliu/snake](https://github.com/chuyangliu/snake)