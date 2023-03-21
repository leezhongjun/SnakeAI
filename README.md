# SnakeAI
Single agent Snake AI using RL and search algorithms

Uses [Gymnasium](https://gymnasium.farama.org/) and [StableBaselines3](https://stable-baselines3.readthedocs.io/en/master/)

This uses a 12x12 grid, with the snake having initial size of 4

RL algorithms implemented: DQN, QR-DQN, PPO, Recurrent PPO, A2C

## Algorithms
Each algorithm is tested for 1000 episodes
| Algorithm | Demo | Average score | Average steps | Average score per step |
| --- | --- | --- | --- | --- |
| Random | ![random_vid](/vid_saves/random_vid_0.gif) | 2.45 | 424.08 | 0.01 |
| Greedy | ![greedy_vid](/vid_saves/greedy_vid_0.gif) | 19.90 | 194.89 | 0.11 |
| DFS | ![dfs_vid](/vid_saves/dfs_vid_0.gif) | 20.13 | 531.34 | 0.04 |
| BFS | ![bfs_vid](/vid_saves/bfs_vid_0.gif) | 31.35 | 323.61 | 0.10 |
| Hamiltonian | ![ham_vid](/vid_saves/ham_vid_0.gif) | 140.00 | 5016.69 | 0.03 |
| DQN | ![dqn_vid](/vid_saves/dqn_vid_0.gif) | 33.36 | 392.85 | 0.09 |

### Random search
 - Randomly choose a direction
 - If there is something blocking, move in another direction

### Greedy search
 - One move horizon
 - Move in the direction of the food
 - If there is something blocking, move in another direction

### DFS
 - Depth first search to find a complete path to the food
 - If there is no path to the food, use greedy search

### BFS
 - Breadth first search to find a complete (and shortest) path to the food
 - If there is no path to the food, use greedy search

### Hamiltonian path
 - Follows a Hamiltonian path (a path that visits every node once)
 - Guarantees that the snake will never die
 - Calculated via longest path
 - Only works with even-sized grids

### DQN
 - [StableBaselines3](https://stable-baselines3.readthedocs.io/en/master/) DQN model to predict the best move
 - DQN trained for 10,000,000 episodes
 - Hyperparameters in `dqn.py` and based on those used on Atari games
 - Download trained DQN model [here](https://www.dropbox.com/s/mt1y5xh6z4s6pn4/dqn_snake.zip?raw=1) (script downloads it automatically)

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

Actions are made with tensor operations, inspired by [this Medium article](https://medium.com/@oknagg/learning-to-play-snake-at-1-million-fps-4aae8d36d2f1)

## Conclusion
Current RL approaches to snake are not very effective compared to algorithmic approaches. Successful RL approaches often have [heavy reward shaping](https://www.reddit.com/r/reinforcementlearning/comments/zfvyq1/ai_beats_snake_game_with_deep_qlearning/) (e.g. [distance to food](https://openreview.net/pdf?id=iu2XOJ45cxo)) or use observations besides the pure RGB display (e.g. [direction to food](https://ieeexplore.ieee.org/document/9480232)).

## To use
1. Install requirements
```
pip install -r requirements.txt
```
2. Run `benchmark.py` and pass optional arguments (defaults to greedy search without rendering)
```
python benchmark.py [-h] [--algo {greedy,random,bfs,dfs,ham,dqn}] [--grid_size GRID_SIZE] [--initial_size INITIAL_SIZE] [--episodes EPISODES] [--show_render SHOW_RENDER] [--delay DELAY] [--save_gif SAVE_GIF]
```
3. Run `dqn.py` to train a DQN model
```
python dqn.py
```
4. View tensorboard logs
```
tensorboard --logdir ./tensorboard/
```

### To do
 - [ ] Add more algorithms (A*, etc.)
 - [ ] Finish training other models
    - [ ] QR-DQN
    - [ ] PPO
    - [ ] Recurrent PPO
    - [ ] A2C

### References
 - [Automated Snake Game Solvers via AI Search Algorithms (pdf)](https://bpb-us-e2.wpmucdn.com/sites.uci.edu/dist/5/1894/files/2016/12/AutomatedSnakeGameSolvers.pdf)
 - [chuyangliu/snake](https://github.com/chuyangliu/snake)