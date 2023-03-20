# SnakeAI
Single agent Snake AI using RL and search algorithms

Uses [Gymnasium](https://gymnasium.farama.org/) and [StableBaselines3](https://stable-baselines3.readthedocs.io/en/master/)

This uses a 12x12 grid, with the snake having initial size of 4

## Algorithms
Each algorithm is tested for 1000 episodes
| Algorithm | Average score | Average steps | Average score per step |
| --- | --- | --- | --- |
| Random | 2.45 | 424.08 | 0.01 |
| Greedy | 19.90 | 194.89 | 0.11 |
| DQN | 28.01 | 314.11 | 0.09 |

### Random search
 - One move horizon
 - Randomly choose a direction
 - If there is something blocking, move in another direction

### Greedy search
 - One move horizon
 - Move in the direction of the food
 - If there is something blocking, move in another direction

### DQN
 - [StableBaselines3](https://stable-baselines3.readthedocs.io/en/master/) DQN model to predict the best move
 - DQN trained for 10,000,000 episodes
 - Hyperparameters in `dqn.py`
 - Download trained DQN model [here]() or let it download automatically when running benchmark

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

## To use
1. Install requirements
```
pip install -r requirements.txt
```
2. Run `benchmark.py` and pass optional arguments (defaults to greedy search without rendering) for benchmarking
```
python benchmark.py [-h] [--algo {greedy,random,dqn}] [--grid_size GRID_SIZE] [--initial_size INITIAL_SIZE] [--episodes EPISODES] [--show_render SHOW_RENDER] [--delay DELAY]
```
3. Run `dqn.py` to train a DQN model
```
python dqn.py
```
4. View tensorboard logs
```
tensorboard --logdir ./tensorboard/
```

## To do
 - [ ] Add more algorithms (A*, Hamiltonian path, etc.)
 - [ ] Finish training other models
    - [ ] QRDQN
    - [ ] PPO
    - [ ] Recurrent PPO
    - [ ] A2C