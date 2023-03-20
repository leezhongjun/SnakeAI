from custom_env import HEAD, BODY, FOOD
import torch
import numpy as np

UP, LEFT, DOWN, RIGHT = 0, 1, 2, 3

dir_to_offset = torch.tensor([[-1, 0], [0, -1], [1, 0], [0, 1]])

def greedy_search(raw_obs):
    '''
    Greedy search on raw observation
    '''
    # get direction towards food
    food_coord = raw_obs[FOOD].nonzero()[0]
    head_coord = raw_obs[HEAD].nonzero()[0]
    cood_priority = []
    if food_coord[0] < head_coord[0]:
        # move up
        cood_priority.append(UP)
    elif food_coord[0] > head_coord[0]:
        # move down
        cood_priority.append(DOWN)
    
    if food_coord[1] < head_coord[1]:
        # move left
        cood_priority.append(LEFT)
    elif food_coord[1] > head_coord[1]:
        # move right
        cood_priority.append(RIGHT)

    for direction in (UP, DOWN, RIGHT, LEFT):
        if direction not in cood_priority:
            cood_priority.append(direction)
            
    for direction in cood_priority:
        if (head_coord + dir_to_offset[direction]).min() < 0:
            continue
        elif (head_coord + dir_to_offset[direction]).max() >= raw_obs.shape[2]:
            continue
        elif raw_obs[BODY][tuple(head_coord + dir_to_offset[direction])] == 0:
            return direction
        
    return 0


def random_search(raw_obs):
    actions = np.arange(4)
    np.random.shuffle(actions)
    head_coord = raw_obs[HEAD].nonzero()[0]
    for direction in actions:
        if (head_coord + dir_to_offset[direction]).min() < 0:
            continue
        elif (head_coord + dir_to_offset[direction]).max() >= raw_obs.shape[2]:
            continue
        elif raw_obs[BODY][tuple(head_coord + dir_to_offset[direction])] == 0:
            return direction
    return 0