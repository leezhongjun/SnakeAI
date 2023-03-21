from custom_env import HEAD, BODY, FOOD, SnakeEnv
import torch, collections
import numpy as np

UP, LEFT, DOWN, RIGHT = 0, 1, 2, 3

dir_to_offset = torch.tensor([[-1, 0], [0, -1], [1, 0], [0, 1]])

path = None

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

def bfs_helper(raw_obs):
    '''
    Helper function for bfs_search
    '''
    head_coord = raw_obs[HEAD].nonzero()[0].tolist()
    food_coord = tuple(raw_obs[FOOD].nonzero()[0])
    q = collections.deque([[head_coord]])
    seen = set(head_coord)
    while q:
        path = q.popleft()
        y, x = path[-1]
        if (y, x) == food_coord:
            return path[1:]
        for x2, y2 in ((x+1,y), (x-1,y), (x,y+1), (x,y-1)):
            if 0 <= x2 < raw_obs.shape[2] and 0 <= y2 < raw_obs.shape[2] and raw_obs[BODY][y2][x2] == 0 and (y2, x2) not in seen:
                q.append(path + [(y2, x2)])
                seen.add((y2, x2))
    
    return None


def bfs_search(raw_obs):
    '''
    BFS search on raw observation
    '''
    global path
    if not path:
        path = bfs_helper(raw_obs)
        if not path:
            return greedy_search(raw_obs)
        
    head_coord = raw_obs[HEAD].nonzero()[0]
    res = torch.where(dir_to_offset == (torch.tensor(path[0]) - head_coord))[0][1]
    path = path[1:]
    return int(res)

def dfs_helper(raw_obs):
    '''
    Helper function for dfs_search
    '''
    head_coord = raw_obs[HEAD].nonzero()[0].tolist()
    food_coord = tuple(raw_obs[FOOD].nonzero()[0])
    s = [[head_coord]]
    seen = set(head_coord)
    while s:
        path = s.pop()
        y, x = path[-1]
        if (y, x) == food_coord:
            return path[1:]
        for x2, y2 in ((x+1,y), (x-1,y), (x,y+1), (x,y-1)):
            if 0 <= x2 < raw_obs.shape[2] and 0 <= y2 < raw_obs.shape[2] and raw_obs[BODY][y2][x2] == 0 and (y2, x2) not in seen:
                s.append(path + [(y2, x2)])
                seen.add((y2, x2))
    
    return None

def dfs_search(raw_obs):
    '''
    DFS search on raw observation
    '''
    global path
    if not path:
        path = dfs_helper(raw_obs)
        if not path:
            return greedy_search(raw_obs)
        
    head_coord = raw_obs[HEAD].nonzero()[0]
    res = torch.where(dir_to_offset == (torch.tensor(path[0]) - head_coord))[0][1]
    path = path[1:]
    return int(res)