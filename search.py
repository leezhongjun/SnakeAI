from custom_env import HEAD, BODY, FOOD, SnakeEnv
import torch, collections
import numpy as np

UP, LEFT, DOWN, RIGHT = 0, 1, 2, 3

dir_to_offset = torch.tensor([[-1, 0], [0, -1], [1, 0], [0, 1]])

path = None
hamiltonian_grid = None
path_h = None

def greedy_search_helper(raw_obs):
    '''
    Greedy search helper function
    Returns coordinate priority list
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
    
    return cood_priority

def greedy_search(raw_obs):
    '''
    Greedy search on raw observation
    '''
    cood_priority = greedy_search_helper(raw_obs)
    head_coord = raw_obs[HEAD].nonzero()[0]
            
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
            return path
        for x2, y2 in ((x+1,y), (x-1,y), (x,y+1), (x,y-1)):
            if 0 <= x2 < raw_obs.shape[2] and 0 <= y2 < raw_obs.shape[2] and raw_obs[BODY][y2][x2] == 0 and (y2, x2) not in seen:
                q.append(path + [(y2, x2)])
                seen.add((y2, x2))
    
    return [[]]


def bfs_search(raw_obs):
    '''
    BFS search on raw observation
    '''
    global path
    if not path:
        path = bfs_helper(raw_obs)[1:]
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
            return path
        for x2, y2 in ((x+1,y), (x-1,y), (x,y+1), (x,y-1)):
            if 0 <= x2 < raw_obs.shape[2] and 0 <= y2 < raw_obs.shape[2] and raw_obs[BODY][y2][x2] == 0 and (y2, x2) not in seen:
                s.append(path + [(y2, x2)])
                seen.add((y2, x2))
    
    return [[]]

def dfs_search(raw_obs):
    '''
    DFS search on raw observation
    '''
    global path
    if not path:
        path = dfs_helper(raw_obs)[1:]
        if not path:
            return greedy_search(raw_obs)
        
    head_coord = raw_obs[HEAD].nonzero()[0]
    res = torch.where(dir_to_offset == (torch.tensor(path[0]) - head_coord))[0][1]
    path = path[1:]
    return int(res)

def longest_path_helper(raw_obs):
    '''
    Helper function for longest_path_search
    '''
    path = bfs_helper(raw_obs)

    if not path:
        return [[]]
    path[0] = tuple(path[0])
    seen = set(path)
    path = np.array(path)
    
    i = 0
    
    while True:
        
        cur = path[i]
        nx = path[i+1]
        inserted = False
        
        if nx[0] == cur[0]:
            # Extend vertically
            if cur[0] + 1 < raw_obs.shape[2] and raw_obs[BODY][cur[0] + 1, cur[1]] == 0 and raw_obs[BODY][nx[0] + 1, nx[1]] == 0:
                # Extend down
                if (cur[0] + 1, cur[1]) not in seen and (nx[0] + 1, nx[1]) not in seen:
                    path = np.insert(path, i+1, [cur[0] + 1, cur[1]], axis=0)
                    path = np.insert(path, i+2, [nx[0] + 1, nx[1]], axis=0)
                    seen.add((cur[0] + 1, cur[1]))
                    seen.add((nx[0] + 1, nx[1]))
                    inserted = True
            if not inserted and cur[0] - 1 >= 0 and raw_obs[BODY][cur[0] - 1, cur[1]] == 0 and raw_obs[BODY][nx[0] - 1, nx[1]] == 0:
                # Extend up
                if (cur[0] - 1, cur[1]) not in seen and (nx[0] - 1, nx[1]) not in seen:
                    path = np.insert(path, i+1, [cur[0] - 1, cur[1]], axis=0)
                    path = np.insert(path, i+2, [nx[0] - 1, nx[1]], axis=0)
                    seen.add((cur[0] - 1, cur[1]))
                    seen.add((nx[0] - 1, nx[1]))
                    inserted = True

        elif nx[1] == cur[1]:
            # Extend horizontally
            if cur[1] + 1 < raw_obs.shape[2] and raw_obs[BODY][cur[0], cur[1] + 1] == 0 and raw_obs[BODY][nx[0], nx[1] + 1] == 0:
                # Extend right
                if (cur[0], cur[1] + 1) not in seen and (nx[0], nx[1] + 1) not in seen:
                    path = np.insert(path, i+1, [cur[0], cur[1] + 1], axis=0)
                    path = np.insert(path, i+2, [nx[0], nx[1] + 1], axis=0)
                    seen.add((cur[0], cur[1] + 1))
                    seen.add((nx[0], nx[1] + 1))
                    inserted = True
            if not inserted and cur[1] - 1 >= 0 and raw_obs[BODY][cur[0], cur[1] - 1] == 0 and raw_obs[BODY][nx[0], nx[1] - 1] == 0:
                # Extend left
                if (cur[0], cur[1] - 1) not in seen and (nx[0], nx[1] - 1) not in seen:
                    path = np.insert(path, i+1, [cur[0], cur[1] - 1], axis=0)
                    path = np.insert(path, i+2, [nx[0], nx[1] - 1], axis=0)
                    seen.add((cur[0], cur[1] - 1))
                    seen.add((nx[0], nx[1] - 1))
                    inserted = True
        
        if not inserted:
            i += 1
            if i + 1 >= len(path):
                break
            
    return path

def hamiltonian_path_helper(raw_obs):
    r_obs = torch.zeros(raw_obs.shape)
    r_obs[BODY][0, 1] = 1
    r_obs[HEAD][0, 1] = 1
    r_obs[FOOD][0, 0] = 1
    path = np.array(longest_path_helper(r_obs))
    path = np.append(path, np.array([[0,1],[0,2]])).reshape(-1, 2)
    return path

def hamiltonian_path_search(raw_obs):
    '''
    Hamiltonian path search on raw observation
    '''
    global path
    if path is None:
        path = hamiltonian_path_helper(raw_obs)
        
    head_coord = raw_obs[HEAD].nonzero()[0]
    i = np.where((path == head_coord.numpy()).all(axis=1))[0][0]
    j = i+1
    if j > len(path): j=0
    res = torch.where(dir_to_offset == (torch.tensor(path[j]) - head_coord))[0][1]
    return int(res)

def bfs_ham_helper(raw_obs):
    '''
    Helper function for bfs_ham_search
    '''
    print(hamiltonian_grid)
    head_coord = raw_obs[HEAD].nonzero()[0].tolist()
    head_val = hamiltonian_grid[head_coord[0], head_coord[1]]

    tail_coord = (raw_obs[BODY] == 1).nonzero()[0]
    tail_val = hamiltonian_grid[tail_coord[0], tail_coord[1]]

    food_coord = tuple(raw_obs[FOOD].nonzero()[0])
    q = collections.deque([[head_coord]])
    seen = set(head_coord)
    while q:
        path = q.popleft()
        y, x = path[-1]
        if (y, x) == food_coord:
            return path
        for x2, y2 in ((x+1,y), (x-1,y), (x,y+1), (x,y-1)):
            if 0 <= x2 < raw_obs.shape[2] and 0 <= y2 < raw_obs.shape[2] and raw_obs[BODY][y2][x2] == 0 and (y2, x2) not in seen:
                val = hamiltonian_grid[y2, x2]
                print(val, y2,x2, tail_val, head_val)
                if (val < tail_val and val > head_val and head_val < tail_val) or ((val < tail_val or val > head_val) and head_val > tail_val):
                    
                    q.append(path + [(y2, x2)])
                    seen.add((y2, x2))
    
    return [[]]

def optimised_hamiltonian_path_search(raw_obs):
    '''
    Optimised hamiltonian path search on raw observation (takes shortcuts)
    '''
    global path, hamiltonian_grid, path_h

    if path_h is None:
        path_h = hamiltonian_path_helper(raw_obs)

    if hamiltonian_grid is None:
        hamiltonian_grid = np.zeros((raw_obs.shape[2],raw_obs.shape[2]))
        for i, x in enumerate(path_h[:-2]):
            hamiltonian_grid[x[0], x[1]] = i

    print(hamiltonian_grid)

    coord_priority = greedy_search_helper(raw_obs)[:2]
    
    tail_coord = (raw_obs[BODY] == 1).nonzero()[0]
    tail_val = hamiltonian_grid[tail_coord[0], tail_coord[1]]

    head_coord = raw_obs[HEAD].nonzero()[0]
    head_val = hamiltonian_grid[head_coord[0], head_coord[1]]

    food_coord = raw_obs[FOOD].nonzero()[0]
    food_val = hamiltonian_grid[food_coord[0], food_coord[1]]

    best_dir = None

    for direction in coord_priority:
        new_head = head_coord + dir_to_offset[direction]

        if new_head.min() < 0:
            continue
        elif new_head.max() >= raw_obs.shape[2]:
            continue

        elif raw_obs[BODY][tuple(new_head)] == 0:
            val = hamiltonian_grid[new_head[0], new_head[1]]


            if food_val == val:
                return direction

            if head_val > tail_val:
                if val > head_val and val > tail_val:
                    best_dir = direction
                    break
                elif val < head_val and val < tail_val:
                    best_dir = direction
                    break
            elif head_val < tail_val:
                if val > head_val and val < tail_val:
                    best_dir = direction
                    break

    
    # print(val, head_val, tail_val, best_dir)
    if best_dir is None:
        i = np.where((path_h == head_coord.numpy()).all(axis=1))[0][0]
        j = i+1
        if j > len(path_h): j=0
        res = torch.where(dir_to_offset == (torch.tensor(path_h[j]) - head_coord))[0][1]
        return int(res)
    else:
        return best_dir
            
    # tail_coord = (raw_obs[BODY] == 1).nonzero()[0]
    # tail_val = hamiltonian_grid[tail_coord[0], tail_coord[1]]

    # head_coord = raw_obs[HEAD].nonzero()[0]
    # head_val = hamiltonian_grid[head_coord[0], head_coord[1]]

    # food_coord = raw_obs[FOOD].nonzero()[0]
    # food_val = hamiltonian_grid[food_coord[0], food_coord[1]]

    

    # best_dir = -1

    # for direction in range(4):
    #     new_head = head_coord + dir_to_offset[direction]

    #     if new_head.min() < 0:
    #         continue
    #     elif new_head.max() >= raw_obs.shape[2]:
    #         continue

    #     if raw_obs[BODY][tuple(new_head)] == 0:
    #         new_head_val = hamiltonian_grid[new_head[0], new_head[1]]

    #         if food_val == new_head_val:
    #             return direction

    #         if head_val > tail_val+1:
    #             if new_head_val > head_val+1 or new_head_val+1 < tail_val:
    #                 best_dir = direction
    #         else:
    #             if new_head_val > head_val+1 and new_head_val+1 < tail_val:
    #                 best_dir = direction

    # if best_dir == -1:
    #     i = np.where((path == head_coord.numpy()).all(axis=1))[0][0]
    #     j = i+1
    #     if j > len(path): j=0
    #     res = torch.where(dir_to_offset == (torch.tensor(path[j]) - head_coord))[0][1]
    #     return int(res)
    
    # return int(best_dir)