import gymnasium as gym
from gym import spaces
import numpy as np
import torch
import torch.nn.functional as F
import cv2, imageio, os

LEFT = torch.tensor([[ 0,  0,  0], [ 0, -1,  1], [ 0,  0,  0]], dtype=torch.float32)
RIGHT = torch.tensor([[ 0,  0,  0], [ 1, -1,  0], [ 0,  0,  0]], dtype=torch.float32)
UP = torch.tensor([[ 0,  0,  0], [ 0, -1,  0], [ 0,  1,  0]], dtype=torch.float32)
DOWN = torch.tensor([[ 0,  1,  0], [ 0, -1,  0], [ 0,  0,  0]], dtype=torch.float32)

ACTIONS = torch.stack([UP, LEFT, DOWN, RIGHT])

OPP_DIRS = {0: 2, 1: 3, 2: 0, 3: 1}

HEAD = 2
BODY = 1
FOOD = 0

FOOD_REWARD = 1 #1
DEATH_REWARD = -1 #-0.5
ALIVE_REWARD = -0.001 #-0.005

class SnakeEnv(gym.Env):
	metadata = {
        'render_mode': ['human']
    }
	 
	def __init__(self, mode='bot', grid_size=20, initial_size=2, save_video=False, alive_reward=ALIVE_REWARD, death_reward=DEATH_REWARD, food_reward=FOOD_REWARD):
		'''Set the action and observation spaces'''

		super().__init__()
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.mode = mode
		self.grid_size = grid_size
		self.action_space = spaces.Discrete(4)
		self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)	# 3 spaces (food, body, head)
		
		self.obs = torch.zeros(3, self.grid_size, self.grid_size)
		self.intitial_size = initial_size
		self.dir = 3 # Inital direction is right
		self.done = False
		self.size = self.intitial_size
		self.alive_reward = alive_reward
		self.death_reward = death_reward
		self.food_reward = food_reward
		self.reward = self.alive_reward
		self.window_name = None
		self.food_coord = (0, 0)
		self.image_lst = []
		self.save_video = save_video

	def spawn_food(self):
		'''Spawn food in a random empty location'''

		self.obs[FOOD][self.food_coord[0], self.food_coord[1]] = 0
		empty = torch.where(self.obs[BODY] == 0)
		empty = torch.stack(empty, dim=1)
		self.food_coord = empty[torch.randint(0, len(empty), (1,))].squeeze()
		self.obs[FOOD][self.food_coord[0], self.food_coord[1]] = 1

	def reset(self):
		'''Return -> starting state of obs'''

		self.obs = torch.zeros(3, self.grid_size, self.grid_size)
		self.dir = 3 # Inital direction is right
		self.done = False
		self.size = self.intitial_size
		self.reward = self.alive_reward
		self.image_lst = []

		# Place head in the middle
		self.obs[HEAD][self.grid_size//2, self.grid_size//2] = 1

		# Place body based on size
		for i in range(self.size):
			self.obs[BODY][self.grid_size//2, self.grid_size//2-i] = self.size - i

		# Place food in a random location
		self.spawn_food()

		if self.save_video:
			img = self.get_rgb()
			self.image_lst.append(img)

		return self.get_rgb()

	def move_in_direction(self, direction):
		'''Move the head in the given direction'''

		# Set new direction
		if direction !=  OPP_DIRS[self.dir]:
			self.dir = direction

		self.reward = self.alive_reward
		# Move head
		self.obs[HEAD] = F.conv2d(self.obs[HEAD].unsqueeze(0).unsqueeze(0), ACTIONS[self.dir].unsqueeze(0).unsqueeze(0), padding=1).squeeze(0).squeeze(0)
		# Apply relu
		self.obs[HEAD] = F.relu(self.obs[HEAD])

		# Check if head reached wall
		if self.obs[HEAD].sum() == 0:
			self.done = True
			self.reward = self.death_reward
			return
		
		# Check if head reached body
		if (self.obs[HEAD] * self.size + self.obs[BODY]).max() > self.size + 1: # Not 0 because body is going to move forward
			self.done = True
			self.reward = self.death_reward
			return
		
		# Check if head reached food
		if (self.obs[HEAD] + self.obs[FOOD]).max() > 1:
			self.size += 1
			self.reward = self.food_reward
			if self.size == self.grid_size * self.grid_size:
				self.done = True
				return
			self.spawn_food()
			# Body doesnt move forward if head reached food
		else:
			# Move body forward
			self.obs[BODY] -= 1
			self.obs[BODY] = F.relu(self.obs[BODY])

		# Add head to body
		self.obs[BODY] += self.obs[HEAD] * self.size
			
	def step_for_algo(self, action):
		'''Take action & update the env
		Return -> next-state(raw), reward, done'''

		self.move_in_direction(action)

		if self.save_video:
			img = self.get_rgb()
			self.image_lst.append(img)

		return self.get_raw_obs(), self.reward, self.done

	def step(self, action):
		'''Take action & update the env
		Return -> next-state, reward, done, info'''

		self.move_in_direction(action)

		if self.save_video:
			img = self.get_rgb()
			self.image_lst.append(img)

		return self.get_rgb(), self.reward, self.done, {}
	
	def get_raw_obs(self):
		'''Return a copy of the obs'''

		return self.obs.clone()
	
	def get_rgb(self):
		'''Return the obs in RGB format'''

		# Convert to numpy
		img = self.obs.clone()
		img[BODY] = torch.minimum(img[BODY], torch.ones(img[BODY].shape))
		img = (img * 255).permute(1, 2, 0).numpy()
		img = np.uint8(img)
		img = cv2.resize(img, (84, 84), interpolation = cv2.INTER_AREA)
		return img

	def get_score(self):
		return self.size - self.intitial_size

	def render(self, mode='human'):
		'''Render the env on the screen'''
		
		if self.window_name is None:
			self.window_name = 'Snake'
			cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
			
		if mode=='human':
			img = self.get_rgb()
			cv2.imshow(self.window_name, img)
			cv2.waitKey(1)

		

	def save_video_func(self, video_name='video.gif'):
		'''Saves the video of the game'''
		if not os.path.exists('vid_saves'):
			os.makedirs('vid_saves')

		if self.save_video:
			imageio.mimsave(f'vid_saves/{video_name}', self.image_lst, fps=60)

