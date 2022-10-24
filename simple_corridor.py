import gym
from ray.rllib.algorithms.ppo import PPO
from colorama import Fore, Back, Style

debug = True
step_count = 0

# problem definition:
# 	a corridor in which an agent must move right to reach the exit
# state space: 		corridor_length = 5, S = start position, G = goal position
# action space: 	0 = move left, 1 = move right
# reward space: 	-0.1 for every step, +1 for reaching goal position
class SimpleCorridor(gym.Env):
	def __init__(self, config):
		# goal position = rightmost position
		self.end_pos = config["corridor_length"]
		# start position = leftmost position
		self.cur_pos = 0
		# |action_space| = 2 (left or right move)
		self.action_space = gym.spaces.Discrete(2)
		# |state_space| = end_pos - 0
		self.observation_space = gym.spaces.Box(0.0, self.end_pos, shape=(1,))
		if debug:
			print(Fore.BLACK + Back.RED + "end_pos =  "  + str(self.end_pos) + Style.RESET_ALL)
			#print(self.end_pos)
			print(Fore.BLACK + Back.RED + "action_space =  "  + Style.RESET_ALL)
			print(self.action_space)
			print(Fore.BLACK + Back.RED + "observation_space = " + Style.RESET_ALL)
			print(self.observation_space)

	def reset(self):
		global step_count
		if debug:
			print(Fore.BLACK + Back.RED + "resetting env... step_count = " + str(step_count) + Style.RESET_ALL)
			#print("resetting env...")
		self.cur_pos = 0
		step_count = 0
		return [self.cur_pos]

	def step(self, action):
		global step_count
		if (action == 0 and self.cur_pos > 0):
			self.cur_pos -= 1
		elif action == 1:
			self.cur_pos += 1
		done = self.cur_pos >= self.end_pos
		reward = 1.0 if done else -0.1
		if not debug:
			print(Fore.BLACK + Back.GREEN + "STEP: action =  " + str(action) + ", cur_pos = " + str(self.cur_pos) + ", reward = " + str(reward) + ", done = " + str(done) + Style.RESET_ALL)
			#print(action)
			#print(Fore.BLACK + Back.WHITE + "cur_pos, reward, done = "  + str(self.cur_pos) + str(reward) + str(done) + Style.RESET_ALL)
			#print(self.cur_pos, reward, done)
		step_count += 1
		return [self.cur_pos], reward, done, {}

# defining the training algorithm
algo = PPO(
	config = {
		"env": SimpleCorridor,
		"env_config": {
			"corridor_length": 20,
			"debug": True,
		},
		"num_workers": 1,
	}
)

for i in range(5):
	results = algo.train()
	print(f"Iter: {i}, avg_reward = {results['episode_reward_mean']}")

env = SimpleCorridor({"corridor_length": 10})
obs = env.reset()
done = False
total_reward = 0.0

while not done:
	action = algo.compute_single_action(obs)
	obs, reward, done, info = env.step(action)
	total_reward += reward

print(f"Played 1 episode, total rewared = {total_reward}")


