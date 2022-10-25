import gym
from ray.rllib.algorithms.ppo import PPO

# color print
from colorama import Fore, Back, Style

debug = True
step_count = 0

# problem definition:
# 	a corridor in which an agent must move right to reach the exit
# state/observation space: 	corridor_length = 5, S = start position, G = goal position
# action space: 		0 = move left, 1 = move right
# reward space: 		-0.1 for every step, +1 for reaching goal position
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


	# run everytime 1 episode is completed
	# each episode has many trials/steps until success
	# returns new start state/observation, reward, and done status
	def reset(self):
		# tracks how many steps until done = True and environment is reset
		global step_count

		# reset S to leftmost position		
		self.cur_pos = 0

		if debug:
			print(Fore.BLACK + Back.RED + "resetting env... step_count = " + str(step_count) + Style.RESET_ALL)
		step_count = 0

		return [self.cur_pos]


	# run everytime 1 action is taken
	# returns a new state/observation 
	def step(self, action):
		# tracks how many steps until done = True and environment is reset
		global step_count

		# go left if not in leftmost position
		if (action == 0 and self.cur_pos > 0):
			self.cur_pos -= 1
		# go right 
		elif action == 1:
			self.cur_pos += 1
		# done = True when cur_pos >= end_pos
		done = self.cur_pos >= self.end_pos
		# reward = 1 when reaching end_pos and -0.1 at every step otherwise
		reward = 1.0 if done else -0.1

		if not debug:
			print(Fore.BLACK + Back.GREEN + "STEP: action =  " + str(action) + ", cur_pos = " + str(self.cur_pos) + ", reward = " + str(reward) + ", done = " + str(done) + Style.RESET_ALL)
		step_count += 1

		return [self.cur_pos], reward, done, {}



# defining the training algorithm
# PPO is the mathematical model
algo = PPO(
	config = {
		"env": SimpleCorridor,
		"env_config": {
			"corridor_length": 20,
		},
		"num_workers": 1,
	}
)



# training the above algorithm to go from S = 0 to G = 20
for i in range(5):
	# algo.train() runs multiple episodes per iteration until some criteria is satisfied
	# e.g. criteria can be a bound on the runtime or the number of steps/actions taken
	results = algo.train()
	print(Fore.BLACK + Back.BLUE + f"Iter: {i}, avg_reward = {results['episode_reward_mean']}" + Style.RESET_ALL)



# creating a new environment and state space
# and using the above trained algorithm to solve the corridor traversal problem in it
env = SimpleCorridor({"corridor_length": 10})
obs = env.reset()
done = False
total_reward = 0.0
while not done:
	# given state = obs, compute action
	action = algo.compute_single_action(obs)
	# take a step given action
	obs, reward, done, info = env.step(action)
	# compute reward
	total_reward += reward

print(Fore.BLACK + Back.GREEN + f"Played 1 episode until done = True, total reward = {total_reward}")


