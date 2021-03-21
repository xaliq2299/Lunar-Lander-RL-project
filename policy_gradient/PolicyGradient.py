import torch.nn as nn
import torch.nn.functional as F # Hyper-parameters
from scipy.special import softmax
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical


ENV_NAME = "LunarLander-v2"

# Since the goal is to attain an average return of 195, horizon should be larger than 195 steps (say 300 for instance)
EPISODE_DURATION = 200 #300
EPISODE_NUMBER = 4000000  #each episode take 10
TARGET_UPDATE = 10

ALPHA_INIT = 0.0001
SCORE = 195.0
NUM_EPISODES = 10 #10
NOTHING = 0
RIGHT = 1
MAIN = 2
LEFT = 3

DISCOUNT_FACTOR = 0.95
VERBOSE = True



class Policy(nn.Module):
	def __init__(self):
		super(Policy, self).__init__()
		self.l1 = nn.Linear(8,10)
		#self.l3 = nn.Linear(10,10)
		self.l2 = nn.Linear(10,4)
		
		
	def forward(self, x):
		device = x.device
		y = F.relu(self.l1(x)).to(device)
		#y = F.relu(self.l2(y)).to(device)
		y = F.softmax(self.l2(y),dim=0)
		return y



# Our Deep Q-Learning model
class PolicyGradientLearning():
	def __init__(self):
		pass


	# Generate an episode
	def play_one_episode(self, env, policy, max_episode_length=EPISODE_DURATION, render=False):
		device = torch.device('cpu')

		s_t = torch.from_numpy(env.reset()).to(device)

		episode_states = []
		episode_actions = []
		episode_rewards = []
		episode_states.append(s_t)

		for t in range(max_episode_length):
			if render:
				env.render_wrapper.render()
			
			a_t = self.draw_action(s_t, policy)
			s_t, r_t, done, info = env.step(a_t)
			s_t = torch.from_numpy(s_t).to(device)

			episode_states.append(s_t)
			episode_actions.append(a_t)
			episode_rewards.append(r_t)

			if done:
				break

		return episode_states, episode_actions, episode_rewards


	def score_on_multiple_episodes(self, env, policy, score=SCORE, num_episodes=NUM_EPISODES, max_episode_length=EPISODE_DURATION, render=False):
		### BEGIN SOLUTION ###
		num_success = 0
		average_return = 0
		num_consecutive_success = [0]
		for episode_index in range(num_episodes):
			_, _, episode_rewards = self.play_one_episode(env, policy, max_episode_length, render)
			total_rewards = sum(episode_rewards)
			# if total_rewards >= score:
			# 	num_success += 1
			# 	num_consecutive_success[-1] += 1
			# else:
			# 	num_consecutive_success.append(0)
			average_return += (1.0 / num_episodes) * total_rewards
			if render:
				print("Test Episode {0}: Total Reward = {1} - Success = {2}".format(episode_index,total_rewards,total_rewards>score))
		if average_return >= 200:    # MAY BE ADAPTED TO SPEED UP THE LERNING PROCEDURE
			success = True
		else:
			success = False
		### END SOLUTION ###
		return success, num_success, average_return


	def draw_action(self, s, policy):
		#Need to create a model here 
		#print(s)
		action_probability_distribution = policy(s).numpy()
		#print(action_probability_distribution)
		#print(action_probability_distribution)
		action = np.random.choice(range(action_probability_distribution.shape[0]), 
                                  p=action_probability_distribution.ravel())
		#print(s)
		# if action ==0:
		# 	print(action)
		return action

    	
	# Train the agent got an average reward greater or equals to 195 over 100 consecutive trials
	def train(self, env, episode_number = EPISODE_NUMBER, max_episode_length = EPISODE_DURATION, alpha_init = ALPHA_INIT):
		device = torch.device('cpu')
		text_all = []
		alpha = alpha_init
		X = []
		Y = []
		policy = Policy().to(device)

# 		chp = torch.load("policy_2layer_best_0.pt")
# 		policy.load_state_dict(chp['model_state_dict'])

		optimizer = optim.Adam(policy.parameters(), lr = alpha)

		average_returns = []

		best_average_return = 0

		with torch.no_grad():
			policy.eval()
			success, _, R = self.score_on_multiple_episodes(env, policy)
			average_returns.append(R)
			best_average_return = R

		# Train until success
		success = False
		for ep in range(episode_number):
			if success:
				break
			#print("epsisode : ", ep)

			# Rollout
			with torch.no_grad():
				policy.eval()
				episode_states, episode_actions, episode_rewards = self.play_one_episode(env, policy, max_episode_length)
			episode_rewards = np.array(episode_rewards)

			policy.train()

			J = 0
			n = len(episode_actions)

			for t in range(n):
				#can also normalize the reward
				#episode_rewards[t:] -= np.mean(episode_rewards[t:])
				#episode_rewards[t:] /= np.std(episode_rewards[t:])+0.00001

				Gt = torch.as_tensor(np.sum(episode_rewards[t:]), dtype = torch.float).to(device)

				logPolicy = torch.log(policy(episode_states[t])[episode_actions[t]])

				J += -Gt * logPolicy*DISCOUNT_FACTOR #-Gt

			J.backward()
			optimizer.step()

			if ep % TARGET_UPDATE == 0:
				policy.eval()

				with torch.no_grad():
					success, _, R = self.score_on_multiple_episodes(env, policy, render=False, num_episodes=10)
					average_returns.append(R)
				#print()
				print(int(ep/10), " average returns : ", R)
				text_all.append(str(int(ep/10))+ " average returns : " + str(R))
				X.append(int(ep/10))
				Y.append(R)
				if int(ep/10)%200 ==0 and int(ep/10) != 0:
					plt.plot(X, Y)
					plt.xlabel('Episode')
					plt.ylabel('average reward')
					# plt.show()
					plt.savefig('plots/avgTrain_vs_episode'+str(int(ep/10))+'.png')
					
					print("Saved averages at ", int(ep/10))
					with open('txt/run_' + str(int(ep/10)) + '.txt', 'w') as filehandle:
						for listitem in text_all:
							filehandle.write('%s\n' % listitem)

				if best_average_return < R:
					best_average_return = R
					print("------------------------------------------------------------Saving model with average return : ", best_average_return)
					torch.save({'model_state_dict': policy.state_dict()}, 'saved_model.pt')
					#print(text_all)

		return policy, episode_number, average_returns, text_all


