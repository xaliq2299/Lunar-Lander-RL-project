import numpy as np
from scipy.special import softmax

import torch
import torch.nn as nn
import torch.nn.functional as F

NOTHING = 0
RIGHT = 1
MAIN = 2
LEFT = 3

def greedy_policy(state, model):
    action = torch.argmax(model(state))

    return action


def epsilon_greedy_policy(state, model, epsilon):
    if np.random.rand() < epsilon:
        action = np.random.randint(4)
    else:
        action = greedy_policy(state, model)

    return action

def compute_policy_gradient(episode_states, episode_actions, episode_rewards, theta):
    ### BEGIN SOLUTION ###
    H = len(episode_rewards)
    PG = 0
    for t in range(H):
        prob_push_right = softmax(np.dot(episode_states[t], np.transpose(theta)))
        a_t = episode_actions[t]
        R_t = sum(episode_rewards[t::])
        g_theta_log_pi = np.empty(4)
        if a_t == LEFT:
        	for i in range(len(prob_push_right)):
        		print("prob_push_left", prob_push_right[i])
        		print("Episode states", episode_states[t])
        		print("Rt", R_t)
        		g_theta_log_pi[i] = - prob_push_right[i] * episode_states[t] * R_t
        else:
            prob_push_left = (1 - prob_push_right)
            # print("I m here")
            # print("Right=", prob_push_right)
            # print("Left=", prob_push_left)
            for i in range(len(prob_push_left)):
            	# print("prob_push_left", prob_push_left[i])
            	# print("Episode states", episode_states[t])
            	# print("Rt", R_t)
            	g_theta_log_pi[i] = prob_push_left[i] * episode_states[t] * R_t
        PG += g_theta_log_pi
    ### END SOLUTION ###
    return PG

class Q_function(nn.Module):
    
    def __init__(self):
        super(Q_function, self).__init__()
        self.l1 = nn.Linear(8,24)
        self.l2 = nn.Linear(24,24)
        self.l3 = nn.Linear(24,4)
        
    def forward(self, x):
        device = x.device
        y = F.relu(self.l1(x)).to(device)
        y = F.relu(self.l2(y))
        y = self.l3(y)
        return y