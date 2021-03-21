#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 12:36:22 2021

@author: ayham
"""

import torch
import gym
#from gym import envs
#import random
import time
#from learning import Learning
from PolicyGradient import PolicyGradientLearning
import numpy as np
import matplotlib.pyplot as plt
from PolicyGradient import Policy

#All Envs in gym
#print(envs.registry.all())

#--------------------------------------------LunarLander
#Action space:
    #0: to nothing
    #1: fire right engine
    #2: fire main engine
    #3: fire left engine


 # state = [
 #            (pos.x - VIEWPORT_W/SCALE/2) / (VIEWPORT_W/SCALE/2),
 #            (pos.y - (self.helipad_y+LEG_DOWN/SCALE)) / (VIEWPORT_H/SCALE/2),
 #            vel.x*(VIEWPORT_W/SCALE/2)/FPS,
 #            vel.y*(VIEWPORT_H/SCALE/2)/FPS,
 #            self.lander.angle,
 #            20.0*self.lander.angularVelocity/FPS,
 #            1.0 if self.legs[0].ground_contact else 0.0,
 #            1.0 if self.legs[1].ground_contact else 0.0
 #            ]

#-------------------------------------------------------

def test_last_model(render=False):
    device = torch.device('cpu')
    policy = Policy().to(device)
    chp = torch.load("40_best_model.pt")
    policy.load_state_dict(chp['model_state_dict'])
    policy.eval()
    
    env = gym.make('LunarLander-v2')
    env._max_episode_steps = 300
    observation = env.reset()

    number_of_trials = 10
    average_reward = 0
    #print(policy.state_dict())
    for i in range(number_of_trials):
        observation = env.reset()
        done = False
        count = 0
        while not done: 
            count += 1
            if render:
                env.render()
                time.sleep(0.02)
            s_t = torch.from_numpy(observation).to(device)    
            #print(policy(s_t))
            action_probability_distribution = policy(s_t).detach().numpy()
            action = np.random.choice(range(action_probability_distribution.shape[0]), 
                                        p=action_probability_distribution.ravel())

            observation, reward, done, info = env.step(action)
            #print(env.step(action))
            
            
        
            if done:
                average_reward += reward
                #print(done, reward)
                
                break
    average_reward /= number_of_trials
    print("Average Reward of ", number_of_trials, " is: ", average_reward)
    env.close()



# Environment creation
env = gym.make('LunarLander-v2')
env._max_episode_steps = 100 # decide

model = PolicyGradientLearning()

# Train or test the agent
test_last_model(render=True)
#policy, episode_number, average_returns, text_all = model.train(env)

# with open('listfile.txt', 'w') as filehandle:
#     for listitem in text_all:
#         filehandle.write('%s\n' % listitem)