#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 12:36:22 2021

@author: ayham
"""

import gym
from gym import envs
import random
import time
from learning import Learning
from deep_ql import DeepQLearning
import numpy as np


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


# Environment creation
env = gym.make('LunarLander-v2')
env._max_episode_steps = 300 # decide


#np.random.seed(1234)
# RenderWrapper.register(env, force_gif=True)
#env.seed(1234)

model = DeepQLearning()


dim = env.observation_space.shape[0]

# Train the agent
target, i, average_returns = model.train(env)
print("Solved after {} iterations".format(i))

# model.test(env)


env.close()
