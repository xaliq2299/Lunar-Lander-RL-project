# Run to create a model as the one we provided (about 10min to train)

import matplotlib
import matplotlib.pyplot as plt

import math
import gym
import numpy as np
import random
from gym import envs
import copy
import pandas as pd
import seaborn as sns
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

env = gym.make('LunarLander-v2')


def build_model():
    model = Sequential()
    model.add(Flatten(input_shape=(1,8)))
    model.add(Dense(32, activation = 'relu'))
    model.add(Dense(32, activation = 'relu'))
    model.add(Dense(4, activation='linear'))
    return model

model = build_model()

def build_agent(model):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit = 30000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                   nb_actions=4, nb_steps_warmup=10, target_model_update=0.01)
    return dqn

dqn = build_agent(model)
dqn.compile(Adam(lr=0.001), metrics=['mae'])
dqn.fit(env, nb_steps=70000, visualize=False, verbose=1)

scores = dqn.test(env, nb_episodes=10, visualize=False)
print(np.mean(scores.history['episode_reward']))

# To save the model
model.save('model_deepQL_other')

_ = dqn.test(env, nb_episodes=5, visualize=True)


