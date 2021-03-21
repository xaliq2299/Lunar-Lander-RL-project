import numpy as np
import keras
from keras.activations import relu, linear
from collections import deque
import gym
import random
from keras.utils import to_categorical
import matplotlib.pyplot as plt


class DeepQL:
	def __init__(self,learning_rate,batch_size):
		self.learning_rate = learning_rate
		self.epsilon = 1
		self.gamma = .99
		self.batch_size = batch_size
		self.memory = deque(maxlen=1000000) # change deque
		self.min_eps = 0.01

		# model
		self.model = keras.Sequential()
		self.model.add(keras.layers.Dense(512, input_dim=8, activation=relu))
		self.model.add(keras.layers.Dense(256, activation=relu))
		self.model.add(keras.layers.Dense(4, activation=linear))
		self.model.compile(loss="mse", optimizer=keras.optimizers.Adam(lr=learning_rate ))


	def fill_memory(self):
	    if len(self.memory) >= self.batch_size:
	        sample_choices = np.array(self.memory)
	        mini_batch_index = np.random.choice(len(sample_choices), self.batch_size)

	        states, actions, next_states, rewards, dones = [], [], [], [], []
	        for index in mini_batch_index:
	            states.append(self.memory[index][0])
	            actions.append(self.memory[index][1])
	            next_states.append(self.memory[index][2])
	            rewards.append(self.memory[index][3])
	            dones.append(self.memory[index][4])
	        states = np.array(states)
	        actions = np.array(actions)
	        next_states = np.array(next_states)
	        rewards = np.array(rewards)
	        dones = np.array(dones)
	        states = np.squeeze(states)
	        next_states = np.squeeze(next_states)
	        next_state_vals = self.model.predict_on_batch(next_states)
	        cur_state_vals = self.model.predict_on_batch(states)
	        cur_state_vals[np.arange(self.batch_size), actions] = rewards + self.gamma * (np.amax(next_state_vals, axis=1)) * (1 - dones)
	        self.model.fit(states, cur_state_vals, verbose=0)
	        if self.epsilon > self.min_eps:
	            self.epsilon *= 0.996


	def train(self, env, nbEpisodes):
	    np.random.seed(0)
	    scores, mean_scores = [], []
	    episodes = [k for k in range(0,nbEpisodes)]
	    for i in range(1,nbEpisodes+1):
	        score = 0
	        state = env.reset()
	        finished = False
	        if i != 0 and i % 50 == 0:
	            self.model.save("saved_models/"+"weights_"+str(i)+"_episodes.h5")
	        for j in range(3000):
	            state = np.reshape(state, (1, 8))
	            if np.random.random() <= self.epsilon:
	                action =  np.random.choice(4)
	            else:
	                action_values = self.model.predict(state)
	                action = np.argmax(action_values[0])

	            # env.render()
	            next_state, reward, finished, info = env.step(action)
	            next_state = np.reshape(next_state, (1, 8))
	            self.memory.append((state, action, next_state, reward, finished))
	            self.fill_memory()
	            score += reward
	            state = next_state
	            if finished:
	                scores.append(score)
	                mean_scores.append(np.mean(scores[-100:]))
	                print("Episode = {}, Score = {}, Avg_Score = {}".format(i+1, score, np.mean(scores[-100:])))
	                break

	    plt.xlabel("Episode")
	    plt.ylabel("Average score")
	    plt.plot(episodes, mean_scores)
	    plt.savefig("DeepQL.png")
