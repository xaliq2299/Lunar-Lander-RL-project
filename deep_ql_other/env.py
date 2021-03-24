import gym
import tensorflow as tf
import numpy as np
from keras.activations import relu, linear
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


def build_agent(model):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit = 30000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                   nb_actions=4, nb_steps_warmup=10, target_model_update=0.01)
    return dqn


def test_model(filename, nbEpisodes=10, render=False):
    learning_rate = 0.001
    model_loaded = load_model(filename)
    model_loaded = load_model('DeepQL_model')
    dqn = build_agent(model_loaded)
    dqn.compile(Adam(lr=learning_rate), metrics=['mae'])
    env = gym.make('LunarLander-v2')
    scores = dqn.test(env, nb_episodes=10, visualize=render)
    
    
    
    print('Average reward :',np.mean(scores.history['episode_reward']))
    



if __name__ == '__main__':
    # train_model()
    test_model("DeepQL_model", 10, True)
