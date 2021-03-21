import gym
from deep_ql import DeepQL
import tensorflow as tf
import numpy as np
from keras.activations import relu, linear


def train_model():
    learning_rate = 0.001
    batch_size = 64
    deep_ql = DeepQL(learning_rate, batch_size)
    env = gym.make('LunarLander-v2')
    num_episodes = 400
    deep_ql.train(env, num_episodes)



def test_model(filename, nbEpisodes=10, render=False):
    learning_rate = 0.001
    batch_size = 64
    deep_ql = DeepQL(learning_rate, batch_size)
    deep_ql.model.load_weights(filename)

    env = gym.make('LunarLander-v2')
    env._max_episode_steps = 300
    action_dim = env.action_space.n
    observation_dim = env.observation_space.shape

    episode_reward = 0
    counter = 0

    for _ in range(nbEpisodes):
        state = env.reset()
        done = False
        state = np.reshape(state, (1, 8))
        while not done:
            if render:
                env.render()
            next_state, reward, done, _ = env.step(np.argmax(deep_ql.model.predict(np.expand_dims(state, axis=0))))
            next_state = np.reshape(next_state, (1, 8))
            state = next_state
            episode_reward += reward
        print("Episode ", counter+1)
        counter += 1
    print('Average Reward:', episode_reward / nbEpisodes)



if __name__ == '__main__':
    # train_model()
    test_model("best_weights.h5", 10, True)
