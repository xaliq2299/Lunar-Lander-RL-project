import torch
import torch.nn as nn
import torch.nn.functional as F # Hyper-parameters
from scipy.special import softmax
import numpy as np
import copy
import torch.optim as optim
import random
import pickle

from policy import Q_function


ENV_NAME = "LunarLander-v2"

# Since the goal is to attain an average return of 195, horizon should be larger than 195 steps (say 300 for instance)
EPISODE_DURATION = 1000

GAMMA = 0.999

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

TARGET_UPDATE = 10

ALPHA_INIT = 0.1

SCORE = 195.0
NUM_EPISODES = 100

NOTHING = 0
RIGHT = 1
MAIN = 2
LEFT = 3

VERBOSE = True



# Our Deep Q-Learning model
class DeepQLearning():
    def __init__(self):
        pass


    def greedy_policy(self, state, model):
        action = torch.argmax(model(state))

        return action.cpu().numpy()


    def epsilon_greedy_policy(self, state, model, epsilon):
        if np.random.rand() < epsilon:
            action = np.random.randint(4)
        else:
            action = self.greedy_policy(state, model)

        return action

    # Generate an episode
    def play_one_episode(self, env, model, max_episode_length=EPISODE_DURATION, render=False):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        s_t = torch.from_numpy(env.reset()).to(device)

        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_states.append(s_t)

        for t in range(max_episode_length):
            if render:
                env.render()
            
            a_t = self.greedy_policy(s_t, model)
            s_t, r_t, done, info = env.step(a_t)
            s_t = torch.from_numpy(s_t).to(device)

            episode_states.append(s_t)
            episode_actions.append(a_t)
            episode_rewards.append(r_t)

            if done:
                break

        return episode_states, episode_actions, episode_rewards


    def score_on_multiple_episodes(self, env, model, score=SCORE, num_episodes=NUM_EPISODES, max_episode_length=EPISODE_DURATION, render=False):
        ### BEGIN SOLUTION ###
        num_success = 0
        average_return = 0
        num_consecutive_success = [0]
        for episode_index in range(num_episodes):
            _, _, episode_rewards = self.play_one_episode(env, model, max_episode_length, render)
            total_rewards = sum(episode_rewards)
            if total_rewards >= score:
                num_success += 1
                num_consecutive_success[-1] += 1
            else:
                num_consecutive_success.append(0)
            average_return += (1.0 / num_episodes) * total_rewards
            if render:
                print("Test Episode {0}: Total Reward = {1} - Success = {2}".format(episode_index,total_rewards,total_rewards>score))
        if max(num_consecutive_success) >= 100:    # MAY BE ADAPTED TO SPEED UP THE LERNING PROCEDURE
            success = True
        else:
            success = False
        ### END SOLUTION ###
        return success, num_success, average_return

    def select_epsilon(self, ep):
        steps_done = ep % EPS_DECAY
        return EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)


    # Train the agent got an average reward greater or equals to 195 over 100 consecutive trials
    def train(self, env, epsilon = 0.2, max_episode_length = EPISODE_DURATION, alpha_init = ALPHA_INIT):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        target = Q_function().to(device)
        current = Q_function().to(device)
        buffer = []

        optimizer = optim.Adam(current.parameters(), lr = 0.0001)
        criterion = nn.MSELoss().to(device)

        
        
        """with open("buffer.txt", "rb") as fp:   # Unpickling
            buffer = pickle.load(fp)"""

        

        chp = torch.load("model_3layer_best_3.pt")
        target.load_state_dict(chp['model_state_dict'])
        target.eval()
        current.load_state_dict(chp['model_state_dict'])
        current.train()
        optimizer.load_state_dict(chp['optimizer_state_dict'])

        average_returns = []

        best_average_returns = 0

        # Train until success
        for ep in range(1000):

            print("episode : ", ep)

            s_t = env.reset()
            s_t = torch.from_numpy(s_t).to(device)

            epsilon = self.select_epsilon(ep)
            print("epsilon : ", epsilon)

            for t in range(max_episode_length):

                a_t = self.epsilon_greedy_policy(s_t, current, epsilon)
                s_t_next, r_t, done_t, info = env.step(a_t)

                buffer.append((s_t.cpu().numpy(), a_t, r_t, s_t_next, done_t))

                s_t = torch.from_numpy(s_t_next).to(device)

                s_j, a_j, r_j, s_j_next, done_j = random.sample(buffer,1)[0]
                s_j = torch.from_numpy(s_j).to(device)
                s_j_next = torch.from_numpy(s_j_next).to(device)

                y_j = torch.as_tensor(np.array(r_j), dtype = torch.float).to(device)
                if not done_j:
                    with torch.no_grad():
                        y_j += GAMMA * torch.max(target(s_j_next))

                predicted = current(s_j)[a_j]

                loss = criterion(y_j, predicted)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if done_t:
                    break

                """for p in target.parameters():
                    print(p.name, p.data)"""

            """if episode_index % 20 == 0:
                s_t = env.reset()
                s_t = torch.from_numpy(s_t).to(device)
                for t in range(max_episode_length):
                    a_t = self.greedy_policy(s_t, target)
                    print(target(s_t))
                    s_t_next, r_t, done_t, info = env.step(a_t)
                    s_t = torch.from_numpy(s_t_next).to(device)
                    env.render()"""

            if ep % TARGET_UPDATE == 0:
                target = copy.deepcopy(current).to(device)
                target.eval()

                with torch.no_grad():
                    success, _, R = self.score_on_multiple_episodes(env, target, render=False, num_episodes=10)
                    average_returns.append(R)

                print("average returns : ", R)
                if best_average_returns < R:
                    best_average_returns = R
                    print("best average returns : ", best_average_returns)
                    torch.save({'model_state_dict': target.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, 'model_3layer_best_4n.pt')

        torch.save({'model_state_dict': target.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, 'model_3layer_4n.pt')

        with open("buffer_2.txt", "wb") as fp:   #Pickling
            pickle.dump(buffer, fp)

        return target, 1000, average_returns

    
    def test(self, env):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = Q_function().to(device)

        chp = torch.load("model_3layer_best_0.pt")
        model.load_state_dict(chp['model_state_dict'])
        model.eval()

        success, _, R = self.score_on_multiple_episodes(env, model, num_episodes=50, render=True)
        print("average reward : ", R)

        return R



# model = VAE().to(device)
# model = DeepQLearning()

# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# print(model)
