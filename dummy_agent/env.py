import gym
from gym import envs
import random
import time

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


#Our Agent
class Agent:
    def __init__(self, env):
        self.action_size = env.action_space.n
    
    def get_action_random(self):
        action = random.choice(range(self.action_size))
        return action

    def get_action_smart_decision(self, observation):
        angle = observation[4]*57
        #print("----y: ", observation[1])
        rotate_right = [2,3]
        rotate_left = [1,2]
        
        #fix height to be able to move
        if observation[1]<0.6 and (observation[0]<-0.15 or observation[0]>0.1):
            return 2
        
        if angle>2: #rotate right to fix
            action = random.choice(rotate_right)
            return action
        
        if angle<-2: #rotate left to fix
            action = random.choice(rotate_left)
            return action
        
        #fix x axis)
        if observation[0]<-0.1:
            return 3
        
        if observation[0]>0.1:
            return 1
        else:
            return 0
            
        


#Env creation
env = gym.make('LunarLander-v2')
print("Observation space: ", env.observation_space)
print("Action space: ", env.action_space)



#initiazlize Agent
agent = Agent(env)



nb_episodes = 10
total_reward = 0

for i in range(nb_episodes):
    observation = env.reset()

    #looping
    episode_reward = 0
    for i_episode in range(300):
        
        action = agent.get_action_smart_decision(observation)
        
        observation, reward, done, info = env.step(action)
        episode_reward += reward
        env.render()
        
        #sleep to be able to see
        time.sleep(0.025)

        if done:
            total_reward += reward
            break
    print("Reward for episode", i+1, "is", episode_reward)


env.close()

print("Nb of episodes ", nb_episodes)
print("Average reward : ", total_reward/nb_episodes)
