# INF581, Advanced Machine Learning and Autonomous agents: Lunar Lander project
### Introduction
In this project repo we study different agents for the [LunarLander-v2](https://gym.openai.com/envs/LunarLander-v2/) environment provided by openAi GYM. In this environment the agent must learn to land a small ship on the surface of the moon between two target flags. The input space state is continuous 8-dimensional and the action space consists of 4 main actions: LEFT engine, RIGHT engine, MAIN engine and nothing. We present to you the main 3 agents we studied during this project: DeepQl, Policy Gradient and dummy Non-RL agent.
### Requirements
### Running the project
Go the directory of the project where one can find three different models. To test any model (deep_ql, for instance), run the followings commands on the terminal:
```
cd deep_ql
python3 env.py
```