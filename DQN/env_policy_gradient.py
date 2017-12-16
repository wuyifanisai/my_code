import gym
from RL_brain_policy_gradient import Policy_Gradient as brain
import matplotlib.pyplot as plt
import numpy as np

env = gym.make('CartPole-v0')
env = env.unwrapped

env.seed(1)

# infomation about the observation
print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

# define the brain
RL = brain(
			n_actions = env.action_space.n,
			n_features = env.observation_space.shape[0],
			learning_rate = 0.02,
			reward_decay = 0.9
			)

RENDER = False 
# at the begging, env dose not render
DISPLAY_REWARD_THRESHOLD = 1000
# env render after running_reward is bigger than 400 

vt_list=[]
# store vt from every round

for i_round in range(900):
	
	observation = env.reset()
	
	while True:
		if RENDER: env.render()
		#env.render()
		
		action = RL.choose_action(observation)

		observation_ , reward, done, info = env.step(action)

		RL.store_transition(observation, action, reward)
		
		if done:
			ep_reward_sum = sum(RL.ep_rs)

			if 'running_reward' not in globals():
				running_reward = ep_reward_sum
			else:
				running_reward = running_reward*0.1 + ep_reward_sum*0.9

			if running_reward > DISPLAY_REWARD_THRESHOLD:
				RENDER = True

			print("round",i_round,"--->reward:",running_reward)

			vt = RL.learn()
			vt_list.append(ep_reward_sum)

			break

		observation = observation_



plt.plot(vt_list)
plt.show()



			


