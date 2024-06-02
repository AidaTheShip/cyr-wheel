import random
import gym 
import gymB
env = gym.make('CyrWheel-v0')

# Training Loop
episodes = 10
epsilon = 0.1 # This is the exploration factor for the Cyr Wheel

# for i in range(1, episodes+1): 
#     state = env.reset() # resetting the states of the environment after each episode 
#     done = False # The epsiode is not done yet. The learning / trial has not happened yet. 
#     reward = 0 # The overall reward is 0 before something has been done. 
    
#     while not done: # while the simulation is not done
#         action = random.choice(0,1) # the choice for the agent to move the center of mass left or right (by how much though?)
#         next_state = None

