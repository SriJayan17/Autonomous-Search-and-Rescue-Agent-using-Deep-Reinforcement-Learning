from random import sample
import torch
from torch.autograd import Variable
import numpy as np

class Memory():
    """This class represents the memory of the agent's brain
    """
    
    def __init__(self,capacity):
        self.capacity = capacity
        self.memory = []
        
    def push(self,state):
        self.memory.append(state)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def pull(self,batch_size):
        ind = np.random.randint(0, len(self.memory), size=batch_size)
        batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []
        for i in ind: 
            state, next_state, action, reward, done = self.memory[i]
            batch_states.append(np.array(state, copy=False))
            batch_next_states.append(np.array(next_state, copy=False))
            batch_actions.append(np.array(action, copy=False))
            batch_rewards.append(np.array(reward, copy=False))
            batch_dones.append(np.array(done, copy=False))
        return np.array(batch_states), np.array(batch_next_states), np.array(batch_actions), np.array(batch_rewards).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1)