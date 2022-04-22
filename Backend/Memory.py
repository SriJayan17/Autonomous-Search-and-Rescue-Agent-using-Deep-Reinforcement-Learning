from random import sample
import torch
from torch.autograd import Variable

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
        rand_sample = zip(*sample(self.memory,batch_size))
        return map(lambda x:Variable(torch.cat(x,0)),rand_sample)