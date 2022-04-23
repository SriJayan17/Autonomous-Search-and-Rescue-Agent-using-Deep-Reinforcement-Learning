from .Memory import Memory
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as functional

class Network(nn.Module):
    """This class represents the neural network of the brain that is used
    to make decisions, given the parameters
    """
    def __init__(self,nb_inputs,nb_actions):
        super(Network,self).__init__()
        self.nb_inputs = nb_inputs
        self.nb_outputs = nb_actions
        self.first_connection = nn.Linear(self.nb_inputs,30)
        self.second_connection = nn.Linear(30,40)
        self.third_connection = nn.Linear(40,self.nb_outputs)
        
    def forward(self,state):
        fc1_activated = functional.relu(self.first_connection(state))
        fc2_activated = functional.relu(self.second_connection(fc1_activated))
        q_values = self.third_connection(fc2_activated)
        return q_values

class DQNBrain():
    """This class represents the brain of the agent that uses deep Q-learning algorithm.
    """
    def __init__(self,input_nodes,nb_actions,gamma):
        self.gamma = gamma
        self.reward_mean = []
        self.memory = Memory(100000)
        self.model = Network(input_nodes,nb_actions)
        self.optimizer = optim.Adam(self.model.parameters(),lr=0.01)
        self.last_state = torch.Tensor(input_nodes).unsqueeze(0)
        self.last_reward = 0
        self.last_action = 0
        
    def __select_action(self,state):
        probs = functional.softmax(self.model.forward(Variable(state,volatile=True))*100)
        action = probs.multinomial(1)
        return action.data[0,0]
    
    def __learn(self,prev_state,current_state,prev_action,prev_reward):
        outputs = self.model.forward(prev_state).gather(1,prev_action.unsqueeze(1)).squeeze(1)
        max_futures = self.model.forward(current_state).detach().max(1)[0]
        targets = self.gamma*max_futures + prev_reward
        loss = functional.smooth_l1_loss(outputs,targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update(self,prev_reward,current_state):
        new_state = torch.Tensor(current_state).float().unsqueeze(0)
        #Pushing a collective state object to memory:
        self.memory.push((self.last_state,new_state,torch.LongTensor([int(self.last_action)]),torch.Tensor([self.last_reward])))
        #Passing the current_state to the neural network to the neurl network:    
        action = self.__select_action(new_state)
        
        if len(self.memory.memory) > 100:
            train_last_state,train_next_state,train_last_action,train_last_reward = self.memory.pull(100)
            self.__learn(train_last_state,train_next_state,train_last_action,train_last_reward)
        
        self.last_state = new_state
        self.last_action = action
        self.last_reward = prev_reward
        
        #Keeping a track of the agent's rewards accumulated over time to record it's performance:
        self.reward_mean.append(prev_reward)
        if len(self.reward_mean) > 1000:
            del self.reward_mean[0]
        return action