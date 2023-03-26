import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
  
  def __init__(self, state_dim, action_dim, max_action_vec):
    super(Actor, self).__init__()
    self.layer_1 = nn.Linear(state_dim, 128)
    self.layer_2 = nn.Linear(128, 64)
    self.layer_3 = nn.Linear(64, action_dim)
    self.max_action = torch.Tensor(max_action_vec)
    
  def forward(self, x):
    x = F.relu(self.layer_1(x))
    x = F.relu(self.layer_2(x))
    x = torch.tanh(self.layer_3(x))
    x = self.max_action *  x
    return x