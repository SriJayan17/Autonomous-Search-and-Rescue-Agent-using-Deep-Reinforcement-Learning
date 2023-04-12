import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
  
  def __init__(self, state_dim, action_dim, max_action_vec):
    super(Actor, self).__init__()
    self.layer_1 = nn.Linear(state_dim, 128)
    # self.layer_2 = nn.Linear(128, 256)
    # self.layer_3 = nn.Linear(256,64)
    self.layer_2 = nn.Linear(128, action_dim)
    self.max_action = torch.Tensor(max_action_vec) if action_dim > 1 else max_action_vec
    
  def forward(self, x):
    x = F.relu(self.layer_1(x))
    # x = F.relu(self.layer_2(x))
    # x = F.relu(self.layer_3(x))
    x = torch.tanh(self.layer_2(x))
    x = self.max_action *  x
    return x
  
import numpy as np
if __name__ == '__main__':
  sample = Actor(3,1,15)
  input = torch.Tensor([[1.56,2.78,3.34]])
  output = sample(input).data.numpy().flatten()
  print(f'Actual output: {output}')
  # output = (output + np.random.normal(0, 0.1, size=output.shape[0])).clip(-15,15)
  # print(f'Noisy output: {output}')