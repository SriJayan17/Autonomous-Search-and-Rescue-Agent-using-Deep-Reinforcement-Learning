import torch
import torch.nn.functional as F
import os

from Project.Backend.Brains.TD3.Actor import Actor
from Project.Backend.Brains.TD3.Critic import Critic
from Project.Backend.Brains.TD3.Memory import ReplayBuffer

# Selecting the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Building the whole Training Process into a class

class TD3(object):
  
  def __init__(self, state_dim, action_dim, max_action_vec, mem_capacity, load, load_path):
    self.actor = Actor(state_dim, action_dim, max_action_vec).to(device)
    self.critic = Critic(state_dim, action_dim).to(device)

    if load and load_path is not None: 
      if os.path.exists(os.path.join(os.getcwd(),load_path)):
        self.load(load_path)

    self.actor_target = Actor(state_dim, action_dim, max_action_vec).to(device)
    self.actor_target.load_state_dict(self.actor.state_dict())
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=0.0001)
    
    self.critic_target = Critic(state_dim, action_dim).to(device)
    self.critic_target.load_state_dict(self.critic.state_dict())
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=0.0001)
    self.max_action = max_action_vec
    # self.memory = ReplayBuffer(mem_capacity)
    self.state_dim = state_dim
    self.action_dim = action_dim

  def select_action(self, state):
    state = torch.Tensor(state.reshape(1, -1)).to(device)
    return self.actor(state).cpu().data.numpy().flatten()

  def train(self, memory, iterations, batch_size, discount, tau, policy_noise, noise_clip, policy_freq):
    for it in range(iterations):
      # Sampling records to train
      batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = memory.sample(batch_size)
      state = torch.Tensor(batch_states).to(device)
      # print(f'State: {state.shape}')
      next_state = torch.Tensor(batch_next_states).to(device)
      # print(f'next_state = {next_state.shape}')
      # print(f'batch_actions: {batch_actions}')
      action = torch.Tensor(batch_actions).to(device)
      # print(f'Action: {action.shape}')
      reward = torch.Tensor(batch_rewards).to(device)
      done = torch.Tensor(batch_dones).to(device)
      
      # Obtain a' for s' from actor_target
      next_action = self.actor_target(next_state)
      
      # Addition of gaussian noise to a'
      noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
      noise = noise.clamp(-noise_clip, noise_clip)
      
      next_action = (next_action + noise).clamp(-self.max_action,self.max_action)
      # upper_limit = torch.Tensor(self.max_action)
      # lower_limit = torch.Tensor([-i for i in self.max_action])
      # #Clipping the next_action tensor within the specified limits:
      # next_action = torch.max(torch.min(next_action,upper_limit),lower_limit)

      # The two Critic targets take each the couple (s’, a’) as input and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
      target_Q1, target_Q2 = self.critic_target(next_state, next_action)
      
      # target_Qval min(Qt1, Qt2)
      target_Q = torch.min(target_Q1, target_Q2)
      
      # Final target of the two Critic models, which is: Qt = r + γ * min(Qt1, Qt2), where γ is the discount factor
      target_Q = reward + ((1 - done) * discount * target_Q).detach()
      
      #The two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,a) as outputs
      current_Q1, current_Q2 = self.critic(state, action)
      
      #Loss of the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
      critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
      
      #Backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer
      self.critic_optimizer.zero_grad()
      critic_loss.backward()
      self.critic_optimizer.step()
      
      # Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model
      if it % policy_freq == 0:
        actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Still once every two iterations, we update the weights of the Actor target by polyak averaging
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        # Still once every two iterations, we update the weights of the Critic target by polyak averaging
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
  
  # Adding a record in the memory:
  # def add_record(self,record):
  #   self.memory.add(record)

  # Making a save method to save a trained model
  def save(self, target_path):
    torch.save(self.actor.state_dict(), f'{target_path}/actor.pth')
    torch.save(self.critic.state_dict(), f'{target_path}/critic.pth')
  
  # Making a load method to load a pre-trained model
  def load(self, target_path):
    try:
      self.actor.load_state_dict(torch.load( f'{target_path}/actor.pth'))
      self.critic.load_state_dict(torch.load( f'{target_path}/critic.pth'))
      print(f'Loaded actor and critic from: {target_path}')
    except Exception as e:
      return