import torch
import torch.nn.functional as F

from Project.Backend.Brains.TD3.Actor import Actor
from Project.Backend.Brains.TD3.Critic import Critic
from Project.Backend.Brains.TD3.Memory import ReplayBuffer

# Selecting the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Building the whole Training Process into a class

class TD3(object):
  
  def __init__(self, state_dim, action_dim, max_action_vec, mem_capacity, expl_noise=0.1):
    self.actor = Actor(state_dim, action_dim, max_action_vec).to(device)
    self.actor_target = Actor(state_dim, action_dim, max_action_vec).to(device)
    self.actor_target.load_state_dict(self.actor.state_dict())
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
    self.critic = Critic(state_dim, action_dim).to(device)
    self.critic_target = Critic(state_dim, action_dim).to(device)
    self.critic_target.load_state_dict(self.critic.state_dict())
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
    self.max_action = max_action_vec
    self.memory = ReplayBuffer(mem_capacity)
    self.prev_reward = None
    self.prev_state = None 
    self.prev_action = None
    self.expl_noise = expl_noise
    self.state_dim = state_dim
    self.action_dim = action_dim

  def select_action(self, state, prev_reward, done=False):
    #Note: This function has to be called for the last state (succesful/failure) of the training
    #process, even if there is no action to take. This is to store that last (s,s',a,r,d) in memory.
    # print(state)
    state = torch.Tensor(state.reshape(1, -1)).to(device)
    self.prev_reward = prev_reward
    if self.prev_state is not None:
      self.memory.add((self.prev_state,state,self.prev_action,self.prev_reward,done))
      if done: return
    self.prev_state = state
    self.prev_action = self.actor(state).cpu().data.numpy().flatten() 
    return self.prev_action

  def train(self, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
    
    for it in range(iterations):
      
      # Sampling records to train
      batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = self.memory.sample(batch_size)
      state = torch.Tensor(batch_states).to(device)
      next_state = torch.Tensor(batch_next_states).to(device)
      action = torch.Tensor(batch_actions).to(device)
      reward = torch.Tensor(batch_rewards).to(device)
      done = torch.Tensor(batch_dones).to(device)
      
      # Obtain a' for s' from actor_target
      next_action = self.actor_target(next_state)
      
      # Addition of gaussian noise to a'
      noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
      noise = noise.clamp(-noise_clip, noise_clip)
      next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
      
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
  
  # Making a save method to save a trained model
  def save(self, filename, directory):
    torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
    torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
  
  # Making a load method to load a pre-trained model
  def load(self, filename, directory):
    self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
    self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))