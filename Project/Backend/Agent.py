from .Brains.DQN_brain import DQNBrain
import pygame

class Agent:
    """This class represents the agent itself
    """
    
    def __init__(self,num_inputs,num_actions,brain_type='DQN'):
        self.nb_inputs = num_inputs
        self.nb_actions = num_actions
        
        #Storage capacity of the brain of the agent:
        if brain_type.upper() == 'DQN':
            #Discount factor for calculating future actions:
            self.gamma = 0.9
            self.brain = DQNBrain(self.nb_inputs,self.nb_actions,self.gamma)
    
    def take_action(self,prev_reward,current_state):
        """Get the action to be taken from the agent

        Args:
            prev_reward (float): The recent reward received by the agent
            current_state (list || tuple): An iterable containing the parameters of the environment

        Returns:
            Action (int): The discreet action to be taken by the agent. Return values :[0,1,2...n actions]
        """
        current_state = list(current_state)
        current_state = current_state[2:]
        current_state[-1] /= 180
        current_state[2] /= 360
        current_state.append(current_state[-1] * -1)

        return self.brain.update(prev_reward,tuple(current_state))
    
    def save_brain(self):
        self.brain.save_nn()
    
    def plot_reward_metric(self):
        self.brain.plot_rewards()
    # def __check_and_load_brain(self):
        
        