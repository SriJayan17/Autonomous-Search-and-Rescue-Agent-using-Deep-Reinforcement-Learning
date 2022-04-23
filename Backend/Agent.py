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
        return self.brain.update(prev_reward,current_state)
            
        