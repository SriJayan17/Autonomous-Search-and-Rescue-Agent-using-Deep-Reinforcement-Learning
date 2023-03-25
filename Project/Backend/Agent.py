# from Brains.DQN_brain import DQNBrain
import math
import numpy as np
from Project.Backend.Brains.TD3.Model import TD3
import pygame

class Agent:
    """This class represents the agent itself
    """

    def __init__(self,num_inputs,num_actions, action_limits, initial_position, memory=100,expl_noise=0.1):
        self.nb_inputs = num_inputs
        self.nb_actions = num_actions
        #Initial Coordinates
        # self.X = 90
        # self.Y = 620
        
        self.original_shape = pygame.Surface((15,30))
        self.original_shape.fill((0,0,255))  # Color of the agent
        self.original_shape.set_colorkey((0,0,0))

        self.shape_copy = self.original_shape.copy()

        self.rect = self.shape_copy.get_rect()
        self.rect.center = initial_position

        self.prev_rect = self.rect
        self.prev_shape_copy = self.shape_copy

        self.angle = 0

        
        #Storage capacity of the brain of the agent:
        # if brain_type.upper() == 'DQN':
            #Discount factor for calculating future actions:
            # self.gamma = 0.9
        # self.brain = DQNBrain(self.nb_inputs,self.nb_actions,self.gamma)
        self.brain = TD3(num_inputs,num_actions,action_limits,memory,expl_noise)
    
    # Turn the agent
    def turn(self,turn_angle=15):
        # Right 
        if turn_angle >= 0:
            if self.angle <= 0:
                self.angle = -1 * ((abs(self.angle) + abs(turn_angle)) % 360)
            else:
                self.angle = self.angle - abs(turn_angle)
        # Left
        elif turn_angle < 0:
            if self.angle >=0:
                self.angle = (self.angle + abs(turn_angle)) % 360
            else:
                self.angle = self.angle + abs(turn_angle)
        
        #Applying the transformation:
        temp_image = pygame.transform.rotate(self.original_shape,self.angle)
        # old_center = self.rect.center
        temp_rect = temp_image.get_rect()
        temp_rect.center = self.rect.center
        #Saving the previous state:
        self.prev_rect = self.rect
        self.prev_shape_copy = self.shape_copy
        #Updating the rect object of the players:
        self.rect = temp_rect
        self.shape_copy = temp_image

    # To move forward
    def move(self,dist=10):
        temp_rect = self.rect.copy()
        old_center = temp_rect.center
        #Modifying the angle for convenience:
        ref_angle = 90 + self.angle
        ref_angle = (math.pi/180) * ref_angle
        #Updating the center:
        new_center = (int(old_center[0] + dist*math.cos(ref_angle)),int(old_center[1] - dist*math.sin(ref_angle)))  
        temp_rect.center = new_center
        # reward = -1
        #Saving the current rect:
        self.prev_rect = self.rect
        #Updating the current player's rect:
        self.rect = temp_rect
    
    # To restore the state of the agent(only for the past one timestep)
    def restore(self):
        self.shape_copy = self.prev_shape_copy
        self.rect = self.prev_rect

    def take_action(self,prev_reward,current_state,is_over):
        """Get the action to be taken from the agent

        Args:
            prev_reward (float): The recent reward received by the agent
            current_state (list || tuple): An iterable containing the parameters of the environment

        Returns:
            Action (int): The discreet action to be taken by the agent. Return values :[0,1,2...n actions]
        """
        #Preprocessing of the parameters:
        # current_state = list(current_state)
        # current_state = current_state[2:]
        # current_state[-1] /= 180
        # current_state[2] /= 360
        # current_state.append(current_state[-1] * -1)

        # return self.brain.select_action(np.array(current_state),prev_reward,is_over)
        return [np.random.randint(-15,15), np.random.randint(10,15)]
    
    # def save_brain(self):
    #     self.brain.save_nn()
    
    # def plot_reward_metric(self):
    #     self.brain.plot_rewards()
    # def __check_and_load_brain(self):

    def __hash__(self):
        return hash(str(self))
        
        