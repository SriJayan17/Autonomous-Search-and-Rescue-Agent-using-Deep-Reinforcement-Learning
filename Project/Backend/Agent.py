# from Brains.DQN_brain import DQNBrain
import math
import numpy as np
import pygame
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler

from Project.Backend.Brains.TD3.Model import TD3
from Project.FrontEnd.Utils.Action_Handler import isPermissible

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

        self.temp_memory = []
        self.scaler = MinMaxScaler()
        self.timer = 0

        # self.prev_rect = self.rect
        # self.prev_shape_copy = self.shape_copy

        self.prev_center = self.rect.center

        self.angle = 0

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
        # self.prev_rect = self.rect.copy()
        # self.prev_shape_copy = self.shape_copy.copy()
        #Updating the rect object of the players:
        self.rect = temp_rect
        self.shape_copy = temp_image

    # To move forward
    def move(self,dist=10):
        # temp_rect = self.rect.copy()
        old_center = self.rect.center
        #Modifying the angle for convenience:
        ref_angle = 90 + self.angle
        ref_angle = (math.pi/180) * ref_angle
        #Updating the center:
        new_center = (int(old_center[0] + dist*math.cos(ref_angle)),int(old_center[1] - dist*math.sin(ref_angle)))  
        self.prev_center = self.rect.center
        self.rect.center = new_center
        
    def restore_move(self):
        self.rect.center = self.prev_center

    def take_action(self,prev_reward,current_state,is_over):
        """Get the action to be taken from the agent

        Args:
            prev_reward (float): The recent reward received by the agent
            current_state (list): An iterable containing the parameters of the environment

        Returns:
            Action (int): The discreet action to be taken by the agent. Return values :[0,1,2...n actions]
        """
        #Preprocessing of the parameters:

        if self.timer > 100:
            # Scale the input and use neural network for decision
            current_state = self.scaler.transform([current_state])[0]
            return self.brain.select_action(np.array(current_state),prev_reward,is_over)
        elif self.timer < 100:
            # Take store record, increment timer, random action
            self.temp_memory.append(current_state)
        else:
            # Fit the scaler, delete temp_memory, increment timer, random action
            self.scaler.fit(self.temp_memory)
            del self.temp_memory

        self.timer += 1
        return [np.random.randint(-15,15), np.random.randint(10,15)]
        
    
    # def save_brain(self):
    #     self.brain.save_nn()
    
    # def plot_reward_metric(self):
    #     self.brain.plot_rewards()
    # def __check_and_load_brain(self):

    def __hash__(self):
        return hash(str(self))
        
        