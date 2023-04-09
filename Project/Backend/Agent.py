import math
import numpy as np
import pygame
from sklearn.preprocessing import MinMaxScaler

from Project.Backend.Brains.TD3.Model import TD3

class Agent:
    """This class represents the agent itself
    """

    def __init__(self,num_inputs,num_actions, action_limits, initial_position, memory=100, load=False, load_path=None):
        self.nb_inputs = num_inputs
        self.nb_actions = num_actions
    
        self.original_shape = pygame.Surface((12,30))
        self.original_shape.fill((0,0,255))  # Color of the agent
        self.original_shape.set_colorkey((0,0,0))

        self.shape_copy = self.original_shape.copy()
        self.previous_shape_copy = self.shape_copy.copy()

        self.rect = self.shape_copy.get_rect()
        self.rect.center = initial_position
        self.previous_rect = self.rect.copy()

        self.temp_memory = []
        self.scaler = MinMaxScaler()
        self.timer = 0

        self.prev_center = self.rect.center

        self.angle = 0

        self.brain = TD3(num_inputs,num_actions,action_limits,memory,load,load_path)
    
    # To get the proper angle value for the calculation of obstacle density
    def get_proper_angle(self):
        if self.angle < 0:
            self.angle += 360
        final_angle = (self.angle + 90) % 360
        return final_angle 

    # Turn the agent
    def turn(self,turn_angle=15):
        rad = (turn_angle * math.pi)/180
        # Right -> Clockwise 
        if turn_angle >= 0:
            if self.angle <= 0:
                self.angle = -1 * ((abs(self.angle) + abs(turn_angle)) % 360)
            else:
                self.angle = self.angle - abs(turn_angle)
            
        # Left -> AntiClockwise
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
        #Updating the rect object of the players:
        self.rect = temp_rect
        self.shape_copy = temp_image

    # To move forward
    def move(self,dist=10):
        self.previous_rect = self.rect.copy()

        old_center = tuple(self.rect.center)
        #Modifying the angle for convenience:
        ref_angle = 90 + self.angle
        ref_angle = (math.pi/180) * ref_angle
        #Updating the center:
        new_center = (int(old_center[0] + dist*math.cos(ref_angle)),int(old_center[1] - dist*math.sin(ref_angle)))  
        
        self.prev_center = self.rect.center
        self.rect.center = new_center
        
    def restore_move(self):
        self.rect.center = self.prev_center
        # self.rect = self.previous_rect
    
    def take_random_action(self):
        return [np.random.randint(-15,15), np.random.randint(-2.5,2.5)]
        
    def take_action(self,current_state):
        """Get the action to be taken from the agent

        Args:
            prev_reward (float): The recent reward received by the agent
            current_state (list): An iterable containing the parameters of the environment

        Returns:
            Action (int): The discreet action to be taken by the agent. Return values :[0,1,2...n actions]
        """
        #Preprocessing of the parameters:
        current_state = np.array(current_state)
        
        return self.brain.select_action(current_state)
    
    def train(self,iterations,batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        self.brain.train(iterations,batch_size,discount,tau,policy_noise,noise_clip,policy_freq)
    
    def save_brain(self,path):
        self.brain.save(path)
    
    def add_to_memory(self,state,next_state,action,reward,done):
        self.brain.add_record((state,next_state,action,reward,done))
    # def plot_reward_metric(self):
    #     self.brain.plot_rewards()
    # def __check_and_load_brain(self):

    def __hash__(self):
        return hash(str(self))
        
        