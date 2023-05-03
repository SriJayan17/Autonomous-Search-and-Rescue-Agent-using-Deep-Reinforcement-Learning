import sys
sys.path.append("e:/AI_projects/RescueAI/")

import pygame
from copy import deepcopy
import os
import random
import pickle

from Project.FrontEnd.Utils.Testing_Env_Obstacles import *
from Project.FrontEnd.Utils.Grapher import *
from Project.Backend.Agent import Agent
from Project.FrontEnd.Utils.Action_Handler import *
from Project.FrontEnd.Utils.Rewards import *
from Project.Backend.Brains.TD3.Memory import ReplayBuffer

class TestingEnvironment:

    reachedVictims = False
    stopSimulation = False

    def __init__(self):

        # Environment Dimension
        self.height = 750
        self.width = 1500

        self.numberOfAgents = 4
        self.state_len = 8
        self.agentModels = []

        # self.agents_episode_rewards = [[] for _ in range(self.numberOfAgents)]
        self.reach_time = []

        self.base_velocity = 3.0

        self.state_extra_info = {
            'scale_x' : self.width,
            'scale_y' : self.height,
            'max_distance' : math.sqrt((self.width)**2 + (self.height)**2),
            'intensity_area_dim' : 10,
        }

        # self.memory = ReplayBuffer()
        # Initialising the agents
        # Action limit:
        #    Angle -> Varies from -15 to 15
        for i in range(self.numberOfAgents):
            self.agentModels.append(Agent(self.state_len,
                                          1,
                                          15,
                                          agents[i],
                                          memory = 1e6,
                                          load = True,
                                          load_path = f'saved_models/agent_{i}'
                                          )
                                    )

        #This is to store the current state of an individual agent   
        # self.state_dict = [None] * self.numberOfAgents
        
        #This is to store the travel record of the agents, to trace back during rescue operation:
        # self.travel_history = [[]] * self.numberOfAgents

        # for i in range(self.numberOfAgents):
        #     self.state_dict[i] = get_state(self.agentModels[i],self.state_extra_info)
        # self.initial_state_dict = deepcopy(self.state_dict)
        
        # for i in range(self.numberOfAgents):
        #     self.actual_state_dict[i] = prepare_agent_state(self.agentModels,i,self.state_dict,self.state_dict)  

        # This is to store the actions taken by the agents:
        # self.action_dict = [None] * self.numberOfAgents
        # This is to check whether the current action was permitted or not:
        self.action_permit = [True] * self.numberOfAgents

        # self.agentRewards = [0] * self.numberOfAgents
        # self.episode_rewards = [0] * self.numberOfAgents

        self.victims = pygame.image.load("Project/Resources/Images/victims.png")
        self.victims = pygame.transform.scale(self.victims,(victimsRect.width, victimsRect.height))

        self.fire1 = pygame.image.load("Project/Resources/Images/fire1.png")
        self.fire2 = pygame.image.load("Project/Resources/Images/fire2.png")
        self.fire3 = pygame.image.load("Project/Resources/Images/fire3.png")
        self.fire1= pygame.transform.scale(self.fire1,(fireFlares[0].width,fireFlares[0].height))
        self.fire2 = pygame.transform.scale(self.fire2,(fireFlares[0].width,fireFlares[0].height))
        self.fire3 = pygame.transform.scale(self.fire3,(fireFlares[0].width,fireFlares[0].height))        


        self.agentIcon = pygame.image.load("Project/Resources/Images/agent.png")
        self.agentIcon = pygame.transform.scale(self.agentIcon,(12, 30))

        self.prepare_dirs()

        # self.flock_center = calc_flock_center(self.agentModels)

    #Preparing the directories required to store the outputs:
    def prepare_dirs(self):
        cwd = os.getcwd()
        if not os.path.isdir(os.path.join(cwd,"Graphs/Test/search")):
            os.makedirs('Graphs/Test/search')
        # if not os.path.isdir(os.path.join(cwd,"saved_models")):
        #     for i in range(self.numberOfAgents):
        #         if not os.path.isdir(os.path.join(cwd,f'saved_models/agent_{i}')):
        #             os.makedirs(f'saved_models/agent_{i}')

    def stop(self):
        self.stopSimulation = True
    
    def perform_action(self,index:int,turn_angle:float,dist:float):
        agent = self.agentModels[index]
        if turn_angle != 0: agent.turn(turn_angle)
        agent.move(self.base_velocity + dist)
        if not isPermissible(self.agentModels, index, testing=True):
            agent.restore_move()
            # print('Action not permitted!')
            # agent.turn(-turn_angle)
            # self.agentRewards[index] = IMPERMISSIBLE_ACTION
            self.action_permit[index] = False

    def run(self):

        pygame.init()
        pygame.display.set_caption("Search and Rescue Simulation")
        environment = pygame.display.set_mode((self.width, self.height))
        environment.blit(background, (0,0))

        episode_timesteps = 0
        total_timesteps = 0
        # random_action_limit = 500
        episode_len = 5_000
        expl_noise = [3,5]
        
        episode_num = 1
        done = False 
        
        while not self.stopSimulation:

            environment.fill((0,0,0))

            for boundary in boundaries:
                pygame.draw.rect(environment, (0, 0, 51), boundary)

            for furniture in objects:
                environment.blit(furniture[0], furniture[1])

            for obstacle in walls:
                pygame.draw.rect(environment,(0, 0, 51), obstacle)
            
            # for rect in objects_rect:
            #     pygame.draw.rect(environment,(0, 0, 255), rect)
            
            environment.blit(self.victims, (test_victimsRect.x,test_victimsRect.y))

            # print(total_timesteps)
            for fire in test_fireFlares:
                # environment.blit(self.fire3, (fire.x, fire.y))
                if total_timesteps % 24 in range(8):
                    pygame.draw.rect(environment, (224,224,224,0), pygame.Rect(fire.x,fire.y,45,45))
                    environment.blit(self.fire1, (fire.x, fire.y))
                elif total_timesteps % 24 in range(8,17):  
                    pygame.draw.rect(environment, (224,224,224,0), pygame.Rect(fire.x,fire.y,45,45))                  
                    environment.blit(self.fire2, (fire.x, fire.y))
                else:    
                    pygame.draw.rect(environment, (224,224,224,0), pygame.Rect(fire.x,fire.y,45,45))                
                    environment.blit(self.fire3, (fire.x, fire.y))

            
            # An episode is over
            if done:        
                # If we are not at the very beginning, we start the training process of the model
                if total_timesteps != 0:
                        self.reach_time.append(episode_timesteps)
                        print(f'Timesteps taken to reach: {episode_timesteps} Episode Num: {episode_num}')
                
                # Reset the state of the environment as well as the travel history 
                for i in range(self.numberOfAgents):
                    self.agentModels[i].rect.center = agents[i]
                    environment.blit(self.agentModels[i].shape_copy,self.agentModels[i].rect)

                # Set the Done to False
                done = False
                
                # Set rewards and episode timesteps to zero
                episode_timesteps = 0
                episode_num += 1    

            # Automated Navigation
            for i in range(self.numberOfAgents):
                # Take random action in the initial 10,000 timesteps
                state = get_state(self.agentModels[i],self.state_extra_info,testing=True)
                action = self.agentModels[i].take_action(state)

                # if expl_noise is not None: # Adding noise to the predicted action
                #     action = (action + random.uniform(expl_noise[0], expl_noise[1])).clip(-15,15) # Clipping the final action between the permissible range of values
            # print(action)
                self.perform_action(i, action[0], 12)
                # print(f'Action of agent_{i}: {action}')
                
                if not self.action_permit[i]: # Action not permitted
                    self.action_permit[i] = True
                else: #Action was permitted
                    done = done or reachedDestination(self.agentModels[i].rect,destination=test_victimsRect)
                environment.blit(self.agentModels[i].shape_copy,self.agentModels[i].rect)
                   
            
            for event in pygame.event.get():  

                if event.type == pygame.QUIT:  
                    if episode_num >= 3:
                        #Plots the rewards obtained by the agents wrt episode
                        # plot_rewards(self.agents_episode_rewards,'Graphs/search')
                        # Plots the time taken to reach the victims wrt episodes
                        plot_reach_time(self.reach_time,'Time taken to reach the victims','Graphs/Test/search')
                    self.stop()
                
            #     # Manual Control:
            #     if event.type == pygame.KEYDOWN:
                    
            #         if event.key == pygame.K_UP:
            #             self.perform_action(self.agentModels, 0, 0, 12)
            #             get_state(self.agentModels[0],self.state_extra_info)

            #         if event.key == pygame.K_LEFT:
            #             self.perform_action(self.agentModels, 0, -15, 12)
            #             get_state(self.agentModels[0],self.state_extra_info)

            #         if event.key == pygame.K_RIGHT:
            #             self.perform_action(self.agentModels, 0, 15, 12)
            #             get_state(self.agentModels[0],self.state_extra_info)
            
            # # # Manual Control
            # environment.blit(self.agentModels[0].shape_copy,self.agentModels[0].rect)
            # if(reachedVictims(self.agentModels[0])):
            #     self.stop()
            
            # if total_timesteps % 1000 == 0: print(f'Timsesteps: {total_timesteps}')
            episode_timesteps += 1
            total_timesteps += 1
            pygame.display.flip()
            pygame.time.delay(50)


obj = TestingEnvironment()
obj.run()

