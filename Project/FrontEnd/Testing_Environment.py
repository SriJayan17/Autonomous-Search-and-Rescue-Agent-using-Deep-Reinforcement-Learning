import sys
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

class TrainingEnvironment:

    reachedVictims = False
    stopSimulation = False

    def __init__(self):

        # Environment Dimension
        self.height = 750
        self.width = 1500

        self.numberOfAgents = 4
        self.state_len = 8
        self.agentModels = []

        self.agents_episode_rewards = [[] for _ in range(self.numberOfAgents)]
        self.reach_time = []

        self.base_velocity = 3.0

        self.state_extra_info = {
            'scale_x' : self.width,
            'scale_y' : self.height,
            'max_distance' : math.sqrt((self.width)**2 + (self.height)**2),
            'intensity_area_dim' : 10,
        }

        self.memory = ReplayBuffer()
        # Initialising the agents
        # Action limit:
        #    Angle -> Varies from -15 to 15
        # for i in range(self.numberOfAgents):
        #     self.agentModels.append(Agent(self.state_len,
        #                                   1,
        #                                   15,
        #                                   agents[i],
        #                                   memory = 1e6,
        #                                   load = False,
        #                                   load_path = f'saved_models/agent_{i}'
        #                                   )
        #                             )

        #This is to store the current state of an individual agent   
        self.state_dict = [None] * self.numberOfAgents
        
        #This is to store the travel record of the agents, to trace back during rescue operation:
        self.travel_history = [[]] * self.numberOfAgents

        # for i in range(self.numberOfAgents):
        #     self.state_dict[i] = get_state(self.agentModels[i],self.state_extra_info)
        # self.initial_state_dict = deepcopy(self.state_dict)
        
        # for i in range(self.numberOfAgents):
        #     self.actual_state_dict[i] = prepare_agent_state(self.agentModels,i,self.state_dict,self.state_dict)  

        # This is to store the actions taken by the agents:
        self.action_dict = [None] * self.numberOfAgents
        # This is to check whether the current action was permitted or not:
        self.action_permit = [True] * self.numberOfAgents

        self.agentRewards = [0] * self.numberOfAgents
        self.episode_rewards = [0] * self.numberOfAgents

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
        if not os.path.isdir(os.path.join(cwd,"Graphs/search")):
            os.makedirs('Graphs/search')
        if not os.path.isdir(os.path.join(cwd,"saved_models")):
            for i in range(self.numberOfAgents):
                if not os.path.isdir(os.path.join(cwd,f'saved_models/agent_{i}')):
                    os.makedirs(f'saved_models/agent_{i}')

    def stop(self):
        self.stopSimulation = True
    
    def perform_action(self,agent_list,index,turn_angle,dist):
        agent = agent_list[index]
        if turn_angle != 0: agent.turn(turn_angle)
        agent.move(self.base_velocity + dist)
        if not isPermissible(agent_list, index):
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
        random_action_limit = 500
        episode_len = 5_000
        expl_noise = [3,5]
        
        episode_num = 1
        done = False 
        
        while not self.stopSimulation:

            # environment.fill((0,0,0))

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
            # if done:        
            #     # If we are not at the very beginning, we start the training process of the model
            #     if total_timesteps != 0:
            #             print(f'Total Timesteps: {total_timesteps} Episode Num: {episode_num}')
            #             print('Total reward obtained by the agents:')
            #             for i in range(self.numberOfAgents):
            #                 self.agents_episode_rewards[i].append(self.episode_rewards[i])
            #                 print(f'Agent_{i}: {self.episode_rewards[i]}')
            #                 self.episode_rewards[i] = 0
            #             popup = None
            #             for i in range(self.numberOfAgents):
            #                 displayPrompt("Training Agent : "+str(i+1))
            #                 print(f'Training Agent_{i}')
            #                 self.agentModels[i].train(memory=self.memory,iterations=episode_timesteps,batch_size=500)
            #                 self.agentModels[i].save_brain(f'./saved_models/agent_{i}')
            #             # agent.train(replay_buffer, episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip, policy_freq)
            #     # When the training step is done, we reset the state of the environment as well as the travel history 
            #     for i in range(self.numberOfAgents):
            #         self.agentModels[i].rect.center = agents[i]
            #         environment.blit(self.agentModels[i].shape_copy,self.agentModels[i].rect)

            #         self.travel_history[i].clear()    
            #     # obs = env.reset()
        
            #     # Set the Done to False
            #     done = False
                
            #     self.reach_time.append(episode_timesteps)
            #     # Set rewards and episode timesteps to zero
            #     episode_timesteps = 0
            #     episode_num += 1    

            # # Last run of the episode.
            # if episode_timesteps + 1 == episode_len:
            #     done = True
            # # To check if any of the agents reached the target:
            # if_reached = [False] * self.numberOfAgents
            # # Automated Navigation
            # for i in range(self.numberOfAgents):
            #     # Take random action in the initial 10,000 timesteps
            #     if total_timesteps < random_action_limit:
            #         action = self.agentModels[i].take_random_action()
            #         # action = self.getManualAction()
            #     else:
            #         action = self.agentModels[i].take_action(self.state_dict[i])

            #         if expl_noise is not None: # Adding noise to the predicted action
            #             action = (action + random.uniform(expl_noise[0], expl_noise[1])).clip(-15,15) # Clipping the final action between the permissible range of values
            #     # print(action)
            #     self.perform_action(self.agentModels, i, action[0], 12)
            #     self.action_dict[i] = action
            #     # print(f'Action of agent_{i}: {action}')
                
            #     if not self.action_permit[i]: # Action not permitted
            #         self.agentRewards[i] = IMPERMISSIBLE_ACTION
            #         self.action_permit[i] = True
            #     else: #Action was permitted
            #         # Keeping track of the permitted actions alone:
            #         self.travel_history[i].append(action)
            #         self.agentRewards[i] = generateReward(self.agentModels[i].prev_center, self.agentModels[i].rect)                
            #     #Adding to the total episode_reward received by a single agent:
            #     self.episode_rewards[i] += self.agentRewards[i]
                
            #     #Storing the record in memory:
            #     prev_state = self.state_dict[i]
            #       # Update the current state of the individual agent
            #       # Update the state in both the cases (Move permitted/not), because the orientation of the rectange might have changed:
            #     self.state_dict[i] = get_state(self.agentModels[i],self.state_extra_info)
            #       # Checking if the agent has reached
            #     reached = reachedVictims(self.agentModels[i].rect)
            #         # If reached, check if this is the minimum time taken to reach.
            #         # If yes, store the path_trace of that agent to use for rescue
            #       # Add the record in the common memory:
            #     self.memory.add((prev_state,self.state_dict[i],self.action_dict[i],self.agentRewards[i],reached))
                
            #     environment.blit(self.agentModels[i].shape_copy,self.agentModels[i].rect)

            #     if_reached[i] = reached

            # # An episode is done if the timelimit has been reached or if any of the agents
            # # has reached the target    
            # done = done or any(if_reached)
                   
            # episode_timesteps += 1
            total_timesteps += 1
            
            for event in pygame.event.get():  

                if event.type == pygame.QUIT:  
                    if episode_num >= 3:
                        #Plots the rewards obtained by the agents wrt episode
                        plot_rewards(self.agents_episode_rewards,'Graphs/search')
                        # Plots the time taken to reach the victims wrt episodes
                        plot_reach_time(self.reach_time,'Time taken to reach the victims','Graphs/search')
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
            pygame.display.flip()
            # pygame.time.delay(200)


obj = TrainingEnvironment()
obj.run()
