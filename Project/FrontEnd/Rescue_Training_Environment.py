import sys
sys.path.append("e:/AI_projects/RescueAI/")

import random
import pygame
import os
import pickle

from Project.Backend.Brains.TD3.Memory import ReplayBuffer
from Project.FrontEnd.Utils.Grapher import *
from Project.FrontEnd.Utils.Training_Env_Obstacles import *
from Project.Backend.Agent import Agent
from Project.FrontEnd.Utils.Action_Handler import *

class TrainingEnvironment:

    reachedVictims = False
    stopSimulation = False

    def __init__(self):

        # Environment Dimension
        self.height = 750
        self.width = 1500

        self.state_len = 8
        self.agentModels = [
            Agent(self.state_len,
                1,
                15,
                (victimsRect.x, victimsRect.y),
                memory = 1e6,
                load = False,
                load_path = 'saved_models/rescue_agent'
                )
        ]
        for agent_center in helper_agents:
            self.agentModels.append(
                Agent(
                self.state_len,
                1,
                15,
                agent_center,
                memory = 1e6,
                load = False,
                load_path = 'saved_models/rescue_agent'
                )
            )

        # Determinig the nearest exit point for the agents based in the ditances
        # between the victims and the exit points:
        min_dist = 1e7
        self.target_rect = None
        for pt in exit_points:
            dist = eucledianDist(victimsRect.center,pt.center)
            if dist < min_dist:
                min_dist = dist
                self.target_rect = pt
        print(f'Target_pt : {self.target_rect.center}')

        self.base_velocity = 3.0

        self.state_extra_info = {
            'scale_x' : self.width,
            'scale_y' : self.height,
            'max_distance' : math.sqrt((self.width)**2 + (self.height)**2),
            'intensity_area_dim' : 10,
        }
        self.memory = ReplayBuffer()
        self.agent_episode_reward = [[]]
        
        
        #This is to store the current state of an individual agent   
        self.state_dict = [None] * len(self.agentModels)
        # This is to store the actual state(combined) that we pass into the neural network
        # when taking decision, this is needed to create the records to train the agent. 
        for i in range(len(self.agentModels)):
            self.state_dict[i] = get_state(self.agentModels[i],self.state_extra_info, destination=self.target_rect)

        # This is to store the actions taken by the agents:
        self.action_dict = None
        self.action_permit = [True] * len(self.agentModels)

        self.agentRewards = 0
        self.episode_rewards = 0

        self.victims = pygame.image.load("Project/Resources/Images/rescue_victims.png")
        self.victims = pygame.transform.scale(self.victims,(20,20))

        self.fire = pygame.image.load("Project/Resources/Images/fire.png")
        self.fire = pygame.transform.scale(self.fire,(fireFlares[0].width,fireFlares[0].height))

        self.exit = pygame.image.load("Project/Resources/Images/exit.jpg")
        self.exit = pygame.transform.scale(self.exit,(30, 30))

        self.check_dirs()

    #Preparing the directories required to store the outputs:
    def check_dirs(self):
        cwd = os.getcwd()
        if not os.path.isdir(os.path.join(cwd,"saved_models/rescue_agent")):
            os.makedirs('saved_models/rescue_agent')
        if not os.path.isdir(os.path.join(cwd,"Graphs/rescue")):
            os.makedirs('Graphs/rescue')


    def stop(self):
        self.stopSimulation = True
    
    def perform_action(self, i, turn_angle,dist):
        if turn_angle != 0: self.agentModels[i].turn(turn_angle)
        self.agentModels[i].move(self.base_velocity + dist)
        if not isPermissible([self.agentModels[i]]):
            self.agentModels[i].restore_move()
            self.action_permit[i] = False


    def run(self):

        pygame.init()
        pygame.display.set_caption("Search and Rescue Simulation")
        environment = pygame.display.set_mode((self.width, self.height))

        episode_timesteps = 0
        total_timesteps = 0
        random_action_limit = 1000
        episode_len = 10_000
        expl_noise = [3,5]
        
        episode_num = 1
        done = False 
        reached = [False] * len(self.agentModels)
        

        while not self.stopSimulation:

            environment.fill((0,0,0))

            for boundary in boundaries:
                pygame.draw.rect(environment, (255, 0, 0), boundary)

            for obstacle in obstacles:
                pygame.draw.rect(environment,(255, 0, 0), obstacle)

            for exit in exit_points:
                environment.blit(self.exit, exit.center)
            
            
            for fire in fireFlares:
                environment.blit(self.fire, (fire.x, fire.y))
            
            
            # An episode is over
            if done:
                # If we are not at the very beginning, we start the training process of the model
                if total_timesteps != 0:
                    self.agent_episode_reward[0].append(self.episode_rewards)
                    print(f'Total Timesteps: {total_timesteps} Episode Num: {episode_num}')
                    print('Total reward obtained by the agent:')                    
                    print(f'Agent: {self.episode_rewards}')
                    self.episode_rewards = 0

                    displayPrompt("Training Agent")
                    print(f'Training Agent')
                    for i in range(len(self.agentModels)):
                        print(f'Training Agent : {i}')
                        self.agentModels[i].train(memory=self.memory,iterations=episode_timesteps,batch_size=300)
                        reached[i] = False
                        if i == 0:
                            self.agentModels[i].rect.center = victimsRect.center
                            self.agentModels[i].save_brain(f'./saved_models/rescue_agent')
                        else:
                            self.agentModels[i].rect.center = helper_agents[i-1]
        
                # When the training step is done, we reset the state of the environment    
                environment.blit(self.agentModels[0].shape_copy,self.agentModels[0].rect)    
                # obs = env.reset()
        
                # Set the Done to False
                done = False
                
                # Set rewards and episode timesteps to zero
                episode_timesteps = 0
                episode_num += 1    

            # Last run of the episode.
            if episode_timesteps + 1 == episode_len:
                done = True
            # # To check if any of the agents reached the target:
            # if_reached = False
            # # Automated Navigation
            for i in range(len(self.agentModels)):
                # Take random action in the initial 10,000 timesteps
                if total_timesteps < random_action_limit:
                    action = self.agentModels[i].take_random_action()
                else:
                # Take action using neural network
                    action = self.agentModels[i].take_action(self.state_dict[i])
                    # action = self.getManualAction()
                    if expl_noise != 0: # Adding noise to the predicted action
                        action = (action + random.uniform(expl_noise[0], expl_noise[1])).clip(-15,15) 
                        # Clipping the final action between the permissible range of values
                # print(action)
                # If the agent has not reached yet, (this is only for hidden agents!)
                if not reached[i]:
                    self.perform_action(i, action[0], 12)
                    # self.action_dict = action

                    if not self.action_permit[i]: # Action not permitted
                        reward = IMPERMISSIBLE_ACTION
                        self.action_permit[i] = True
                    else: #Action was permitted
                        reward = generateReward(self.agentModels[i].prev_center, self.agentModels[i].rect,
                                                rescue_op=True, nearest_exit = self.target_rect)                

                    prev_state = self.state_dict[i]
                    # Update the current state of the individual agent
                    self.state_dict[i] = get_state(self.agentModels[i],self.state_extra_info, destination=self.target_rect)       

                    reached[i] = reachedExit(self.agentModels[i].rect)

                    # Add the record in the memory of the agent's brain:
                    self.memory.add((prev_state,self.state_dict[i],action,reward,reached[i]))

                    if i == 0:
                        #Adding to the total episode_reward received by a single agent:
                        self.episode_rewards += reward
                        if reached[i]: done = True
                        environment.blit(self.agentModels[0].shape_copy,self.agentModels[0].rect)   
            # print(f'proper_angle: {self.agentModel.get_proper_angle()}')
            # environment.blit(self.victims, (self.agentModel.rect.x+10, self.agentModel.rect.y+10))
            
            # # An episode is done if the timelimeet has been reached or if any of the agents
            # # has reached the target    
            # done = done or if_reached

            # episode_timesteps += 1
            # total_timesteps += 1
            
            
            # Manual Control:
            for event in pygame.event.get():  

                if event.type == pygame.QUIT:  
                    if episode_num >= 3:
                        plot_rewards(self.agent_episode_reward,'Graphs/rescue')
                        # plot_reach_time()
                        
                    self.stop()
                
                # if event.type == pygame.KEYDOWN:
                    
                #     if event.key == pygame.K_UP:
                #         self.perform_action(self.agentModels, 0, 0, 10)

                #     if event.key == pygame.K_LEFT:
                #         self.perform_action(self.agentModels, 0, -15, 10)

                #     if event.key == pygame.K_RIGHT:
                #         self.perform_action(self.agentModels, 0, 15, 10)
            
            # # Manual Control
            # environment.blit(self.agentModels[0].shape_copy,self.agentModels[0].rect)
            # if(reachedVictims(self.agentModels[0])):
            #     self.stop()
            episode_timesteps += 1
            total_timesteps += 1
            if total_timesteps % 1000 == 0: print(f'Timesteps: {total_timesteps}')
            pygame.display.flip()
            # pygame.time.delay(20)


obj = TrainingEnvironment()
obj.run()

