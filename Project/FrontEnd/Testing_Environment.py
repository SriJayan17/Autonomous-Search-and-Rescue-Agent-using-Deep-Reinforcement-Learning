import sys
sys.path.append("e:/AI_projects/RescueAI/")

import pygame
import os

from Project.FrontEnd.Utils.Testing_Env_Obstacles import *
from Project.FrontEnd.Utils.Grapher import *
from Project.Backend.Agent import Agent
from Project.FrontEnd.Utils.Action_Handler import *
from Project.FrontEnd.Utils.Rewards import *

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

        self.search_time_list = []
        self.rescue_time_list = []

        self.base_velocity = 3.0
        self.impermissible_action_count = [0] * self.numberOfAgents
        self.cooldown = [0] * self.numberOfAgents
        self.impermissible_action_threshold = 15

        self.state_extra_info = {
            'scale_x' : self.width,
            'scale_y' : self.height,
            'max_distance' : math.sqrt((self.width)**2 + (self.height)**2),
            'intensity_area_dim' : 10,
        }

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

        self.action_permit = [True] * self.numberOfAgents

        
        self.victims = pygame.image.load("Project/Resources/Images/victims.png")
        self.victims = pygame.transform.scale(self.victims,(test_victimsRect.width, test_victimsRect.height))

        self.fire1 = pygame.image.load("Project/Resources/Images/fire1.png")
        self.fire2 = pygame.image.load("Project/Resources/Images/fire2.png")
        self.fire3 = pygame.image.load("Project/Resources/Images/fire3.png")
        self.fire1= pygame.transform.scale(self.fire1,(fireFlares[0].width,fireFlares[0].height))
        self.fire2 = pygame.transform.scale(self.fire2,(fireFlares[0].width,fireFlares[0].height))
        self.fire3 = pygame.transform.scale(self.fire3,(fireFlares[0].width,fireFlares[0].height))        

        self.exit = pygame.image.load("Project/Resources/Images/exit.jpg")
        self.exit = pygame.transform.scale(self.exit,(50, 30))

        self.agentIcon = pygame.image.load("Project/Resources/Images/agent.png")
        self.agentIcon = pygame.transform.scale(self.agentIcon,(12, 30))

        self.prepare_dirs()

    #Preparing the directories required to store the outputs:
    def prepare_dirs(self):
        cwd = os.getcwd()
        if not os.path.isdir(os.path.join(cwd,"Graphs/Test/search")):
            os.makedirs('Graphs/Test/search')
        if not os.path.isdir(os.path.join(cwd,"Graphs/Test/rescue")):
            os.makedirs('Graphs/Test/rescue')
        
    def stop(self):
        self.stopSimulation = True
    
    def perform_action(self,index:int,turn_angle:float,dist:float,rescue_op:bool=False):
        agent = self.agentModels[index]
        if turn_angle != 0: agent.turn(turn_angle)
        agent.move(self.base_velocity + dist)
        if not isPermissible(self.agentModels, index, testing=True):
            agent.restore_move()
            # if rescue_op:
            self.impermissible_action_count[index] += 1
            if self.impermissible_action_count[index] == self.impermissible_action_threshold:
                self.cooldown[index] = self.impermissible_action_count[index]
                self.impermissible_action_count[index] = 0 
            self.action_permit[index] = False
        # elif rescue_op:
        else:
            self.impermissible_action_count[index] = 0

    def run(self):

        pygame.init()
        pygame.display.set_caption("Search and Rescue Simulation")
        environment = pygame.display.set_mode((self.width, self.height))
        
        reached_agent = -1
        rescued_victims = False
        
        search_time = rescue_time = 0
        total_timesteps = 0
        
        episode_num = 1
        
        target_exit = exit_points[0]
        # Get the nearest exit:
        min_dist = 1e7
        for exit_pt in test_exit_points:
            dist = eucledianDist(test_victimsRect.center,exit_pt.center)
            if dist < min_dist:
                min_dist = dist
                target_exit = exit_pt
        
        while not self.stopSimulation:

            environment.blit(background, (0,0))

            for boundary in boundaries:
                pygame.draw.rect(environment, (0, 0, 51), boundary)

            for furniture in objects:
                environment.blit(furniture[0], furniture[1])

            for obstacle in walls:
                pygame.draw.rect(environment,(0, 0, 51), obstacle)
            
            for fire in test_fireFlares:
                # environment.blit(self.fire3, (fire.x, fire.y))
                if total_timesteps % 24 in range(8):
                    # pygame.draw.rect(environment, (224,224,224,0), pygame.Rect(fire.x,fire.y,45,45))
                    environment.blit(self.fire1, (fire.x, fire.y))
                elif total_timesteps % 24 in range(8,17):  
                    # pygame.draw.rect(environment, (224,224,224,0), pygame.Rect(fire.x,fire.y,45,45))                  
                    environment.blit(self.fire2, (fire.x, fire.y))
                else:    
                    # pygame.draw.rect(environment, (224,224,224,0), pygame.Rect(fire.x,fire.y,45,45))                
                    environment.blit(self.fire3, (fire.x, fire.y))

            
            # An episode over:
            if reached_agent != -1 and rescued_victims:        
                if total_timesteps != 0:
                        self.search_time_list.append(search_time)
                        self.rescue_time_list.append(rescue_time)
                        print(f'Timesteps taken for search operation: {search_time}')
                        print(f'Timesteps taken for rescue operation: {rescue_time}')
                        print(f'Episode Number: {episode_num}')

                # Reset the state of the environment as well as the travel history 
                for i in range(self.numberOfAgents):
                    self.agentModels[i].rect.center = agents[i]
                    environment.blit(self.agentModels[i].shape_copy,self.agentModels[i].rect)
                
                #Reset the switches:
                reached_agent = -1
                rescued_victims = False
                # Reset the timers:
                search_time = rescue_time = 0
                episode_num += 1

                
            if reached_agent == -1:
                environment.blit(self.victims, (test_victimsRect.x,test_victimsRect.y))
            else:
                for exit in exit_points:
                    environment.blit(self.exit, exit.center)

            # Automated Navigation
            # Search Operation:
            if reached_agent == -1:
                search_time += 1
                for i in range(self.numberOfAgents):
                    state = get_state(self.agentModels[i],self.state_extra_info,testing=True,destination=test_victimsRect)
                    action = self.agentModels[i].take_action(state)

                    if self.cooldown[i] > 0:
                        self.perform_action(i, -45 if action[0] > 0 else 45, -self.base_velocity)
                        self.cooldown[i] -= 1
                    else:
                        self.perform_action(i, action[0], 12)
                    
                    if not self.action_permit[i]: # Action not permitted
                        self.action_permit[i] = True
                    else: #Action was permitted
                        if reachedDestination(self.agentModels[i].rect,destination=test_victimsRect):
                            reached_agent = i
                    environment.blit(self.agentModels[i].shape_copy,self.agentModels[i].rect)
            # Rescue Operation:
            else:
                rescue_time += 1
                state = get_state(self.agentModels[reached_agent],self.state_extra_info,testing=True,destination=target_exit)
                action = self.agentModels[reached_agent].take_action(state)
                if self.cooldown[reached_agent] > 0:
                    self.perform_action(reached_agent, -45 if action[0] > 0 else 45, -self.base_velocity, rescue_op=True)
                    self.cooldown[reached_agent] -= 1
                else:
                    self.perform_action(reached_agent, action[0], 12, rescue_op=True)
                if not self.action_permit[reached_agent]:
                    self.action_permit[reached_agent] = True
                else:
                    # TODO: There's something wrong with this, check it!
                    rescued_victims = reachedDestination(self.agentModels[reached_agent].rect,destination=target_exit)
                    if rescued_victims: 
                        print('Reached the exit!')
                        print(f'Agent: {reached_agent}')
                environment.blit(self.agentModels[reached_agent].shape_copy,self.agentModels[reached_agent].rect)
            
            for event in pygame.event.get():  

                if event.type == pygame.QUIT:  
                    if episode_num >= 3:
                        #Plots the rewards obtained by the agents wrt episode
                        # plot_rewards(self.agents_episode_rewards,'Graphs/search')
                        # Plots the time taken to reach the victims wrt episodes
                        plot_reach_time(self.search_time_list,'Time taken to reach the victims','Graphs/Test/search')
                        plot_reach_time(self.rescue_time_list,'Time taken to rescue the victims','Graphs/Test/rescue')
                    self.stop()
                
                # Manual Control:
                if event.type == pygame.KEYDOWN:
                    
                    if event.key == pygame.K_UP:
                        self.perform_action(reached_agent, 0, 12)
                        # get_state(self.agentModels[reached_agent],self.state_extra_info)

                    if event.key == pygame.K_LEFT:
                        self.perform_action(reached_agent, -45, 0)
                        # get_state(self.agentModels[0],self.state_extra_info)

                    if event.key == pygame.K_RIGHT:
                        self.perform_action(reached_agent, 45, 0)
                        # get_state(self.agentModels[0],self.state_extra_info)
            
            # # # Manual Control
            # environment.blit(self.agentModels[0].shape_copy,self.agentModels[0].rect)
            # if(reachedVictims(self.agentModels[0])):
            #     self.stop()
            
            # if total_timesteps % 1000 == 0: print(f'Timsesteps: {total_timesteps}')
            total_timesteps += 1
            pygame.display.flip()
            pygame.time.delay(10)


obj = TestingEnvironment()
obj.run()

