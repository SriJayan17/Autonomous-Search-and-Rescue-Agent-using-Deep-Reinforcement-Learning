import sys
sys.path.append("e:/AI_projects/RescueAI/")

import pygame
from copy import deepcopy

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

        self.numberOfAgents = 4
        self.state_len = 10
        self.agentModels = []

        self.base_velocity = 7.5

        self.state_extra_info = {
            'scale_x' : self.width,
            'scale_y' : self.height,
            'max_distance' : math.sqrt((self.width)**2 + (self.height)**2),
            'intensity_area_dim' : 30,
        }

        # Initialising the agents
        # Action limit:
        #    Angle -> Varies from -15 to 15
        #    Velocity -> Varies from (7.5 - 2.5 = 5) to (7.5 + 2.5 = 10)
        for agent in agents:
            self.agentModels.append(Agent(self.numberOfAgents * self.state_len,
                                          2,
                                          [15,2.5],
                                          (agent[0], agent[1])
                                          )
                                    )

           
        self.state_dict = [None] * self.numberOfAgents
        for i in range(self.numberOfAgents):
            self.state_dict[i] = get_state(self.agentModels[i],self.state_extra_info)
        self.initial_state_dict = deepcopy(self.state_dict) 

        self.agentRewards = [0] * self.numberOfAgents

        self.victims = pygame.image.load("Project/Resources/Images/victims.png")
        self.victims = pygame.transform.scale(self.victims,(victimsRect.width, victimsRect.height))

        self.fire = pygame.image.load("Project/Resources/Images/fire.png")
        self.fire = pygame.transform.scale(self.fire,(fireFlares[0].width,fireFlares[0].height))

        self.agentIcon = pygame.image.load("Project/Resources/Images/agent.png")
        self.agentIcon = pygame.transform.scale(self.agentIcon,(30,30))

        # self.flock_center = calc_flock_center(self.agentModels)
        #Training specific parameters:
        

    def stop(self):

        self.stopSimulation = True
    
    def perform_action(self,agent_list,index,turn_angle,dist):
        agent = agent_list[index]
        if turn_angle != 0: agent.turn(turn_angle)
        agent.move(self.base_velocity + dist)
        if not isPermissible(agent_list, index):
            agent.restore_move()
            self.agentRewards[index] = -2
    
    def run(self):

        pygame.init()
        pygame.display.set_caption("Search and Rescue Simulation")
        environment = pygame.display.set_mode((self.width, self.height)) 
        
        # print(f'Initial top_left of agent: {self.agentModels[0].rect.topleft}')
        # print(f'Initial bottom right of agent: {self.agentModels[0].rect.bottomright}')
        # print(f'Center of the agent: {self.agentModels[0].rect.center}')

        while not self.stopSimulation:

            environment.fill((0,0,0))

            for boundary in boundaries:
                pygame.draw.rect(environment, (0, 255, 0), boundary)

            for obstacle in obstacles:
                pygame.draw.rect(environment,(255, 0, 0), obstacle)
            
            environment.blit(self.victims, (victimsRect.x, victimsRect.y))

            for fire in fireFlares:
                environment.blit(self.fire, (fire.x, fire.y))

            # Automated Navigation
            for i in range(self.numberOfAgents):
                action = self.agentModels[i]\
                             .take_action(self.agentRewards[i],
                                          prepare_agent_state(self.agentModels, 
                                                              i, 
                                                              self.state_dict, 
                                                              self.initial_state_dict,
                                                              None
                                                            #   self.flock_center,
                                                              ),
                                          False
                                          )
                # print(action)
                self.perform_action(self.agentModels, i, action[0], action[1])
                
                
                if self.agentRewards[i] != -2: #Action was permitted
                    self.agentRewards[i] = generateReward(self.agentModels[i].prev_center, self.agentModels[i].rect)
                
                # Update the state in both the cases, because, the orientation of the rectange might have changed:
                self.state_dict[i] = get_state(self.agentModels[i],self.state_extra_info)

                environment.blit(self.agentModels[i].shape_copy,self.agentModels[i].rect)
                if(reachedVictims(self.agentModels[i])):
                    self.stop()
            
            # Manual Control:
            # for event in pygame.event.get():  

            #     if event.type == pygame.QUIT:  
            #         self.stop()
                
            #     if event.type == pygame.KEYDOWN:
                    
            #         if event.key == pygame.K_UP:
            #             self.perform_action(self.agentModels, 0, 0, 10)

            #         if event.key == pygame.K_LEFT:
            #             self.perform_action(self.agentModels, 0, -15, 10)

            #         if event.key == pygame.K_RIGHT:
            #             self.perform_action(self.agentModels, 0, 15, 10)
            
            # # Manual Control
            # environment.blit(self.agentModels[0].shape_copy,self.agentModels[0].rect)
            # if(reachedVictims(self.agentModels[0])):
            #     self.stop()
            
            pygame.display.flip()
            pygame.time.delay(20)


obj = TrainingEnvironment()
obj.run()

