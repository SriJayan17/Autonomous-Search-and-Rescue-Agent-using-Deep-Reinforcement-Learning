import pygame
from Project.FrontEnd.Utils.Training_Env_Obstacles import *
from Project.Backend.Agent import Agent
from Project.FrontEnd.Utils.Action_Handler import *

class TrainningEnvironment:

    reachedVictims = False
    stopSimulation = False

    def __init__(self):

        # Environment Dimension
        self.height = 750
        self.width = 1500

        self.numberOfAgents = 4
        self.agentModels = []

        for agent in agents:
            self.agentModels.append(Agent(12,2,2, (agent.x, agent.y)))

        self.initialState = {}
        for agent in self.agentModels:
            self.initialState[agent] = updateOwnState(agent, self.agentModels)

        self.agentsState = self.initialState
        self.agentRewards = [0]* self.numberOfAgents

        self.victims = pygame.image.load("Project/Resources/Images/victims.png")
        self.victims = pygame.transform.scale(self.victims,(victimsRect.width, victimsRect.height))

        self.fire = pygame.image.load("Project/Resources/Images/fire.png")
        self.fire = pygame.transform.scale(self.fire,(fireFlares[0].width,fireFlares[0].height))

        self.agentIcon = pygame.image.load("Project/Resources/Images/agent.png")
        self.agentIcon = pygame.transform.scale(self.agentIcon,(30,30))

    
    def stop(self):
        
        self.stopSimulation = True
    
    def run(self):

        pygame.init()
        pygame.display.set_caption("Search and Rescue Simulation")
        environment = pygame.display.set_mode((self.width, self.height)) 

        while not self.stopSimulation:

            environment.fill((0,0,0))

            for boundary in boundaries:
                pygame.draw.rect(environment, (255, 0, 0), boundary)

            for obstacle in obstacles:
                pygame.draw.rect(environment,(255, 0, 0), obstacle)
            
            environment.blit(self.victims, (victimsRect.x, victimsRect.y))

            for fire in fireFlares:
                environment.blit(self.fire, (fire.x, fire.y))

            # To get Manual Control
            # if(not isPermissible(self.agentModels[0].rect)):
            #     print("Collision")
            #     self.agentModels[0].restore()

            # environment.blit(self.agentModels[0].shape_copy,self.agentModels[0].rect)
            # if(reachedVictims(self.agentModels[0])):
            #     self.stop()

            # Automated Navigation
            for i in range(self.numberOfAgents):
                action = self.agentModels[i].take_action(self.agentRewards[i],self.agentsState[self.agentModels[i]],False)
                self.agentModels[i].turn(action[0])
                self.agentModels[i].move(action[1])
                
                neighbours = []
                for agent in self.agentModels:
                    if(agent != self.agentModels[i]):
                        neighbours.append(agent.rect)

                if(not isPermissible(self.agentModels[i].rect, neighbours)):
                    print("Collision")
                    self.agentModels[i].restore()
                    self.agentRewards[i] = -2
                else:
                    self.agentRewards[i] = generateReward(self.agentModels[i].prev_rect, self.agentModels[i].rect)
                    self.agentsState = updateState(self.agentModels, self.initialState)

                # print(self.agentsState)
                # print(self.agentRewards)
                environment.blit(self.agentModels[i].shape_copy,self.agentModels[i].rect)
                if(reachedVictims(self.agentModels[i])):
                    self.stop()

            for event in pygame.event.get():  

                if event.type == pygame.QUIT:  

                    self.stop()
                
                if event.type == pygame.KEYDOWN:
                    
                    if event.key == pygame.K_UP:
                        self.agentModels[0].move()

                    if event.key == pygame.K_LEFT:
                        self.agentModels[0].turn(-15)

                    if event.key == pygame.K_RIGHT:
                        self.agentModels[0].turn(15)
            
            pygame.display.flip()
            pygame.time.delay(20)


obj = TrainningEnvironment()
obj.run()

