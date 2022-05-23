import pygame
from Project.Backend.Agent import Agent
from Project.FrontEnd.Utils.RealTimeObstacles import fires
from Project.FrontEnd.Utils.RewardHandler import RewardHandler
from tkinter import messagebox
from tkinter import *

class RealTimeEnvironment:

    def __init__(self):

        # Initialising objects
        # agent = Agent(9,3,'DQN')
        # rewardHandler = RewardHandler(grid, obstacles, fireFlares, borders, victimsRect)

        # Environment Dimensions
        width = 700
        height = 700

        # Inilialize pygame
        pygame.init()
        pygame.display.set_caption("Autonomous Search and Rescue Simulation")


        dimensions = (width, height)
        environment = pygame.display.set_mode(dimensions)  

        # Loading and scaling up the images
        backGround = pygame.image.load("Project/Resources/Images/floor.png")
        backGround = pygame.transform.scale(backGround,(width,height))

        wall = pygame.image.load("Project/Resources/Images/wall1.png")
        walls = []
        walls.append([pygame.transform.scale(wall,(20,height)),(0,0)])
        walls.append([pygame.transform.scale(wall,(20,height)),(width-20,0)])
        walls.append([pygame.transform.scale(wall,(width-40,20)),(20,0)])
        walls.append([pygame.transform.scale(wall,(60,20)),(20,height-20)])
        walls.append([pygame.transform.scale(wall,(width-160,20)),(160,height-20)])
        walls.append([pygame.transform.scale(wall,(20,300)),(400,20)])
        walls.append([pygame.transform.scale(wall,(20,280)),(400,400)])
        walls.append([pygame.transform.scale(wall,(180,20)),(500,300)])
        walls.append([pygame.transform.scale(wall,(300,20)),(20,400)])
        walls.append([pygame.transform.scale(wall,(90,20)),(420,500)])
        walls.append([pygame.transform.scale(wall,(90,20)),(590,500)])

        table1 = pygame.image.load("Project/Resources/Images/table1.png")
        table1 = pygame.transform.scale(table1,(150,100))
        table1 = pygame.transform.rotate(table1,90)

        sofa1 = pygame.image.load("Project/Resources/Images/sofa1.png")
        sofa1 = pygame.transform.scale(sofa1,(100,100))
        sofa1 = pygame.transform.rotate(sofa1,90)

        sofa2 = pygame.transform.rotate(sofa1,180)

        sofa3 = pygame.image.load("Project/Resources/Images/sofa2.png")
        sofa3 = pygame.transform.scale(sofa3,(180,60))
        sofa3 = pygame.transform.rotate(sofa3,180)

        table2 = pygame.image.load("Project/Resources/Images/table2.png")
        table2 = pygame.transform.scale(table2,(130,130))
        table2 = pygame.transform.rotate(table2,-90)

        roundTable = pygame.image.load("Project/Resources/Images/roundtable.png")
        roundTable = pygame.transform.scale(roundTable,(100,60))

        fire1 = pygame.image.load("Project/Resources/Images/fire1.png")
        fire1 = pygame.transform.scale(fire1,(40,40))

        fire2 = pygame.image.load("Project/Resources/Images/fire2.png")
        fire2 = pygame.transform.scale(fire2,(40,40))

        fire3 = pygame.image.load("Project/Resources/Images/fire3.png")
        fire3 = pygame.transform.scale(fire3,(40,40))

        victims = pygame.image.load("Project/Resources/Images/victims.png")
        victims = pygame.transform.scale(victims,(50,50))

        agent_icon = pygame.image.load("Project/Resources/Images/agent.png")
        agent_icon = pygame.transform.scale(agent_icon,(30,30))

        # Control variable
        running = True  

        # Agent initial coordinates in environment
        agentX = 90
        agentY = 620
        dynamicAgent = agent_icon
        state = (agentX, agentY, 0, 0, 0, 0, 0, 0, 0, 0)

        # Initial Reward
        reward = 0
        count=0
        iteration = 0

        while running:

            iteration += 1
            environment.blit(backGround,(0,0))

            for wall in walls:
                environment.blit(wall[0],wall[1])

            environment.blit(table1,(50,50))
            environment.blit(table2,(260,50))
            environment.blit(sofa1,(30,430))
            environment.blit(sofa2,(290,570))
            environment.blit(roundTable,(180,500))
            environment.blit(sofa3,(460,30))

            for f in fires:
                if iteration%3 == 0:
                    environment.blit(fire1,(f.left,f.top))
                elif iteration%3 == 1:
                    environment.blit(fire2,(f.left,f.top))
                else:
                    environment.blit(fire3,(f.left,f.top))



            # Actively listen for event performed
            for event in pygame.event.get():  

                if event.type == pygame.QUIT:  

                    # Reset control variable to break
                    running = False
            
            pygame.time.delay(20)
            pygame.display.flip()


RealTimeEnvironment()

