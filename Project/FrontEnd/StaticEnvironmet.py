import pygame
from Project.Backend.Agent import Agent
from Project.FrontEnd.Utils.StaticObstacles import grid,borders,boundaries,obstacles,fireFlares,victimsRect
from Project.FrontEnd.Utils.RewardHandler import RewardHandler

class StaticEnvironment:

    def __init__(self):

        # Initialising objects
        agent = Agent(8,3,'DQN')
        rewardHandler = RewardHandler(grid, obstacles, fireFlares, borders, victimsRect)

        # Environment Dimensions
        width = 700
        height = 700

        # Inilialize pygame
        pygame.init()
        pygame.display.set_caption("Autonomous Search and Rescue Simulation")


        dimensions = (width, height)
        environment = pygame.display.set_mode(dimensions)  

        # Loading and scaling up the images
        fire = pygame.image.load("Project/Resources/Images/fire.png")
        fire = pygame.transform.scale(fire,(40,40))

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

        while running:

            # Set background color as white
            environment.fill((0,0,0))
            # Fill the borders
            for boundary in boundaries:
                pygame.draw.rect(environment, (255, 0, 0), boundary)

            # Load the fire flares into environment 
            environment.blit(fire,(305,70))
            environment.blit(fire,(600,240))
            environment.blit(fire,(430,390))

            # Loads the Victims into environment
            environment.blit(victims,(420,250))

            # Rendering the obstacles in environment
            for obstacle in obstacles:
                pygame.draw.rect(environment,(255, 0, 0), obstacle)
            
            action = agent.take_action(reward,state)
            reward,nextState = rewardHandler.generateReward(dynamicAgent,state,action)

            # Blits the agent into environment based on currentState
            state = nextState
            dynamicAgent = pygame.transform.rotate(agent_icon,state[2])
            environment.blit(dynamicAgent,(state[0],state[1]))

            if reward == 2:
                pygame.image.save(environment,"./Resources/Images/Destination Reached.jpg")
                running = False

            # Actively listen for event performed
            for event in pygame.event.get():  

                if event.type == pygame.QUIT:  

                    # Reset control variable to break
                    running = False  
                    
                if event.type == pygame.KEYDOWN:
                    
                    if event.key == pygame.K_UP:
                        reward,nextState = rewardHandler.generateReward(dynamicAgent,state,0)
                        print(reward,nextState)
                        state = nextState
                        dynamicAgent = pygame.transform.rotate(agent_icon,state[2])
                        environment.blit(dynamicAgent,(state[0],state[1]))

                    if event.key == pygame.K_LEFT:
                        reward,nextState = rewardHandler.generateReward(dynamicAgent,state,1)
                        print(reward,nextState)
                        state = nextState
                        dynamicAgent = pygame.transform.rotate(agent_icon,state[2])
                        environment.blit(dynamicAgent,(state[0],state[1]))

                    if event.key == pygame.K_RIGHT:
                        reward,nextState = rewardHandler.generateReward(dynamicAgent,state,2)
                        print(reward,nextState)
                        state = nextState
                        dynamicAgent = pygame.transform.rotate(agent_icon,state[2])
                        environment.blit(dynamicAgent,(state[0],state[1]))

            # To make simulation smooth                 
            # pygame.time.delay(5)
            pygame.display.flip()