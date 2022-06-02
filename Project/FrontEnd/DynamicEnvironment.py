import time
import pygame
import os
from Project.Backend.Agent import Agent
from Project.FrontEnd.Utils.DecisionGrapher import DecisionGrapher
from Project.FrontEnd.Utils.DynamicObstacles import borders,boundaries,dynamicObstacles,dynamicFireFlares,dynamicVictims
from Project.FrontEnd.Utils.RewardHandler import RewardHandler
from Project.FrontEnd.Utils.DynamicGrid import computeGrid
from tkinter import messagebox
from tkinter import *
from Project.FrontEnd.Utils.TimeGrapher import TimeGrapher

class DynamicEnvironment:

    def __init__(self):

        # Initialising objects
        agent = Agent(9,3,'DQN')

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

        agentIcon = pygame.image.load("Project/Resources/Images/agent.png")
        agentIcon = pygame.transform.scale(agentIcon,(30,30))

        # Control variable
        running = True  

        # Agent initial coordinates in environment
        agentX = 90
        agentY = 620
        dynamicAgent = agentIcon
        state = (agentX, agentY, 0, 0, 0, 0, 0, 0, 0, 0)

        # Initial Reward
        reward = 0

        # Dynamic rendnering
        count = 0
        index = 0
        grid = computeGrid(dynamicObstacles[index], dynamicFireFlares[index])
        rewardHandler = RewardHandler(grid, dynamicObstacles[index], dynamicFireFlares[index], borders, dynamicVictims[index])
        
        #switch to denote the starting of the timer to record the time taken to reach
        # the victims:
        timer_switch = True
        
        #Creating the TimeGrapher object to plot the time taken:
        time_grapher = TimeGrapher(os.path.join(os.getcwd(),'Project\\Resources\\log\\dynamic.txt'))
        
        #Creating the decision grapher to plot the number of correct decsions made during
        # the runtime:
        dec_grapher = DecisionGrapher()
        
        while running:

            count+=1
            if count > 2500 and count <= 5000:
                index = 1
                grid = computeGrid(dynamicObstacles[index], dynamicFireFlares[index])
                rewardHandler = RewardHandler(grid, dynamicObstacles[index], dynamicFireFlares[index], borders, dynamicVictims[index])
            elif count > 5000 and count <= 10000:
                index = 2
                grid = computeGrid(dynamicObstacles[index], dynamicFireFlares[index])
                rewardHandler = RewardHandler(grid, dynamicObstacles[index], dynamicFireFlares[index], borders, dynamicVictims[index])
            elif count > 10000:
                count = 0
                index = 0
                grid = computeGrid(dynamicObstacles[index], dynamicFireFlares[index])
                rewardHandler = RewardHandler(grid, dynamicObstacles[index], dynamicFireFlares[index], borders, dynamicVictims[index])

            # Set background color as white
            environment.fill((0,0,0))

            # Fill the borders
            for boundary in boundaries:
                pygame.draw.rect(environment, (255, 0, 0), boundary)

            # Load the fire flares into environment 
            for fireFlare in dynamicFireFlares[index]:
                environment.blit(fire,(fireFlare.left,fireFlare.top))

            # Loads the Victims into environment
            environment.blit(victims,(dynamicVictims[index].left,dynamicVictims[index].top))

            # Rendering the obstacles in environment
            for obstacle in dynamicObstacles[index]:
                pygame.draw.rect(environment,(255, 0, 0), obstacle)
            
            action = agent.take_action(reward,state)
            reward,nextState = rewardHandler.generateReward(dynamicAgent,state,action)

            dec_grapher.correct_decision(reward > 0)
            
            if timer_switch:
                start = time.time()
                timer_switch = False
                
            # Blits the agent into environment based on currentState
            state = nextState
            dynamicAgent = pygame.transform.rotate(agentIcon,state[2])
            environment.blit(dynamicAgent,(state[0],state[1]))

            if reward == 2:

                # Plotting the time taken graph:
                time_lapsed = time.time() - start
                time_grapher.plot_graph(time_lapsed)
                
                # Plotting the number of correct decisions made with time:
                dec_grapher.plot_decision_graph()
                
                # Plotting the cumulative reward of the agent:
                agent.plot_reward_metric()
                
                agent.save_brain()
                pygame.image.save(environment,"Project/Resources/Images/Destination-Reached-dynamic.jpg")
                root = Tk()
                root.withdraw()
                messagebox.showinfo("Result","Agent successfully reached destination!")
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
                        dynamicAgent = pygame.transform.rotate(agentIcon,state[2])
                        environment.blit(dynamicAgent,(state[0],state[1]))

                    if event.key == pygame.K_LEFT:
                        reward,nextState = rewardHandler.generateReward(dynamicAgent,state,1)
                        print(reward,nextState)
                        state = nextState
                        dynamicAgent = pygame.transform.rotate(agentIcon,state[2])
                        environment.blit(dynamicAgent,(state[0],state[1]))

                    if event.key == pygame.K_RIGHT:
                        reward,nextState = rewardHandler.generateReward(dynamicAgent,state,2)
                        print(reward,nextState)
                        state = nextState
                        dynamicAgent = pygame.transform.rotate(agentIcon,state[2])
                        environment.blit(dynamicAgent,(state[0],state[1]))

            # To make simulation smooth                 
            pygame.time.delay(5)
            pygame.display.flip()