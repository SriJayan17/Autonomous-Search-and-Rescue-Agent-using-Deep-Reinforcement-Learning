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

    flag = False

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

        exit_icon = pygame.image.load('Project/Resources/Images/exit.jpg')
        exit_icon = pygame.transform.scale(exit_icon,(50,30))

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
        rewardHandler = RewardHandler(grid, dynamicObstacles[index], dynamicFireFlares[index], borders, dynamicVictims[index], DynamicEnvironment.flag)
        
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
            if count > 4000 and count <= 8000:
                index = 1
                grid = computeGrid(dynamicObstacles[index], dynamicFireFlares[index])
                if DynamicEnvironment.flag:
                    rewardHandler = RewardHandler(grid, dynamicObstacles[index], dynamicFireFlares[index], borders, pygame.Rect(60,35,50,30),DynamicEnvironment.flag)
                else:
                    rewardHandler = RewardHandler(grid, dynamicObstacles[index], dynamicFireFlares[index], borders, dynamicVictims[index],DynamicEnvironment.flag)

            elif count > 8000 and count <= 16000:
                index = 2
                grid = computeGrid(dynamicObstacles[index], dynamicFireFlares[index])
                if DynamicEnvironment.flag:
                    rewardHandler = RewardHandler(grid, dynamicObstacles[index], dynamicFireFlares[index], borders, pygame.Rect(60,35,50,30),DynamicEnvironment.flag)
                else:
                    rewardHandler = RewardHandler(grid, dynamicObstacles[index], dynamicFireFlares[index], borders, dynamicVictims[index],DynamicEnvironment.flag)

            elif count > 16000:
                count = 0
                index = 0
                grid = computeGrid(dynamicObstacles[index], dynamicFireFlares[index])
                if DynamicEnvironment.flag:
                    rewardHandler = RewardHandler(grid, dynamicObstacles[index], dynamicFireFlares[index], borders, pygame.Rect(60,35,50,30),DynamicEnvironment.flag)
                else:
                    rewardHandler = RewardHandler(grid, dynamicObstacles[index], dynamicFireFlares[index], borders, dynamicVictims[index],DynamicEnvironment.flag)


            # Set background color as white
            environment.fill((0,0,0))

            # Fill the borders
            for boundary in boundaries:
                pygame.draw.rect(environment, (255, 0, 0), boundary)

            # Load the fire flares into environment 
            for fireFlare in dynamicFireFlares[index]:
                environment.blit(fire,(fireFlare.left,fireFlare.top))

            # Loads the Victims into environment
            if not DynamicEnvironment.flag: 
                environment.blit(victims,(dynamicVictims[index].left,dynamicVictims[index].top))
            else : environment.blit(exit_icon,(60,35))
            

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

            if DynamicEnvironment.flag and reward == 2:
                #Stop the timer and measure the time:
                time_lapse = time.time() - start
                time_grapher.plot_graph(time_lapse)
                
                #Plotting the number of correct decision made with time:
                dec_grapher.plot_decision_graph()
                
                agent.save_brain()
                pygame.image.save(environment,"Project/Resources/Images/Destination-Reached-realtime.jpg")
                
                agent.plot_reward_metric()
                
                root = Tk()
                root.withdraw()
                messagebox.showinfo("Result","Agent successfully rescued the victims!")
                running = False
                
            if not DynamicEnvironment.flag and reward == 2:
                root = Tk()
                root.withdraw()
                messagebox.showinfo("Result","Agent successfully reached the victims!")
                reward = 0
                DynamicEnvironment.flag = True
                rewardHandler = RewardHandler(grid, dynamicObstacles[index], dynamicFireFlares[index],borders, pygame.Rect(60,35,50,30), DynamicEnvironment.flag)

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
            # pygame.time.delay(5)
            pygame.display.flip()