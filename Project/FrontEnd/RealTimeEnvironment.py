import os
import time
import pygame
from Project.Backend.Agent import Agent
from Project.FrontEnd.Utils.DecisionGrapher import DecisionGrapher
from Project.FrontEnd.Utils.RealTimeObstacles import fireFlares,grid,obstacles,borders,victimsRect,walls
from Project.FrontEnd.Utils.RewardHandler import RewardHandler
from tkinter import messagebox
from tkinter import *

from Project.FrontEnd.Utils.TimeGrapher import TimeGrapher

class RealTimeEnvironment:

    def __init__(self):

        # Initialising objects
        agent = Agent(9,3,'DQN')
        obstacles.extend(walls)
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
        backGround = pygame.image.load("Project/Resources/Images/floor.png")
        backGround = pygame.transform.scale(backGround,(width,height))

        wall = pygame.image.load("Project/Resources/Images/wall1.png")
        wall_images = []
        wall_images.append([pygame.transform.scale(wall,(20,height)),(0,0)])
        wall_images.append([pygame.transform.scale(wall,(20,height)),(width-20,0)])
        wall_images.append([pygame.transform.scale(wall,(width-40,20)),(20,0)])
        wall_images.append([pygame.transform.scale(wall,(60,20)),(20,height-20)])
        wall_images.append([pygame.transform.scale(wall,(width-160,20)),(160,height-20)])
        wall_images.append([pygame.transform.scale(wall,(20,20)),(400,20)])
        wall_images.append([pygame.transform.scale(wall,(20,200)),(400,120)])
        wall_images.append([pygame.transform.scale(wall,(20,280)),(400,400)])
        wall_images.append([pygame.transform.scale(wall,(180,20)),(500,300)])
        wall_images.append([pygame.transform.scale(wall,(300,20)),(20,400)])
        wall_images.append([pygame.transform.scale(wall,(90,20)),(420,500)])
        wall_images.append([pygame.transform.scale(wall,(90,20)),(590,500)])

        table1 = pygame.image.load("Project/Resources/Images/table1.png")
        table1 = pygame.transform.scale(table1,(130,90))
        table1 = pygame.transform.rotate(table1,90)

        sofa1 = pygame.image.load("Project/Resources/Images/sofa1.png")
        sofa1 = pygame.transform.scale(sofa1,(100,100))
        sofa1 = pygame.transform.rotate(sofa1,90)

        sofa2 = pygame.transform.rotate(sofa1,180)

        sofa3 = pygame.image.load("Project/Resources/Images/sofa2.png")
        sofa3 = pygame.transform.scale(sofa3,(180,60))
        sofa3 = pygame.transform.rotate(sofa3,180)

        table2 = pygame.image.load("Project/Resources/Images/table2.png")
        table2 = pygame.transform.scale(table2,(110,110))
        table2 = pygame.transform.rotate(table2,-90)

        roundTable = pygame.image.load("Project/Resources/Images/roundtable.png")
        roundTable = pygame.transform.scale(roundTable,(100,60))

        fridge = pygame.image.load("Project/Resources/Images/fridge.png")
        fridge = pygame.transform.scale(fridge,(70,70))

        bed = pygame.image.load("Project/Resources/Images/bed.png")
        bed = pygame.transform.scale(bed,(70,120))

        fire1 = pygame.image.load("Project/Resources/Images/fire1.png")
        fire1 = pygame.transform.scale(fire1,(60,60))

        fire2 = pygame.image.load("Project/Resources/Images/fire2.png")
        fire2 = pygame.transform.scale(fire2,(60,60))

        fire3 = pygame.image.load("Project/Resources/Images/fire3.png")
        fire3 = pygame.transform.scale(fire3,(60,60))

        victims = pygame.image.load("Project/Resources/Images/victims.png")
        victims = pygame.transform.scale(victims,(65,65))

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
        iteration = 0
        
        #Switch to start the timer:
        timer_switch = True
        
        #Loading the time-lapse record:
        time_grapher = TimeGrapher(os.path.join(os.getcwd(),'Project\\Resources\\log\\real.txt'))
        
        #To plot the number of correct decisions taken with time:
        dec_grapher = DecisionGrapher()

        while running:

            iteration += 1
            environment.blit(backGround,(0,0))

            for wall in wall_images:
                environment.blit(wall[0],wall[1])

            # for obs in obstacles:
            #     pygame.draw.rect(environment, (255, 0, 0), obs)

            environment.blit(table1,(50,50))
            environment.blit(sofa1,(60,480))
            environment.blit(sofa2,(250,540))
            environment.blit(sofa3,(425,100))
            environment.blit(table2,(260,150))
            environment.blit(roundTable,(180,480))
            environment.blit(fridge,(370,325))
            environment.blit(bed,(580,350))

            for f in fireFlares:
                if iteration%12 == 0:
                    environment.blit(fire1,(f.left,f.top))
                elif iteration%3 in [1,2,3,4,5,6]:
                    environment.blit(fire2,(f.left,f.top))
                else:
                    environment.blit(fire3,(f.left,f.top))

            environment.blit(victims,(430,600))

            #The timer is started when the agent makes the first move
            if timer_switch:
                start = time.time()
                timer_switch = False
                
            action = agent.take_action(reward,state)
            reward,nextState = rewardHandler.generateReward(dynamicAgent,state,action)
            
            dec_grapher.correct_decision(reward > 0)
            
            if reward == 2:
                #Stop the timer and measure the time:
                time_lapse = time.time() - start
                time_grapher.plot_graph(time_lapse)
                
                #Plotting the number of correct decision made with time:
                dec_grapher.plot_decision_graph()
                
                agent.save_brain()
                pygame.image.save(environment,"Project/Resources/Images/Destination Reached.jpg")
                
                agent.plot_reward_metric()
                
                root = Tk()
                root.withdraw()
                messagebox.showinfo("Result","Agent successfully reached destination!")
                running = False

            state = nextState
            dynamicAgent = pygame.transform.rotate(agent_icon,state[2])
            environment.blit(dynamicAgent,(state[0],state[1]))


            # Actively listen for event performed
            for event in pygame.event.get():  

                if event.type == pygame.QUIT:  

                    # Reset control variable to break
                    running = False
            
            pygame.time.delay(5)
            pygame.display.flip()

