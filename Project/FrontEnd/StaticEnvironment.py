from copyreg import pickle
from matplotlib import pyplot as plt
import pygame
from Project.Backend.Agent import Agent
from Project.FrontEnd.Utils.StaticObstacles import grid,borders,boundaries,obstacles,fireFlares,victimsRect
from Project.FrontEnd.Utils.RewardHandler import RewardHandler
from tkinter import messagebox
from tkinter import *
import time,os
import pickle

class StaticEnvironment:

    def __init__(self):

        # Initialising objects
        agent = Agent(9,3,'DQN')
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
        
        #Switch to start the timer:
        timer_switch = True
        
        #Loading the time-lapse record:
        time_lapse_record = []
        log_file_path =  os.path.join(os.getcwd(),'Project\\Resources\\log\\static.txt')
        try:
            time_lapse_record = pickle.load(open(log_file_path,'rb'))
        except EOFError as e:
            print('There wasn\'t any saved instance of log record!')
            time_lapse_record = []
        
        #Keeping track of the no.of correct decisions taken:
        num_correct_dec = []
        correct_dec_track = 0
        iter_count = 0
        
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
            
            #The timer is started when the agent makes the first move
            if timer_switch:
                start = time.time()
                timer_switch = False
                
            action = agent.take_action(reward,state)
            reward,nextState = rewardHandler.generateReward(dynamicAgent,state,action)
            
            if reward > 0: correct_dec_track += 1
            #Tracking the number f correct decisions for evry 100 iterations:
            if iter_count == 100:
                num_correct_dec.append(correct_dec_track)
                correct_dec_track = 0
                iter_count = 0
                
            # Blits the agent into environment based on currentState
            state = nextState
            dynamicAgent = pygame.transform.rotate(agent_icon,state[2])
            environment.blit(dynamicAgent,(state[0],state[1]))

            if reward == 2:
                #Stop the timer and measure the time:
                time_lapse = time.time() - start
                print(f'Time taken: {time_lapse} seconds')
                time_lapse_record.append(time_lapse)
                if len(time_lapse_record) > 1:
                    plt.plot(time_lapse_record)
                    plt.xlabel('No. of executions/training epochs')
                    plt.ylabel('Time to reach victims(in seconds)')
                    plt.show()
                try:
                    pickle.dump(time_lapse_record,open(log_file_path,'wb'))
                except Exception as e:
                    print(f'Exception occured: {e}')
                
                #Plotting the number of correct decision made with time:
                print(num_correct_dec)
                plt.plot(num_correct_dec)
                plt.xlabel('Time')
                plt.ylabel('Number of correct decisions made')
                plt.show()
                
                agent.save_brain()
                
                pygame.image.save(environment,"Project/Resources/Images/Destination Reached.jpg")
                
                #The reward accumulation plot is made:
                agent.plot_reward_metric()
                root = Tk()
                root.withdraw()
                messagebox.showinfo("Result","Agent successfully reached destination!")
                running = False
            
            iter_count += 1
            
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