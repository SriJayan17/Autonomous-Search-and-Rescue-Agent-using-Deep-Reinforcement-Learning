from os import stat
from numpy import angle
import pygame
from Utils.StaticObstacles import obstacles,grid
import random
# from Backend.Agent import Agent

# To set the environment dimensions same as screen resolution
width = 700
height = 700

# Parameters
thetha = 90

# Predict next state based on current state and action:
def predictNextState(state,action):

    nextState = state
    if action == 0:
        if state[2] > 0 and state[2] < 90:
            nextState = (state[0],state[1] - 5,state[2])

        elif state[2] >= 90 and state[2] < 180:
            nextState = (state[0] - 5,state[1],state[2])

        elif state[2] >= 180 and state[2] < 270:
            nextState = (state[0],state[1] + 5,state[2])
        
        elif state[2] >= 270 and state[2] < 360:
            nextState = (state[0] + 5,state[1],state[2])

        elif state[2] < 0 and state[2] > -90:
            nextState = (state[0] + 5,state[1],state[2])

        elif state[2] <= -90 and state[2] > -180:
            nextState = (state[0],state[1] + 5,state[2])

        elif state[2] <= -180 and state[2] > -270:
            nextState = (state[0] - 5,state[1],state[2])
        
        elif state[2] <= -270 and state[2] > -360:
            nextState = (state[0],state[1] - 5,state[2])
        
        else:
            nextState = (state[0],state[1] - 5,state[2])

    elif action == 1:
        angle = 0
        if state[2] + thetha < 360:
            angle = state[2] + thetha

        nextState = (state[0],state[1],angle)

    else:
        angle = 0
        if state[2] - thetha < -360:
            angle = state[2] - thetha
        nextState = (state[0],state[1],angle)
    
    return nextState

# Reward generation based on action performed
def generateReward(currentState,action):
    
    nextState = predictNextState(currentState,action)

    # Negative reward if agent hits the obstacles
    x = nextState[0]
    y = nextState[1]
    height = 40
    width = 40
    if x+width > 700 or y+height > 700:
        return -1,currentState

    if 1 in grid[x:x+width][y:y+width]:
        return -1,currentState
    
    # Higher Negative reward if agent will hit fire flares
    elif 2 in grid[x:x+width][y:y+width]:
        return -5,currentState
    
    else:
        return 1,nextState

# To inilialize
pygame.init()
pygame.display.set_caption("Autonomous Search and Rescue Simulation")


dimensions = (width, height)
environment = pygame.display.set_mode(dimensions)  

# Loading and scaling up the images
fire = pygame.image.load("Resources/Images/fire.png")
fire = pygame.transform.scale(fire,(40,40))

victims = pygame.image.load("Resources/Images/victims.png")
victims = pygame.transform.scale(victims,(50,50))

# agent = pygame.image.load("Resources/Images/agent.png")
# agent = pygame.transform.scale(agent,(40,40))

# Control variable
running = True  

# Agent coordinates in environment
agentX = 90
agentY = 620
agent = pygame.Rect(agentX, agentY,40,40)
state = (agentX, agentY, 0)
reward = 0

while running:

    # Set background color as white
    environment.fill((0,0,0))

    # Fill the borders
    pygame.draw.rect(environment, (255, 0, 0), pygame.Rect(0, 0, 50, 20))
    pygame.draw.rect(environment, (255, 0, 0), pygame.Rect(120, 0, width-120, 20))
    pygame.draw.rect(environment, (255, 0, 0), pygame.Rect(0, 0, 20, height))
    pygame.draw.rect(environment, (255, 0, 0), pygame.Rect(0, height-20, width, 20))
    pygame.draw.rect(environment, (255, 0, 0), pygame.Rect(width-20, 0, 20, 410))
    pygame.draw.rect(environment, (255, 0, 0), pygame.Rect(width-20, 500, 20, 200))

    # Load the fire flares into environment 
    environment.blit(fire,(305,70))
    environment.blit(fire,(600,240))
    environment.blit(fire,(430,390))

    # Loads the Victims into environment
    environment.blit(victims,(420,250))

    # Rendering the obstacles in environment
    for obstacle in obstacles:
        pygame.draw.rect(environment,(255, 0, 0), obstacle)
    
    # action = Agent.take_action(reward,state)
    action = random.randint(0,1)
    reward,nextState = generateReward(state,action)

    # Loads the agent into environment

    state = nextState
    # rotatedAgent = pygame.transform.rotate(agent,state[2])
    
    pygame.draw.rect(environment,(255,255,255),agent)
    # environment.blit(agent,(state[0],state[1]))

    # Actively listen for event performed
    for event in pygame.event.get():  

        if event.type == pygame.QUIT:  

            # Reset control variable if even type is quit
            running = False  
            
        if event.type == pygame.KEYDOWN:

            
            if event.key == pygame.K_UP:
                agentY -= 10
                environment.blit(agent,(agentX,agentY))

            if event.key == pygame.K_LEFT:
                print("In")
                angle += 30
                rotatedAgent = pygame.transform.rotate(agent,angle)
                environment.blit(agent,(state[0],state[1]))
                # state = (state[0], state[1], angle)

            if event.key == pygame.K_RIGHT:
                print("In")
                angle -= 30
                rotatedAgent = pygame.transform.rotate(agent,angle)
                environment.blit(agent,(state[0],state[1]))
                # state = (state[0], state[1], angle)
        
    pygame.display.flip() 
