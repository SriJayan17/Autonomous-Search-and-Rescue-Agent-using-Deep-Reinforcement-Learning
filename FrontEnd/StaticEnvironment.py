import pygame
from Utils.StaticObstacles import borders,boundries,obstacles,fireFlares,victimRect
import random
# from Backend.Agent import Agent

# Environment Dimensions
width = 700
height = 700

# Action Parameters
theta = 90
unit = 5

# Inilialize pygame
pygame.init()
pygame.display.set_caption("Autonomous Search and Rescue Simulation")


dimensions = (width, height)
environment = pygame.display.set_mode(dimensions)  

# Loading and scaling up the images
fire = pygame.image.load("Resources/Images/fire.png")
fire = pygame.transform.scale(fire,(40,40))

victims = pygame.image.load("Resources/Images/victims.png")
victims = pygame.transform.scale(victims,(50,50))

agent = pygame.image.load("Resources/Images/agent.png")
agent = pygame.transform.scale(agent,(33,33))

# Control variable
running = True  

# Agent initial coordinates in environment
agentX = 90
agentY = 620
state = (agentX, agentY, 0)

# Initial Reward
reward = 0

# Predict next state based on current state and action:
def predictNextState(state,action):

    nextState = state
    if action == 0:
        if state[2] > 0 and state[2] < 90:
            nextState = (state[0],state[1] - unit,state[2])

        elif state[2] >= 90 and state[2] < 180:
            nextState = (state[0] - unit,state[1],state[2])

        elif state[2] >= 180 and state[2] < 270:
            nextState = (state[0],state[1] + unit,state[2])
        
        elif state[2] >= 270 and state[2] < 360:
            nextState = (state[0] + unit,state[1],state[2])

        elif state[2] < 0 and state[2] >= -90:
            nextState = (state[0] + unit,state[1],state[2])

        elif state[2] < -90 and state[2] >= -180:
            nextState = (state[0],state[1] + unit,state[2])

        elif state[2] < -180 and state[2] >= -270:
            nextState = (state[0] - unit,state[1],state[2])
        
        elif state[2] < -270 and state[2] >= -360:
            nextState = (state[0],state[1] - unit,state[2])
        
        else:
            nextState = (state[0],state[1] - unit,state[2])

    elif action == 1:
        angle = 0
        if state[2] + theta < 360:
            angle = state[2] + theta

        nextState = (state[0],state[1],angle)

    elif action == 2:
        angle = 0
        if state[2] - theta > -360:
            angle = state[2] - theta
        nextState = (state[0],state[1],angle)
    
    return nextState

# Reward generation based on action performed
def generateReward(currentState,action):
    
    nextState = predictNextState(currentState,action)

    agentRect = agent.get_rect(topleft = (nextState[0],nextState[1]))

    # Stop if agent reaches Destination
    if agentRect.colliderect(victimRect):
        print("Reached Destination")
        return 100,nextState

    # Negative reward if agent hits the obstacles
    for obstacle in obstacles:
        if agentRect.colliderect(obstacle):
            return -0.5,currentState
    
    # Negative reward if agent hits the Boundries
    for border in borders:
        if agentRect.colliderect(border):
            return -0.5,currentState

    # Higher Negative reward if agent will hit fire flares
    for fire in fireFlares:
        if agentRect.colliderect(fire):
            return -1,currentState

    return 1,nextState

while running:

    # Set background color as white
    environment.fill((0,0,0))

    # Fill the borders
    for boundry in boundries:
        pygame.draw.rect(environment, (255, 0, 0), boundry)

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
    action = random.randint(0,2)
    reward,nextState = generateReward(state,action)

    if reward == 100:
        running = False

    # Blits the agent into environment based on currentState
    state = nextState
    rotatedAgent = pygame.transform.rotate(agent,state[2])
    environment.blit(rotatedAgent,(state[0],state[1]))

    # Actively listen for event performed
    for event in pygame.event.get():  

        if event.type == pygame.QUIT:  

            # Reset control variable to break
            running = False  
            
        if event.type == pygame.KEYDOWN:

            if event.key == pygame.K_UP:
                reward,nextState = generateReward(state,0)
                state = nextState
                rotatedAgent = pygame.transform.rotate(agent,state[2])
                environment.blit(rotatedAgent,(state[0],state[1]))

            if event.key == pygame.K_LEFT:
                reward,nextState = generateReward(state,1)
                state = nextState
                rotatedAgent = pygame.transform.rotate(agent,state[2])
                environment.blit(rotatedAgent,(state[0],state[1]))

            if event.key == pygame.K_RIGHT:
                reward,nextState = generateReward(state,2)
                state = nextState
                rotatedAgent = pygame.transform.rotate(agent,state[2])
                environment.blit(rotatedAgent,(state[0],state[1]))

    # To make simulation smooth                 
    pygame.time.delay(50)
    pygame.display.flip() 
