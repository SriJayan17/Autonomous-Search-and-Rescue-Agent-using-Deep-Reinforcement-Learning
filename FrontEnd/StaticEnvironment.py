import pygame
from Utils.StaticObstacles import obstacles,grid

# To inilialize
pygame.init()
pygame.display.set_caption("Autonomous Search and Rescue Simulation")

# To set the environment dimensions same as screen resolution
width = 700
height = 700

dimensions = (width, height)
environment = pygame.display.set_mode(dimensions)  

# Loading Fire flare Image
fire = pygame.image.load("Resources/Images/fire.png")

# Scaling up the image
fire = pygame.transform.scale(fire,(40,40))

# Loading Victims image
victims = pygame.image.load("Resources/Images/victims.png")
victims = pygame.transform.scale(victims,(50,50))

# Loading agent image
agent = pygame.image.load("Resources/Images/agent.png")
agent = pygame.transform.scale(agent,(40,40))

# Control variable
running = True  
agentX = 90
agentY = 620

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
    
    # Loads the agent into environment
    environment.blit(agent,(agentX,agentY))


    # Actively listen for event performed
    for event in pygame.event.get():  

        if event.type == pygame.QUIT:  

            # Reset control variable if even type is quit
            running = False  
            
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                agentX -= 10
                environment.blit(agent,(agentX,agentY))

            if event.key == pygame.K_RIGHT:
                agentX += 10
                environment.blit(agent,(agentX,agentY))

            if event.key == pygame.K_UP:
                agentY -= 10
                environment.blit(agent,(agentX,agentY))

            if event.key == pygame.K_DOWN:
                agentY += 10
                environment.blit(agent,(agentX,agentY))

    if grid[agentX][agentY] == 1 :
        print("collision")
        
    pygame.display.flip() 
