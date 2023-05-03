import pygame
import numpy as np
from math import sin,cos

from Project.Backend.Agent import Agent
from Project.FrontEnd.Utils.Action_Handler import get_sensors
from Project.FrontEnd.Utils.Training_Env_Obstacles import test_obstacleGrid

pygame.init()
pygame.display.set_caption("Search and Rescue Simulation")
environment = pygame.display.set_mode((600, 400))


stopSimulation = False
agent = Agent(3,1,10,(300,200))
# agent_shape = pygame.Surface((30,12))
# agent_shape.fill((0,0,255))  # Color of the agent
# agent_shape.set_colorkey((0,0,0))

# shape_copy = agent_shape.copy()
# agent_rect = shape_copy.get_rect()
# agent_rect.center = (300,200)

# environment.blit(shape_copy,agent_rect)
# velocity_vec = (6,0)

# def rotate_vec(vec,angle):
#     target = vec
#     theta = np.deg2rad(angle)
#     rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
#     return np.dot(rot,target)

# def move(angle):
#     global velocity_vec, agent_rect
#     velocity_vec = rotate_vec(velocity_vec,angle)
#     agent_rect.center = (agent_rect.centerx + velocity_vec[0], agent_rect.centery + velocity_vec[1])
agent.turn(50)


while not stopSimulation:

    environment.fill((0,0,0))

    for event in pygame.event.get():  

        if event.type == pygame.QUIT:  
            stopSimulation = True
        
        # Manual Control:
        if event.type == pygame.KEYDOWN:
            
            if event.key == pygame.K_UP:
                agent.move(12)

            if event.key == pygame.K_LEFT:
                agent.turn(-15)
                agent.move(12)

            if event.key == pygame.K_RIGHT:
                agent.turn(15)
                agent.move(12)

    # # Manual Control
    environment.blit(agent.shape_copy,agent.rect)
    _,sensor_pts = get_sensors(agent,test_obstacleGrid,10,30)
    for pt in sensor_pts:
        pygame.draw.circle(environment,(255,0,0),pt,10)
    # if(reachedVictims(self.agentModels[0])):
    #     self.stop()

    # total_timesteps += 1
    # if total_timesteps % 1000 == 0: print(f'Timsesteps: {total_timesteps}')
    pygame.display.flip()
    # pygame.time.delay(10)