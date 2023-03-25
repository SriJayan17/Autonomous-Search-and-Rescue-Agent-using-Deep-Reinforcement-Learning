import math
import numpy as np
from Project.FrontEnd.Utils.Training_Env_Obstacles import *

def isColliding(objects, rect):
    for object in objects:
        if object.colliderect(rect):
            return True
    
    return False

def isPermissible(current_rect, neighbours):

    objects = [borders,boundaries, obstacles, neighbours]

    for object in objects:
        if(isColliding(object, current_rect)):
            return False
    
    return True

def generateReward(previous_rect, current_rect):

    reward = 1

    previous_dist = eucledianDist((previous_rect.x,previous_rect.y),(victimsRect.x, victimsRect.y))
    current_dist = eucledianDist((current_rect.x, current_rect.y), (victimsRect.x, victimsRect.y))

    if previous_dist > current_dist:
        reward += 1
    
    for fire in fireFlares:
        if(fire.colliderect(current_rect)):
            reward -= 1.5
    
    return reward
    

def eucledianDist(a,b):
    return math.sqrt(math.pow((a[0]-b[0]),2) + math.pow((a[1]-b[1]),2))

def updateOwnState(agent, agents):
    x = agent.rect.x
    y = agent.rect.y
    
    # add obstacle_density
    # add heat_intensity

    sumX = sumY = 0
    for agent in agents:
        sumX = sumX + agent.rect.x
        sumY = sumY + agent.rect.y
    
    flock_center = (sumX/len(agents), sumY/len(agents))

    return [x, y, flock_center]

def appendNeighbourState(agent, agents, agents_state, agents_state_copy, initial_state):

    vicinity_radius = 600
    for neighbour in agents:
        if agent.rect != neighbour.rect:
            distance = eucledianDist((agent.rect.x, agent.rect.y),(neighbour.rect.x,neighbour.rect.y))
            if distance <= vicinity_radius:
                agents_state[agent].extend(agents_state_copy[neighbour])
            else:
                agents_state[agent].extend(initial_state[neighbour])



def updateState(agents, initial_state):
    state_map = {}
    for agent in agents:
        state_map[agent] = updateOwnState(agent, agents)

    state_map_copy = {}

    for map in state_map:
        state_map_copy[map] = list(state_map[map])

    for agent in agents:
        appendNeighbourState(agent, agents, state_map, state_map_copy, initial_state)
    
    return state_map

def reachedVictims(agent):
    return victimsRect.colliderect(agent.rect)
# Testing

# state_map = updateState(agents)
# for key in state_map:
#     print(key,state_map[key])

# print(generateReward(agents[1], agents[3]))   

