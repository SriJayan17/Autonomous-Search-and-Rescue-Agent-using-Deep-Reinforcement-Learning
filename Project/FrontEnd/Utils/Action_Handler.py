import math
import numpy as np
from Project.FrontEnd.Utils.Training_Env_Obstacles import *
from Project.FrontEnd.Utils.Training_Env_Obstacles import height,width

#To check if two rectangles are colliding
def isColliding(objects, rect):
    for object in objects:
        if object.colliderect(rect):
            return True
    
    return False

# To check if a particular move causes an agent to collide with borders, obstacles, boundaries,
# or other agents
def isPermissible(agent_list, index):
    objects = [boundaries, obstacles]
    objects.append((agent_list[i].rect for i in range(len(agent_list)) if i != index))
    current_rect = agent_list[index].rect
    current_rect_center = current_rect.center

    #Checking if the agent has collided with an obstacle
    for object in objects:
        if(isColliding(object, current_rect)):
            return False

    #Checking if the agent has hit the borders    
    if (current_rect_center[0] < 0 or current_rect_center[0] > width) or \
       (current_rect_center[1] < 0 or current_rect_center[1] > height):
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
    
# To calculate euclidean distance
def eucledianDist(a,b):
    return math.sqrt(math.pow((a[0]-b[0]),2) + math.pow((a[1]-b[1]),2))

#Generating the agent state that is has to be fed into the neural network to make decisions
# State = own_state + neighbor_state + flock_center (precise)
# TODO: Add a switch to toggle between training scenario and testing scenario
def prepare_agent_state(agent_list,target_index,state_dict,initial_state_dict,vicinity_radius=600):
    running_x = running_y = 0
    result = []
    n = len(agent_list)
    for i in range(n):
        if i == target_index:
            result.extend(state_dict[i])
        else:
            target_agent = agent_list[target_index]
            distance = eucledianDist((target_agent.rect.x, target_agent.rect.y),
                                     (agent_list[i].rect.x,agent_list[i].rect.y))
            if distance <= vicinity_radius:
                result.extend(state_dict[i])
            else:
                result.extend(initial_state_dict[i])
        running_x += agent_list[i].rect.x
        running_y += agent_list[i].rect.y

    # Adding the flock center
    result.append(running_x / n)
    result.append(running_y / n)
    return result

# To calculate the obstacle/fire intensity, given the top left point and the grid in concern.
def get_sensor_output(left,top,dim_x,dim_y,boundary,grid):
    # row_num = top + 30
    # col_num = left + 30
    row_num = top + boundary
    col_num = left + boundary
    sensor_area = grid[row_num:row_num+dim_y,col_num:col_num+dim_x].copy()
    sensor_area = sensor_area.ravel()
    sensor_output = np.sum(sensor_area)
    return sensor_output

# Generate state for an individual agent
def get_state(agent_rect,obstacle_grid,fire_grid,target_point,extra_info):
    
    x = agent_rect.x
    y = agent_rect.y
    
    # add obstacle_density


    # add heat_intensity

    return [x/extra_info['scale_x'], y/extra_info['scale_y']]

def reachedVictims(agent):
    return victimsRect.colliderect(agent.rect)

