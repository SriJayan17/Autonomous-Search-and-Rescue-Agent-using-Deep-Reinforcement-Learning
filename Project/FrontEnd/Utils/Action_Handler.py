import math
import numpy as np
from Project.FrontEnd.Utils.Training_Env_Obstacles import *
from Project.FrontEnd.Utils.Training_Env_Obstacles import height,width
from functools import reduce

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

def generateReward(previous_center, current_rect, flock_center=None):
    reward = 0
    current_center = current_rect.center
    target_pt = victimsRect.center

    previous_target_dist = eucledianDist(previous_center, target_pt)
    current_target_dist = eucledianDist(current_center, target_pt)
    
    # Highly positive reward if the agent has reached the target:
    if reachedVictims(current_rect):
        reward += 5

    # Positive reward if the agent has moved towards the target
    if previous_target_dist > current_target_dist:
        reward += 1
    else:
        reward -= 1.5
    
    # Negative reward if the agent has flown into fire
    for fire in fireFlares:
        if(fire.colliderect(current_rect)):
            reward -= 3
    
    return reward
    
# To calculate euclidean distance
def eucledianDist(a,b):
    return math.sqrt(math.pow((a[0]-b[0]),2) + math.pow((a[1]-b[1]),2))

#To calculate flock center:
def calc_flock_center(agent_list):
    n = len(agent_list)
    x = sum([agent.rect.centerx for agent in agent_list])
    y = sum([agent.rect.centery for agent in agent_list])
    return (x/n, y/n)

# Generating the agent state that is has to be fed into the neural network to make decisions
# State = own_state + neighbor_state + flock_center (precise)
# TODO: Add a switch to toggle between training scenario and testing scenario
def prepare_agent_state(agent_list,target_index,state_dict,
                        initial_state_dict,flock_center=None,vicinity_radius=600):
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

    # Adding the flock center
    if flock_center is not None: result.extend(flock_center)
    return result

# Discreet function to calculate the obstacle/fire intensity, given the top left point, dimensions of target area and the grid in concern.
def get_sensor_output(left,top,dim_x,dim_y,boundary,grid):
    scaling_factor = dim_x * dim_y
    row_num = top + boundary
    col_num = left + boundary
    sensor_area = grid[row_num:row_num+dim_y,col_num:col_num+dim_x].copy()
    sensor_area = sensor_area.ravel()
    sensor_output = np.sum(sensor_area)
    return sensor_output / float(scaling_factor)

def get_sensors(target_player,target_grid,dim_x,dim_y,boundary):
    center = target_player.rect.center
    #Conversion of angle_track:
    rel_angle = target_player.get_proper_angle()
    result = []
    
    if rel_angle == 0:
        #Front
        result.append(get_sensor_output(center[0]+15,center[1]-15,dim_x,dim_y,boundary,target_grid))
        #Left
        result.append(get_sensor_output(center[0]-15,center[1]-36,dim_x,dim_y,boundary,target_grid))
        #Right
        result.append(get_sensor_output(center[0]-15,center[1]+6,dim_x,dim_y,boundary,target_grid))
    elif rel_angle == 90:
        result.append(get_sensor_output(center[0]-15,center[1]-45,dim_x,dim_y,boundary,target_grid))
        result.append(get_sensor_output(center[0]-36,center[1]-15,dim_x,dim_y,boundary,target_grid))
        result.append(get_sensor_output(center[0]+6,center[1]-15,dim_x,dim_y,boundary,target_grid))
    elif rel_angle == 180:
        result.append(get_sensor_output(center[0]-45,center[1]-15,dim_x,dim_y,boundary,target_grid))
        result.append(get_sensor_output(center[0]-15,center[1]+6,dim_x,dim_y,boundary,target_grid))
        result.append(get_sensor_output(center[0]-15,center[1]-36,dim_x,dim_y,boundary,target_grid))
    elif rel_angle == 270:
        result.append(get_sensor_output(center[0]-15,center[1]+15,dim_x,dim_y,boundary,target_grid))
        result.append(get_sensor_output(center[0]+6,center[1]-15,dim_x,dim_y,boundary,target_grid))
        result.append(get_sensor_output(center[0]-36,center[1]-15,dim_x,dim_y,boundary,target_grid))
    else:
        if rel_angle > 0 and rel_angle < 90:
            result.append(get_sensor_output(center[0],center[1]-30,dim_x,dim_y,boundary,target_grid))
            result.append(get_sensor_output(center[0]-30,center[1]-30,dim_x,dim_y,boundary,target_grid))
            result.append(get_sensor_output(center[0],center[1],dim_x,dim_y,boundary,target_grid))
        elif rel_angle > 90 and rel_angle < 180:
            result.append(get_sensor_output(center[0]-30,center[1]-30,dim_x,dim_y,boundary,target_grid))
            result.append(get_sensor_output(center[0]-30,center[1],dim_x,dim_y,boundary,target_grid))
            result.append(get_sensor_output(center[0],center[1]-30,dim_x,dim_y,boundary,target_grid))
        elif rel_angle > 180 and rel_angle < 270:
            result.append(get_sensor_output(center[0]-30,center[1],dim_x,dim_y,boundary,target_grid))
            result.append(get_sensor_output(center[0],center[1],dim_x,dim_y,boundary,target_grid))
            result.append(get_sensor_output(center[0]-30,center[1]-30,dim_x,dim_y,boundary,target_grid))
        else:
            result.append(get_sensor_output(center[0],center[1],dim_x,dim_y,boundary,target_grid))
            result.append(get_sensor_output(center[0],center[1]-30,dim_x,dim_y,boundary,target_grid))
            result.append(get_sensor_output(center[0]-30,center[1],dim_x,dim_y,boundary,target_grid))

    return result

# Generate state for an individual agent
def get_state(agent,extra_info):
    state_vec = []

    #Agent's current position:
    x = agent.rect.x / extra_info['scale_x']
    y = agent.rect.y / extra_info['scale_y']
    state_vec.extend([x,y])

    # add obstacle_density
    state_vec.extend(get_sensors(agent,obstacleGrid,extra_info['intensity_area_dim'],
                                 extra_info['intensity_area_dim'],row-height))
    # add heat_intensity
    state_vec.extend(get_sensors(agent,fireGrid,extra_info['intensity_area_dim'],
                                 extra_info['intensity_area_dim'],row-height))
    #Distance between agent and target:
    state_vec.append(eucledianDist(agent.rect.center,victimsRect.center) / extra_info['max_distance'])
    
    #Deviation_angle:
    state_vec.append(calc_deviation_angle(agent))

    return state_vec

def reachedVictims(agent_rect):
    return victimsRect.colliderect(agent_rect)

# Function to calculate the angle of deviation of the agent wrt the destination
def calc_deviation_angle(agent):
    pt_1 = list(agent.rect.center)
    pt_2 = list(victimsRect.center)
    # Apply transformation to resemble pts in a real cartesian plane
    pt_1[1] = height - pt_1[1]
    pt_2[1] = height - pt_2[1]
    dest_vec = (pt_2[0]-pt_1[0], pt_2[1]-pt_1[1])
    # Getting the orientation angle wrt X-axis:
    orient_angle = agent.get_proper_angle()
    # Getting the angle made by the vector(agent_cent -> target) with X-axis
    dest_vec_angle = angle_between_vectors(dest_vec,(1,0))
    # Get the transformed y-cordinate (Like the usual cartesian plane):
    pt_1[1] -= int(height / 2)
    pt_2[1] -= int(height / 2)
    #Check if the vector lies in the third or fourth quadrant and modify angle accordingly:
    if pt_2[1] - pt_1[1] < 0:
        dest_vec_angle = 360 - dest_vec_angle

    if dest_vec_angle > orient_angle:
        left = dest_vec_angle - orient_angle
        right = 360 - left
    else:
        right = orient_angle - dest_vec_angle
        left = 360 - right

    # print(f'From left : {left}')
    # print(f'From right: {right}')

    if right < left : return right/180
    else: return -(left/180)

# Calculate the shortest angle between two vectors (in degrees)
def angle_between_vectors(vec_1,vec_2):
    vec_1,vec_2 = np.array(vec_1),np.array(vec_2)
    unit_vec_1 = vec_1 / np.linalg.norm(vec_1)
    unit_vec_2 = vec_2 / np.linalg.norm(vec_2)
    cos_angle = np.dot(unit_vec_1,unit_vec_2)
    angle_radians = np.arccos(cos_angle)
    return math.degrees(angle_radians)


# if __name__ == '__main__':
#     vec_1 = [-34,54]
#     vec_2 = [1,0]
#     print(angle_between_vectors(vec_1,vec_2))