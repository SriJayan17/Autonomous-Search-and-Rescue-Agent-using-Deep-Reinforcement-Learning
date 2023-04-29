import math
import numpy as np
from tkinter import *

from Project.FrontEnd.Utils.Training_Env_Obstacles import *
from Project.FrontEnd.Utils.Training_Env_Obstacles import height,width
from Project.FrontEnd.Utils.Rewards import *
from Project.Backend.Agent import Agent
#To check if two rectangles are colliding
def isColliding(objects, rect):
    for object in objects:
        if object.colliderect(rect):
            # print("collision")
            return True
    
    return False

# To check if a particular move causes an agent to collide with borders, obstacles, boundaries,
# or other agents
def isPermissible(agent_list=[], index=0, include_borders = True):
    objects = [boundaries, obstacles]

    if include_borders:
        objects.append(borders)
    
    objects.append((agent_list[i].rect for i in range(len(agent_list)) if i != index))
    current_rect = agent_list[index].rect
    current_rect_center = current_rect.center

    #Checking if the agent has collided with an obstacle
    for object in objects:
        if(isColliding(object, current_rect)):
            return False

    # Checking if the agent has hit the borders    
    # if (current_rect_center[0] < 0 or current_rect_center[0] > width) or \
    #    (current_rect_center[1] < 0 or current_rect_center[1] > height):
    #     return False
    
    return True

def generateRescueReward(previous_center, current_rect):
    reward = 0
    current_center = current_rect.center

    if current_center == previous_center:
        reward = AWAY_FROM_DESTINATION

    prev_min_dist = 1e6
    prev_target = None
    for exit_pt in exit_points:
        prev_dist = eucledianDist(previous_center, exit_pt)
        if prev_dist < prev_min_dist: 
            prev_min_dist = prev_dist
            prev_target = exit_pt
    current_dist = eucledianDist(current_center, prev_target)
        # if current_dist < current_min_dist: current_min_dist = current_dist
    if prev_min_dist > current_dist:
        # print(f'prev_min: {prev_min_dist}, current_dist: {current_dist}')
        reward = TOWARDS_DESTINATION
    else:
        reward = AWAY_FROM_DESTINATION

    if reachedExit(current_rect):
        return REACHED_VICTIMS
    
    for fire in fireFlares:
        if(fire.colliderect(current_rect)):
            reward = HIT_FIRE

    return reward

def generateReward(previous_center, current_rect, rescue_op=False, nearest_exit:pygame.Rect=None):
    reward = 0
    current_center = current_rect.center
    target_pt = nearest_exit.center if rescue_op else victimsRect.center

    previous_target_dist = eucledianDist(previous_center, target_pt)
    current_target_dist = eucledianDist(current_center, target_pt)
    
    reward = IMPERMISSIBLE_ACTION

    # Highly positive reward if the agent has reached the target:
    if reachedVictims(current_rect):
        return REACHED_VICTIMS

    # Positive reward if the agent has moved towards the target
    if previous_target_dist > current_target_dist:
        reward = TOWARDS_DESTINATION
    
    # Negative reward if the agent has flown into fire
    for fire in fireFlares:
        if(fire.colliderect(current_rect)):
            reward = HIT_FIRE
    
    #Check if the agent is near borders:
    if current_rect.centerx < 30 or (width - current_rect.centerx) < 30:
        reward = NEAR_BORDERS
    if current_rect.centery  < 30 or (height - current_rect.centery) < 30:
        reward = NEAR_BORDERS

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
# def prepare_agent_state(agent_list,target_index,state_dict,
#                         initial_state_dict,flock_center=None,vicinity_radius=600):
#     result = []
#     n = len(agent_list)
#     for i in range(n):
#         if i == target_index:
#             result.extend(state_dict[i])
#         else:
#             target_agent = agent_list[target_index]
#             distance = eucledianDist((target_agent.rect.x, target_agent.rect.y),
#                                      (agent_list[i].rect.x,agent_list[i].rect.y))
#             if distance <= vicinity_radius:
#                 result.extend(state_dict[i])
#             else:
#                 result.extend(initial_state_dict[i])

#     # Adding the flock center
#     if flock_center is not None: result.extend(flock_center)
#     return result

def move_pt(target_pt:list,angle,dist):
    angle += 90
    rad = math.radians(angle)
    target_pt[0] += dist * math.cos(rad)
    target_pt[1] -= dist * math.sin(rad)

def get_sensor_output(x,y,dim,grid,boundary):
    row_num = y + boundary
    col_num = x + boundary
    return int(np.sum(grid[int(row_num-dim):int(row_num+dim),int(col_num-dim):int(col_num+dim)]))/float((dim*2)**2)
    
def get_sensors(target_player:Agent,target_grid,dim,boundary):
    result = []
    # pt_tray = []
    # For left sensor:
    pt = list(target_player.rect.center)
    if target_player.angle >=0:
        target_angle = (target_player.angle + 40) % 360
    else:
        target_angle = target_player.angle + 40
    # dist = height/2 + 5(extra)
    # Angle modificatin prior to movement is done within the function
    move_pt(pt,target_angle,35)
    # pt_tray.append(pt)
    result.append(get_sensor_output(pt[0],pt[1],dim,target_grid,boundary))
    #For forward sensor:
    pt = list(target_player.rect.center)
    move_pt(pt,target_player.angle,35)
    # pt_tray.append(pt)
    result.append(get_sensor_output(pt[0],pt[1],dim,target_grid,boundary))
    # For right sensor:
    pt = list(target_player.rect.center)
    if target_player.angle <= 0:
        target_angle = -1 * ((abs(target_player.angle) + 40) % 360)
    else:
        target_angle = target_player.angle - 40
    move_pt(pt,target_angle,35)
    # pt_tray.append(pt)
    result.append(get_sensor_output(pt[0],pt[1],dim,target_grid,boundary))

    return result

# Generate state for an individual agent
def get_state(agent,extra_info, destination = victimsRect):
    """
    For search training, state_len = 8
    For rescue training, state_len = 8 
    """
    state_vec = []

    # add obstacle_density
    state_vec.extend(get_sensors(agent,obstacleGrid,extra_info['intensity_area_dim'],
                                row-height))
    
    # add heat_intensity
    state_vec.extend(get_sensors(agent,fireGrid,extra_info['intensity_area_dim'],
                                 row-height))
    #Distance between agent and target:
    # print(type(destination))
    # if type(destination) is tuple:
    #     # distances = []
    #     min_dist = 1e6
    #     target_destination = None
    #     deviations = []
    #     for item in destination:
    #         dist = eucledianDist(agent.rect.center,item.center)
    #         if dist < min_dist:
    #             min_dist = dist
    #             target_destination = item
        

    #     deviation_angle = calc_deviation_angle(agent, target_destination)
    #     state_vec.append(deviation_angle)
    #     state_vec.append(-deviation_angle)
        
    # else:
        # state_vec.append(eucledianDist(agent.rect.center,destination.center) / extra_info['max_distance'])
        #Deviation_angle:
    deviation_angle = calc_deviation_angle(agent, destination)
    state_vec.append(deviation_angle)
    state_vec.append(-deviation_angle)
    
    return state_vec

def reachedVictims(agent_rect):
    return victimsRect.colliderect(agent_rect)

def reachedExit(agent_rect):
    for exit in exit_points:
        if exit.colliderect(agent_rect):
            print("exit")
            return True
    
    return False

# Function to calculate the angle of deviation of the agent wrt the destination
def calc_deviation_angle(agent, destination:pygame.Rect):
    pt_1 = list(agent.rect.center)
    pt_2 = list(destination.center)
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


def displayPrompt(prompt):
    root = Tk()
    root.withdraw()
    popup = Toplevel()
    popup.title("Prompt")
    msg = Label(popup,text=prompt)
    popup.geometry("150x40+655+300")
    msg.pack()
    root.after(1000,lambda:root.destroy())
    popup.mainloop()

# if __name__ == '__main__':
#     vec_1 = [-34,54]
#     vec_2 = [1,0]
#     print(angle_between_vectors(vec_1,vec_2))