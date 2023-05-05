import numpy as np
import pygame

# Function to mark the presence of obstacle in the grid:
def opaque_obstacle(obstacle_rect,grid,boundary):
    x = obstacle_rect.left + boundary
    y = obstacle_rect.top + boundary

    height = obstacle_rect.height
    width = obstacle_rect.width

    grid[y:y+height,x:x+width] = 1

width = 1500
height = 750

# Here, extra offset = border = 30 (1500+'30',750+'30')
cols = 1530
row = 780

background = pygame.image.load("Project/Resources/Images/floor.png")
background = pygame.transform.scale(background,(width, height))

desk = pygame.image.load("Project/FrontEnd/Images/desk.png")
desk = pygame.transform.scale(desk, (120,140))
desk_rotated = pygame.transform.rotate(desk, 180)
furniture_sofa = pygame.image.load("Project/FrontEnd/Images/furniture_2.png")
furniture_sofa = pygame.transform.scale(furniture_sofa, (160,120))
furniture_sofa_rotated = pygame.transform.rotate(furniture_sofa, 180)

furniture_table = pygame.image.load("Project/FrontEnd/Images/furniture_3.png")
furniture_table = pygame.transform.scale(furniture_table, (160,120))
furniture_table_rotated = pygame.transform.rotate(furniture_table, 180)

beam_bag = pygame.image.load("Project/FrontEnd/Images/beam_bag.png")
beam_bag = pygame.transform.scale(beam_bag, (100,100))

conference_table = pygame.image.load("Project/FrontEnd/Images/conference_table.png")
conference_table = pygame.transform.scale(conference_table, (200,100))

plant = pygame.image.load("Project/FrontEnd/Images/plant.png")
plant = pygame.transform.scale(plant, (50,50))

game_table = pygame.image.load("Project/FrontEnd/Images/game_table.png")
game_table = pygame.transform.scale(game_table, (150,100))

objects = [
    (furniture_sofa, (60,230)),
    (furniture_sofa_rotated, (60,380)),
    (furniture_sofa, (1250,230)),
    (furniture_sofa_rotated, (1250,380)),
    (furniture_table, (60,40)),
    (furniture_table, (300,40)),
    (furniture_table, (60,550)),
    (furniture_table, (300,550)),
    (desk_rotated, (280, 280)),
    (desk, (1080, 280)),
    (beam_bag, (550,150)),
    (beam_bag, (820,150)),
    (conference_table, (1150,40)),
    (conference_table, (1150,570)),
    (game_table, (520,450)),
    (game_table, (800,450)),
    (plant, (630,30)),
    (plant, (780,30)),
    (plant, (630,660)),
    (plant, (780,660))
]
objects_rect = []

for object in objects:
    rect = object[0].get_rect()
    rect.left = object[1][0]
    rect.top = object[1][1]
    objects_rect.append(rect)

walls = (
    pygame.Rect(80,200,370,20), 
    pygame.Rect(430,220,20,300), 
    pygame.Rect(80,500,370,20), 
    pygame.Rect(1040,200,370,20), 
    pygame.Rect(1040,220,20,300), 
    pygame.Rect(1040,500,370,20), 
    pygame.Rect(500,345,490,20), 
    pygame.Rect(720,80,20,200), 
    pygame.Rect(720,425,20,200), 
)


test_fireFlares = (
    pygame.Rect(30,180,45,45),
    pygame.Rect(480,180,45,45),
    pygame.Rect(480,630,45,45),
    pygame.Rect(850,380,45,45),
    pygame.Rect(1230,670,45,45),
    pygame.Rect(850,270,45,45),
    pygame.Rect(1230,140,45,45),
)

# Width of boundaries = 20
boundaries = (
    pygame.Rect(0,20,20,(height-40)/2-50),
    pygame.Rect(0,(height-40)/2+50,20,(height-40)/2+50),
    pygame.Rect(0,0,(width-40)/2-50,20),
    pygame.Rect((width-40)/2+50,0,(width-40)/2+50,20),
    pygame.Rect(width-20,20,20,(height-40)/2-50),
    pygame.Rect(width-20,(height-40)/2+50,20,(height-40)/2+50),
    pygame.Rect(0,height-20,(width-40)/2-50,20),
    pygame.Rect((width-40)/2+50,height-20,(width-40)/2+50,20),
)

test_exit_points = (
    pygame.Rect((width-40)/2-50-15,0,50,50),
    pygame.Rect((width-40)/2-50-15,height-50,50,50),
    pygame.Rect(0, (height-40)/2-50,50,50),
    pygame.Rect(width-50, (height-40)/2-50,50,50),
)

borders = (
    pygame.Rect(0,0,width,20),
    pygame.Rect(0,20,20,height),
    pygame.Rect(width-20, 20, 20, height-20),
    pygame.Rect(20,height-20,width-40, 20),
)

# Tuple containing the pair of center points where the agents should be placed initially
agents = (
    # (32,(height-40)/2),
    # (1422,(height-40)/2),
    # ((width-40)/2-33,30),
    # ((width-40)/2-33,690)
    (50,(height-40)/2),
    (1440,(height-40)/2),
    ((width-40)/2-15,30),
    ((width-40)/2-15,690),
    # pygame.Rect(50,(height-40)/2,30,30),
    # pygame.Rect(1440,(height-40)/2,30,30),
    # pygame.Rect((width-40)/2-15,30,30,30),
    # pygame.Rect((width-40)/2-15,690,30,30),
)
test_victimsRect = pygame.Rect(705,290,50,50)

test_obstacleGrid = np.zeros((row,cols))
test_fireGrid = np.zeros((row,cols))

helper_agents = (
    (400,(height)/2),
    (1070,(height)/2),
    (700,(height-80))
)


# Making the borders opaque
test_obstacleGrid[:30,:] = 1
test_obstacleGrid[row-30:,:] = 1
test_obstacleGrid[30:row-30,:30] = 1
test_obstacleGrid[30:row-30,cols-30:] = 1

# Making the obstacles opaque
for obstacle in walls:
    opaque_obstacle(obstacle,test_obstacleGrid,row-height)

#Making the boundaries opaque:
for boundary in boundaries:
    opaque_obstacle(boundary,test_obstacleGrid,row-height)

# Making the objects opaque:
# for obj in objects_rect:
#     opaque_obstacle(obj,test_obstacleGrid,row-height)

# Marking the fire in fireGrid
for fire in test_fireFlares:
    opaque_obstacle(fire,test_fireGrid,row-height)

#Visualising the grid:
# from PIL import Image
# import cv2
# if __name__ == '__main__':
#     print(test_obstacleGrid.shape)
#     # img = Image.fromarray(obstacleGrid, 'RGB')
#     # img.save('my.png')
#     cv2.imshow('image',test_fireGrid)
#     cv2.waitKey(0) 
#     cv2.destroyAllWindows()
