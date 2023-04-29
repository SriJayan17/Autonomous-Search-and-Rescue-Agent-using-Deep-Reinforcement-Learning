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

obstacles = (
    pygame.Rect(330,80,20,650),
    pygame.Rect(1120,20,20,650),

    pygame.Rect(20,305,250,20),
    pygame.Rect(1200,305,280,20),

    pygame.Rect(410,305,640,20),

    pygame.Rect(130,80,20,225),
    pygame.Rect(1340,80,20,225),

    pygame.Rect(720,80,20,225),
    pygame.Rect(720,305,20,345),

    pygame.Rect(410,170,250,20),
    pygame.Rect(800,170,250,20),

    pygame.Rect(410,510,250,20),
    pygame.Rect(800,510,250,20),

    pygame.Rect(80,480,190,20),
    pygame.Rect(1200,480,220,20),

    pygame.Rect(80,650,190,20),
    pygame.Rect(1200,650,220,20),
)

fireFlares = (
    pygame.Rect(40,230,45,45),
    pygame.Rect(500,230,45,45),
    pygame.Rect(500,420,45,45),
    pygame.Rect(880,420,45,45),
    pygame.Rect(150,580,45,45),
    pygame.Rect(1300,580,45,45),
    pygame.Rect(1330,25,45,45),
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

exit_points = (
    pygame.Rect((width-40)/2-50-15,0,100,20),
    pygame.Rect((width-40)/2-50-15,height-50,100,20),
    pygame.Rect(0, (height-40)/2-50,20,100),
    pygame.Rect(width-50, (height-40)/2-50,20,100),
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
victimsRect = pygame.Rect(880,230,50,50)

obstacleGrid = np.zeros((row,cols))
fireGrid = np.zeros((row,cols))

helper_agents = (
    (400,(height)/2),
    (1070,(height)/2),
    (700,(height-80))
)


# Making the borders opaque
obstacleGrid[:30,:] = 1
obstacleGrid[row-30:,:] = 1
obstacleGrid[30:row-30,:30] = 1
obstacleGrid[30:row-30,cols-30:] = 1

# Making the obstacles opaque
for obstacle in obstacles:
    opaque_obstacle(obstacle,obstacleGrid,row-height)

#Making the boundaries opaque:
for boundary in boundaries:
    opaque_obstacle(boundary,obstacleGrid,row-height)

# Marking the fire in fireGrid
for fire in fireFlares:
    opaque_obstacle(fire,fireGrid,row-height)

#Visualising the grid:
# from PIL import Image
# import cv2
# if __name__ == '__main__':
#     print(obstacleGrid.shape)
#     # img = Image.fromarray(obstacleGrid, 'RGB')
#     # img.save('my.png')
#     cv2.imshow('image',obstacleGrid)
#     cv2.waitKey(0) 
#     cv2.destroyAllWindows()
