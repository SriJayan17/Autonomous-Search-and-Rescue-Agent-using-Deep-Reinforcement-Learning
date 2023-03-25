import numpy as np
import pygame

width = 1500
height = 750

cols = 1510
row = 760

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

borders = (
    pygame.Rect(0,20,20,height-40),
    pygame.Rect(0,0,width,20),
    pygame.Rect(width-20,20,20,height-40),
    pygame.Rect(0,height-20,width,20),
)

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

agents = (
    pygame.Rect(30,(height-40)/2,30,30),
    pygame.Rect(1440,(height-40)/2,30,30),
    pygame.Rect((width-40)/2-15,30,30,30),
    pygame.Rect((width-40)/2-15,690,30,30),
)
victimsRect = pygame.Rect(880,230,50,50)

obstacleGrid = np.zeros((row,cols))
fireGrid = np.zeros((row,cols))

obstacleGrid[:30,:] = 1
obstacleGrid[row-30:,:] = 1
obstacleGrid[30:row-30,:30] = 1
obstacleGrid[30:row-30,cols-30:] = 1

for obstacle in obstacles:

    x = obstacle.left
    y = obstacle.top

    height = obstacle.height
    width = obstacle.width

    obstacleGrid[x:x+width,y:y+height] = 1

for fire in fireFlares:

    x = fire.left
    y = fire.top

    height = fire.height
    width = fire.width

    fireGrid[x:x+width, y:y+height] = 2
