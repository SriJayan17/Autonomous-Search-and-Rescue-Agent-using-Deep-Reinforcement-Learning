import numpy as np
import pygame


width = 700
height = 700

row = 710
cols = 710

walls = [
    pygame.Rect((0,0,20,height)),
    pygame.Rect((width-20,0,20,height)),
    pygame.Rect((20,0,width-40,20)),
    pygame.Rect((20,height-20,60,20)),
    pygame.Rect((160,height-20,width-160,20)),
    pygame.Rect((400,20,20,20)),
    pygame.Rect((400,120,20,200)),
    pygame.Rect((400,400,20,280)),
    pygame.Rect((500,300,180,20)),
    pygame.Rect((20,400,300,20)),
    pygame.Rect((420,500,30,20)),
    pygame.Rect((590,500,90,20)),
]

fireFlares = [
    pygame.Rect(50,280,60,60),
    pygame.Rect(140,30,60,60),
    pygame.Rect(580,210,60,60),
    pygame.Rect(430,420,60,60),
    pygame.Rect(590,550,60,60),
]

borders = (
    pygame.Rect(0,0,20,height-20),
    pygame.Rect(0,height-20,width-20,20),
    pygame.Rect(20,0,width-20,20),
    pygame.Rect(width-20,0,20,height),
)

obstacles = [

    # table 1
    pygame.Rect(50,50,90,130),
    # sofa 1
    pygame.Rect(60,480,100,40),
    pygame.Rect(60,520,40,60),
    # sofa 2
    pygame.Rect(310,540,40,60),
    pygame.Rect(250,600,100,40),
    # sofa 3
    pygame.Rect(425,100,180,60),
    # table 2
    pygame.Rect(260,150,110,110),
    # round table
    pygame.Rect(200,480,60,60),
    # fridge
    pygame.Rect(395,325,25,70),
    # bed
    pygame.Rect(590,360,50,95),
]

victimsRect = pygame.Rect(430,600,65,65)

grid = np.zeros((row,cols))

grid[:30,:] = 1
grid[row-30:,:] = 1
grid[30:row-30,:30] = 1
grid[30:row-30,cols-30:] = 1

for obstacle in obstacles:

    x = obstacle.left
    y = obstacle.top

    height = obstacle.height
    width = obstacle.width

    grid[x:x+width,y:y+height] = 1

for fire in fireFlares:

    x = fire.left
    y = fire.top

    height = fire.height
    width = fire.width

    grid[x:x+width, y:y+height] = 2
