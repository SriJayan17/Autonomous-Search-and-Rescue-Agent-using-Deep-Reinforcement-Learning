import numpy as np
import pygame

width = 700
height = 700

row = 710
cols = 710

obstacles = (
    pygame.Rect(20,290,70,20),
    pygame.Rect(20,350,70,20),
    pygame.Rect(20,500,120,20),
    pygame.Rect(140,500,20,80),
    pygame.Rect(80,580,80,20),
    pygame.Rect(230,500,20,180),
    pygame.Rect(500,600,20,80),
    pygame.Rect(590,520,20,80),
    pygame.Rect(410,500,200,20),
    pygame.Rect(400,500,20,80),
    pygame.Rect(330,580,90,20),
    pygame.Rect(310,350,20,250),
    pygame.Rect(90,410,220,20),
    pygame.Rect(150,300,20,110),
    pygame.Rect(90,220,140,20),
    pygame.Rect(70,160,20,80),
    pygame.Rect(210,160,20,80),
    pygame.Rect(70,140,60,20),
    pygame.Rect(210,140,340,20),
    pygame.Rect(130,80,20,80),
    pygame.Rect(270,80,20,80),
    pygame.Rect(130,80,80,20),
    pygame.Rect(550,220,20,130),
    pygame.Rect(570,220,40,20),
    pygame.Rect(610,140,20,100),
    pygame.Rect(360,80,340,20),
    pygame.Rect(480,410,220,20),
    pygame.Rect(400,290,20,140),
    pygame.Rect(250,290,150,20),
    pygame.Rect(230,290,20,80),
    pygame.Rect(480,220,20,200),
    pygame.Rect(300,220,180,20),
    pygame.Rect(615,290,20,120),
)

fireFlares = (
    pygame.Rect(305,70,40,40),
    pygame.Rect(600,240,40,40),
    pygame.Rect(430,390,40,40),
)

borders = (
    pygame.Rect(0,0,20,height-20),
    pygame.Rect(0,height-20,width-20,20),
    pygame.Rect(20,0,width-20,20),
    pygame.Rect(width-20,0,20,height),
)

boundaries = (
    pygame.Rect(0, 0, 50, 20),
    pygame.Rect(120, 0, width-120, 20),
    pygame.Rect(0, 0, 20, height),
    pygame.Rect(0, height-20, width, 20),
    pygame.Rect(width-20, 0, 20, 410),
    pygame.Rect(width-20, 500, 20, 200)
)

victimsRect = pygame.Rect(420,250,50,50)

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