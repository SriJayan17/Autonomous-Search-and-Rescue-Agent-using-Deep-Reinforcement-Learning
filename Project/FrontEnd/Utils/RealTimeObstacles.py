from turtle import width
import pygame


width = 700
height = 700

walls = [
    pygame.Rect((0,0,20,height)),
    pygame.Rect((width-20,0,20,height)),
    pygame.Rect((20,0,width-40,20)),
    pygame.Rect((20,height-20,60,20)),
    pygame.Rect((160,height-20,width-160,20)),
    pygame.Rect((400,20,20,300)),
    pygame.Rect((400,400,20,280)),
    pygame.Rect((500,300,180,20)),
    pygame.Rect((20,400,300,20)),
    pygame.Rect((420,500,90,20)),
    pygame.Rect((590,500,90,20)),
]

fires = [
    pygame.Rect(200,200,40,40),
    pygame.Rect(600,240,40,40),
    pygame.Rect(430,390,40,40),
]