import numpy as np

def computeGrid(obstacles,fireFlares):

    rows = 700
    cols = 700

    grid = np.zeros((rows,cols))

    grid[:30,:] = 1
    grid[rows-30:,:] = 1
    grid[30:rows-30,:30] = 1
    grid[30:rows-30,cols-30:] = 1
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
    
    return grid