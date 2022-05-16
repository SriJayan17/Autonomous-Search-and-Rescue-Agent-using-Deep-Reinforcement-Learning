import math
import numpy as np

class StateHandler:

    # Parameters
    theta = 90
    step = 10
    proportionalityConstant = 100

    def __init__(self, grid, fireFlares, victims):
        self.grid = grid
        self.victimsCenter = victims.center
        self.fireFlares = fireFlares
        

    # Calculate eucledian distance between two points
    def calculateDistance(self, a, b):
        return math.sqrt(math.pow((a[0]-b[0]), 2) + math.pow((a[1]-b[1]), 2))
    
    # To calculate Obstacle density
    def calculateObstacleDenstity(self, agent, state):
        
        agentRect = agent.get_rect(topleft = (state[0],state[1]))
        
        topLeft =  agentRect.topleft
        topRight = agentRect.topright
        bottomLeft = agentRect.bottomleft
        bottomRight = agentRect.bottomright

        if state[2] in [0, 360]:

            front = self.grid[topLeft[0]:topLeft[0]+30, topLeft[1]-30:topLeft[1]]
            left = self.grid[topLeft[0]-30:topLeft[0], topLeft[1]:topLeft[1]+30]
            right = self.grid[topRight[0]:topRight[0]+30, topRight[1]:topRight[1]+30]

        elif state[2] == 90:
            
            front = self.grid[bottomLeft[0]-30:bottomLeft[0], topLeft[1]:topLeft[1]+30]
            left = self.grid[bottomLeft[0]:bottomLeft[0]+30, bottomLeft[1]:bottomLeft[1]+30]
            right = self.grid[topLeft[0]:topLeft[0]+30, topLeft[1]-30:topLeft[1]]

        elif state[2] == 180:
            
            front = self.grid[bottomLeft[0]:bottomLeft[0]+30, bottomLeft[1]:bottomLeft[1]+30]
            left = self.grid[bottomRight[0]:bottomRight[0]+30, bottomRight[1]-30:bottomRight[1]]
            right = self.grid[bottomLeft[0]-30:bottomLeft[0], bottomLeft[1]-30:bottomLeft[1]]


        elif state[2] == 270:

            front = self.grid[topRight[0]:topRight[0]+30, topLeft[1]:topLeft[1]+30]
            left = self.grid[topRight[0]-30:topRight[0], topRight[1]-30:topRight[1]]
            right = self.grid[bottomRight[0]-30:bottomRight[0], bottomRight[1]:bottomRight[1]+30]

        frontDensity = (np.sum(front) - (2 * np.count_nonzero(front == 2))) / (30*30)
        leftDensity = (np.sum(left) - (2 * np.count_nonzero(front == 2))) / (30*30)
        rightDensity = (np.sum(right) - (2 * np.count_nonzero(front == 2))) / (30*30)

        return frontDensity,leftDensity,rightDensity

    # Calculate angle between agent and victims vectors
    def calculateAngle(self, agent,state):

        agentRect = agent.get_rect(topleft=(state[0],state[1]))
        vectorA = np.array([self.victimsCenter[0] - agentRect.center[0], self.victimsCenter[1] - agentRect.center[1]])

        if state[2] in [0,360]:
            frontMid = ((agentRect.topleft[0]+agentRect.topright[0])/2,(agentRect.topleft[1]+agentRect.topright[1])/2)
        elif state[2] in [90,-270]:
            frontMid = ((agentRect.topleft[0]+agentRect.bottomleft[0])/2,(agentRect.topleft[1]+agentRect.bottomleft[1])/2)
        elif state[2] in [180,-180]:
            frontMid = ((agentRect.bottomleft[0]+agentRect.bottomright[0])/2,(agentRect.bottomleft[1]+agentRect.bottomright[1])/2)
        elif state[2] in [270,-90]:
            frontMid = ((agentRect.topright[0]+agentRect.bottomright[0])/2,(agentRect.topright[1]+agentRect.bottomright[1])/2)

        vectorB = np.array([frontMid[0]-agentRect.center[0],frontMid[1]-agentRect.center[1]])

        angle = math.degrees(np.arccos((np.dot(vectorB,vectorA))/(math.sqrt(vectorA[0]**2 + vectorA[1]**2)*math.sqrt(vectorB[0]**2 + vectorB[1]**2))))
        # print(f' The angle is {angle}')
        return angle

    # Caculate heat intesity from the heat source
    def calculateHeatIntensity(self, agent, state):
        
        findIntensity = lambda d : StateHandler.proportionalityConstant if d==0 else StateHandler.proportionalityConstant / math.pow(d,2)

        agentRect = agent.get_rect(topleft = (state[0], state[1]))

        if state[2] in [0, 360]:
            front = ((agentRect.topleft[0] + agentRect.topright[0])/2, (agentRect.topleft[1]+agentRect.topright[1])/2)
            left = ((agentRect.topleft[0] + agentRect.bottomleft[0])/2, (agentRect.topleft[1]+agentRect.bottomleft[1])/2)
            right = ((agentRect.topright[0] + agentRect.bottomright[0])/2, (agentRect.topright[1] +agentRect.bottomright[1])/2)

        elif state[2] in [90, -270]:
            front = ((agentRect.topleft[0] + agentRect.bottomleft[0])/2, (agentRect.topleft[1]+agentRect.bottomleft[1])/2)
            left = ((agentRect.bottomleft[0] + agentRect.bottomright[0])/2, (agentRect.bottomleft[1]+agentRect.bottomright[1])/2)
            right = ((agentRect.topleft[0] + agentRect.topright[0])/2, (agentRect.topleft[1] +agentRect.topright[1])/2)

        elif state[2] in [180, -180]:
            front = ((agentRect.bottomleft[0] + agentRect.bottomright[0])/2, (agentRect.bottomleft[1]+agentRect.bottomright[1])/2)
            left = ((agentRect.topright[0] + agentRect.bottomright[0])/2, (agentRect.bottomright[1] +agentRect.topright[1])/2)
            right = ((agentRect.topleft[0] + agentRect.bottomleft[0])/2, (agentRect.topleft[1]+agentRect.bottomleft[1])/2)
        
        elif state[2] in [270, -90]:
            front = ((agentRect.topright[0] + agentRect.bottomright[0])/2, (agentRect.topright[1]+agentRect.bottomright[1])/2)
            left = ((agentRect.topleft[0] + agentRect.topright[0])/2, (agentRect.topleft[1]+agentRect.topright[1])/2)
            right = ((agentRect.bottomleft[0] + agentRect.bottomright[0])/2, (agentRect.bottomleft[1] +agentRect.bottomright[1])/2)

        frontIntensity = 0
        leftIntensity = 0
        rightIntensity = 0

        for fire in self.fireFlares:

            frontIntensity += findIntensity(self.calculateDistance(fire.center, front))
            leftIntensity += findIntensity(self.calculateDistance(fire.center, left))
            rightIntensity += findIntensity(self.calculateDistance(fire.center, right))
        
        return frontIntensity,leftIntensity,rightIntensity
        

    # Predict next state based on current state and action:
    def predictNextState(self, agent, state, action):

        nextState = list(state)

        if action == 0:
            if state[2] > 0 and state[2] < 90:
                nextState[1] = state[1] - StateHandler.step
            elif state[2] >= 90 and state[2] < 180:
                nextState[0] = state[0] - StateHandler.step

            elif state[2] >= 180 and state[2] < 270:
                nextState[1] = state[1] + StateHandler.step
            
            elif state[2] >= 270 and state[2] < 360:
                nextState[0] = state[0] + StateHandler.step

            elif state[2] < 0 and state[2] >= -90:
                nextState[0] = state[0] + StateHandler.step

            elif state[2] < -90 and state[2] >= -180:
                nextState[1] = state[1] + StateHandler.step

            elif state[2] < -180 and state[2] >= -270:
                nextState[0] = state[0] - StateHandler.step
            
            elif state[2] < -270 and state[2] >= -360:
                nextState[1] = state[1] - StateHandler.step
            
            else:
                nextState[1] = state[1] - StateHandler.step

        elif action == 1:
            angle = 0
            if state[2] + StateHandler.theta < 360:
                angle = state[2] + StateHandler.theta
            nextState[2] = (angle if angle >= 0 else 360 + angle)
            nextState = list(self.predictNextState(agent,tuple(nextState),0))

        elif action == 2:
            angle = 0
            if state[2] - StateHandler.theta > -360:
                angle = state[2] - StateHandler.theta
            nextState[2] = (angle if angle >= 0 else 360 + angle) 
            nextState = list(self.predictNextState(agent,tuple(nextState),0))
            
        
        frontDensity,leftDensity,rightDensity = self.calculateObstacleDenstity(agent, tuple(nextState))
        frontIntensity, leftIntensity, rightIntensity = self.calculateHeatIntensity(agent, tuple(nextState))
        destinationAngle = self.calculateAngle(agent, tuple(nextState))

        if tuple(nextState) != state:
            nextState[3] = frontDensity
            nextState[4] = leftDensity
            nextState[5] = rightDensity
            nextState[6] = frontIntensity
            nextState[7] = leftIntensity
            nextState[8] = rightIntensity
            nextState[9] = destinationAngle

        return tuple(nextState)
