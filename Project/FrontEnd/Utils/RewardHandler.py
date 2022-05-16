from Project.FrontEnd.Utils.StateHandler import StateHandler
from tkinter import *

class RewardHandler:

    previousActions = [1,1,1]
    def __init__(self, grid, obstacles, fireFlares, borders, victims):

        self.obstacles = obstacles
        self.fireFlares = fireFlares
        self.borders = borders
        self.victims = victims
        self.victimsCenter = victims.center
        self.stateHandler = StateHandler(grid, fireFlares, victims)

    # Reward generation based on action performed
    def generateReward(self, agent, currentState, action):
        
        # Restrict the agent from performing contiinuous turn actions
        if len(RewardHandler.previousActions) >=3:
            RewardHandler.previousActions.pop(0)
            RewardHandler.previousActions.append(action)
            if RewardHandler.previousActions.count(action) == 3 and action in [1,2]:
                return -1,currentState

        # Calculating Agents's distance from destination
        currentAgentRect = agent.get_rect(topleft = (currentState[0],currentState[1]))
        currentAgentCenter = currentAgentRect.center
        currDist = self.stateHandler.calculateDistance(self.victimsCenter, currentAgentCenter)

        # Predicting next state of agent 
        nextState = self.stateHandler.predictNextState(agent, currentState, action)

        # Calculating Agent's updated distance from destination
        agentRect = agent.get_rect(topleft = (nextState[0],nextState[1]))
        agentCenter = agentRect.center
        updatedDist = self.stateHandler.calculateDistance(self.victimsCenter, agentCenter)
        
        # Stop if agent reaches Destination
        if agentRect.colliderect(self.victims):
            print("Reached Destination")
            return 2,nextState

        # Negative reward if agent hits the Boundaries
        for border in self.borders:
            if agentRect.colliderect(border):
                return -0.8,currentState

        # Negative reward if agent hits the obstacles
        for obstacle in self.obstacles:
            if agentRect.colliderect(obstacle):
                return -0.8,currentState
        
        # Higher Negative reward if agent approaches to fire flares
        for fire in self.fireFlares:
            if agentRect.colliderect(fire):
                root = Tk()
                root.withdraw()
                popup = Toplevel()
                popup.title("Prompt")
                msg = Label(popup,text="Agent is approaching fire")
                popup.geometry("150x50+680+380")
                msg.pack()
                root.after(300,lambda:root.destroy())
                popup.mainloop()
                return -1.5,nextState

        # Negative reward if agent moves away from destination (victims)
        if currDist <= updatedDist:
            return -0.7,nextState

        # Positive Reward if agent approaches near to victims
        if currDist > updatedDist:
            return 0.8,nextState        
        
