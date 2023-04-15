import sys
sys.path.append("e:/AI_projects/RescueAI/")

import random
import pygame
import os

from Project.Backend.Brains.TD3.Memory import ReplayBuffer
from Project.FrontEnd.Utils.Grapher import *
from Project.FrontEnd.Utils.Training_Env_Obstacles import *
from Project.Backend.Agent import Agent
from Project.FrontEnd.Utils.Action_Handler import *

class TrainingEnvironment:

    reachedVictims = False
    stopSimulation = False

    def __init__(self):

        # Environment Dimension
        self.height = 750
        self.width = 1500

        self.state_len = 14
        self.agentModel = Agent(self.state_len,
                                1,
                                15,
                                (victimsRect.x, victimsRect.y),
                                memory = 1e6,
                                load = False,
                                load_path = 'saved_models/rescue_agent'
                                )

        self.base_velocity = 3.0

        self.state_extra_info = {
            'scale_x' : self.width,
            'scale_y' : self.height,
            'max_distance' : math.sqrt((self.width)**2 + (self.height)**2),
            'intensity_area_dim' : 10,
        }
        self.memory = ReplayBuffer()
        self.agent_episode_reward = [[]]
        
        # Initialising the agents
        # Action limit:
        #    Angle -> Varies from -15 to 15
        #    Velocity -> Varies from (base_velocity - 2.5 = 5) to (base_velocity + 2.5 = 10)

        #This is to store the current state of an individual agent   
        self.state_dict = None
        # This is to store the actual state(combined) that we pass into the neural network
        # when taking decision, this is needed to create the records to train the agent. 

        self.state_dict = get_state(self.agentModel,self.state_extra_info, destination=exit_points)
        
        # This is to store the actions taken by the agents:
        self.action_dict = None
        self.action_permit = True

        self.agentRewards = 0
        self.episode_rewards = 0

        self.victims = pygame.image.load("Project/Resources/Images/rescue_victims.png")
        self.victims = pygame.transform.scale(self.victims,(20,20))

        self.fire = pygame.image.load("Project/Resources/Images/fire.png")
        self.fire = pygame.transform.scale(self.fire,(fireFlares[0].width,fireFlares[0].height))

        self.exit = pygame.image.load("Project/Resources/Images/exit.jpg")
        self.exit = pygame.transform.scale(self.exit,(30, 30))

        self.prepare_dirs()

        # self.flock_center = calc_flock_center(self.agentModels)

    #Preparing the directories required to store the outputs:
    def prepare_dirs(self):
        cwd = os.getcwd()
        if not os.path.isdir(os.path.join(cwd,"saved_models")):
            os.makedirs(f'saved_models/rescue_agent')

    def stop(self):
        self.stopSimulation = True
    
    def perform_action(self, agent, turn_angle,dist):
        if turn_angle != 0: agent.turn(turn_angle)
        agent.move(self.base_velocity + dist)
        if not isPermissible([agent]):
            agent.restore_move()
            # print('Action not permitted!')
            # agent.turn(-turn_angle)
            # self.agentRewards[index] = IMPERMISSIBLE_ACTION
            self.action_permit = False


    def run(self):

        pygame.init()
        pygame.display.set_caption("Search and Rescue Simulation")
        environment = pygame.display.set_mode((self.width, self.height))

        episode_timesteps = 0
        total_timesteps = 0
        random_action_limit = 100
        episode_len = 1_000
        expl_noise = [3,5]
        expl_prob = 0.2
        deter_count = 0
        # timesteps_since_eval = 0
        episode_num = 1
        done = False 
        
        
        while not self.stopSimulation:

            environment.fill((0,0,0))

            for boundary in boundaries:
                pygame.draw.rect(environment, (255, 0, 0), boundary)

            for obstacle in obstacles:
                pygame.draw.rect(environment,(255, 0, 0), obstacle)

            for exit in exit_points:
                environment.blit(self.exit, exit.center)
            
            # environment.blit(self.victims, (victimsRect.x, victimsRect.y))

            for fire in fireFlares:
                environment.blit(self.fire, (fire.x, fire.y))
            
            # An episode is over
            if done:
                # If we are not at the very beginning, we start the training process of the model
                if total_timesteps != 0:
                    self.agent_episode_reward[0].append(self.episode_rewards)
                    print(f'Total Timesteps: {total_timesteps} Episode Num: {episode_num}')
                    print('Total reward obtained by the agents:')                    
                    print(f'Agent: {self.episode_rewards}')
                    self.episode_rewards = 0

                    displayPrompt("Training Agent")
                    print(f'Training Agent')
                    self.agentModel.train(memory=self.memory,iterations=episode_timesteps,batch_size=500)
                    self.agentModel.save_brain(f'./saved_models/rescue_agent')
                    # agent.train(replay_buffer, episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip, policy_freq)

            # We evaluate the episode and we save the policy
            # if timesteps_since_eval >= eval_freq:
            #     timesteps_since_eval %= eval_freq
            #     evaluations.append(evaluate_policy(policy))
            #     np.save("./results/%s" % (file_name), evaluations)
        
                # When the training step is done, we reset the state of the environment
                self.agentModel.rect.center = victimsRect.center
                environment.blit(self.agentModel.shape_copy,self.agentModel.rect)    
                # obs = env.reset()
        
                # Set the Done to False
                done = False
                
                # Set rewards and episode timesteps to zero
                episode_timesteps = 0
                episode_num += 1    

            # Last run of the episode.
            if episode_timesteps + 1 == episode_len:
                done = True
            # To check if any of the agents reached the target:
            if_reached = False
            # Automated Navigation
            # Take random action in the initial 10,000 timesteps
            if total_timesteps < random_action_limit:
                action = self.agentModel.take_random_action()
                # action = self.getManualAction()
            else:
            # Take action using neural network
                action = self.agentModel.take_action(self.state_dict)
                # action = self.getManualAction()
                if expl_noise != 0: # Adding noise to the predicted action
                    action = (action + random.uniform(expl_noise[0], expl_noise[1])).clip(-15,15) 
                    # Clipping the final action between the permissible range of values
            # print(action)
            self.perform_action(self.agentModel, action[0], 12)
            self.action_dict = action

            if not self.action_permit: # Action not permitted
                self.agentReward = IMPERMISSIBLE_ACTION
                self.action_permit = True
            else: #Action was permitted
                self.agentRewards = generateReward(self.agentModel.prev_center, self.agentModel.rect)                

            #Adding to the total episode_reward received by a single agent:
            self.episode_rewards += self.agentRewards

            # Update the current state of the individual agent
            prev_state = self.state_dict
            self.state_dict = get_state(self.agentModel,self.state_extra_info, destination=exit_points)       

            if_reached = reachedExit(self.agentModel.rect)


            # Add the record in the memory of the agent's brain:
            self.memory.add((prev_state,self.state_dict,self.action_dict,self.agentRewards,if_reached))

            # Update the state in both the cases (Move permitted/not), because the orientation of the rectange might have changed:
            environment.blit(self.agentModel.shape_copy,self.agentModel.rect)   
            environment.blit(self.victims, (self.agentModel.rect.x+10, self.agentModel.rect.y+10))
            
            # An episode is done if the timelimeet has been reached or if any of the agents
            # has reached the target    
            done = done or if_reached

            episode_timesteps += 1
            total_timesteps += 1
            
            # Manual Control:
            for event in pygame.event.get():  

                if event.type == pygame.QUIT:  
                    if episode_num >= 3:
                        plot_rewards(self.agent_episode_reward)
                        # plot_reach_time()
                        
                    self.stop()
                
                if event.type == pygame.KEYDOWN:
                    
                    if event.key == pygame.K_UP:
                        self.perform_action(self.agentModels, 0, 0, 10)

                    if event.key == pygame.K_LEFT:
                        self.perform_action(self.agentModels, 0, -15, 10)

                    if event.key == pygame.K_RIGHT:
                        self.perform_action(self.agentModels, 0, 15, 10)
            
            # # Manual Control
            # environment.blit(self.agentModels[0].shape_copy,self.agentModels[0].rect)
            # if(reachedVictims(self.agentModels[0])):
            #     self.stop()
            
            # total_timesteps += 1
            if total_timesteps % 1000 == 0: print(f'Timsesteps: {total_timesteps}')
            pygame.display.flip()
            # pygame.time.delay(20)


obj = TrainingEnvironment()
obj.run()
