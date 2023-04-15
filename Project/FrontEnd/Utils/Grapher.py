import matplotlib.pyplot as plt

def plot_rewards(agents_episode_rewards):
    fig, axes = plt.subplots(2,2,figsize=(10,10))
    col = row = 0
    for (index, episode_rewards) in enumerate(agents_episode_rewards):
        x = []
        y = []
        for (episode, reward) in enumerate(episode_rewards):
            x.append(episode+1)
            y.append(reward)
        
        axes[row,col].plot(x,y)
        axes[row,col].set(title=f'Agent {index+1}',xlabel='Episode number', ylabel='Reward')
        
        col += 1
        if col == 2:
            row += 1
            col = 0

    fig.tight_layout()
    plt.show()
    fig.savefig("Graphs/episode_rewards.jpg")

def plot_reach_time(time_list,title):
    x = list(range(1,len(time_list)+1))
    plt.plot(x,time_list)
    plt.xlabel('Episode')
    plt.ylabel('Timesteps')
    plt.title(title)
    plt.show()
    plt.savefig('Graphs/reach_time.jpg')