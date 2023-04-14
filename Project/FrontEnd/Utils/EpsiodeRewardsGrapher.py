import matplotlib.pyplot as plt

def plot(agents_episode_rewards, rows=1):
    fig, axes = plt.subplots(nrows=rows, figsize=(10,7))
    for (index, episode_rewards) in enumerate(agents_episode_rewards):
        x = []
        y = []
        for (episode, reward) in enumerate(episode_rewards):
            x.append(episode)
            y.append(reward)
        if(rows==1):
            axes.set(title="Agent "+str(index+1), xlabel="Episode Number", ylabel="Reward")
            axes.plot(x,y)
        else:
            axes[index].set(title="Agent "+str(index+1), xlabel="Episode Number", ylabel="Reward")
            axes[index].plot(x,y)
    fig.tight_layout()
    plt.show()
    fig.savefig("./Project/Graphs/episode_rewards.jpg")