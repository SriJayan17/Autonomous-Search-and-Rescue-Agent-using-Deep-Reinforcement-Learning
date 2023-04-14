import matplotlib.pyplot as plt

def plot(agents_episode_rewards):
    fig, axes = plt.subplots(nrows=4, figsize=(10,7))
    for (index, episode_rewards) in enumerate(agents_episode_rewards):
        x = []
        y = []
        for (episode, reward) in enumerate(episode_rewards):
            x.append(episode)
            y.append(reward)
        axes[index].set(title="Agent "+str(index+1), xlabel="Episode Number", ylabel="Reward")
        axes[index].plot(x,y)
    fig.tight_layout()
    plt.show()
    fig.savefig("./Project/Graphs/episode_rewards.jpg")