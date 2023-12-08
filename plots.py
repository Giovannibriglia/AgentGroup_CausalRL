import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy.ndimage import gaussian_filter1d


def plot_av_rew_steps(dir_results, algorithms, n_games, n_episodes):
    targetPattern = fr"{dir_results}\*.npy"
    directories = glob.glob(targetPattern)
    # data_names = [os.path.basename(directories[s]) for s in range(len(directories))]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, dpi=500)
    fig.suptitle(f'Grid 5x5 - 1 enemy - Averaged over {n_games} games', fontsize=15)

    for alg in algorithms:
        filename_rewards = [s for s in directories if f'{alg}_rewards' in s]
        filename_steps = [s for s in directories if f'{alg}_steps' in s]

        av_rew = np.zeros(n_episodes)
        av_steps = np.zeros(n_episodes)
        for n_game in range(n_games):
            av_rew = np.sum([av_rew, np.load(filename_rewards[n_game])], axis=0)
            av_steps = np.sum([av_steps, np.load(filename_steps[n_game])], axis=0)

        av_steps = av_steps / n_games
        # av_rewards = av_rew / n_games
        av_rewards = np.cumsum(av_rew, dtype=int)

        x = np.arange(0, len(av_rewards), 1)
        ax1.plot(x, av_rewards, label=f'{alg} = {round(np.mean(av_rew)/n_games, 3)}')
        confidence_interval_rew = np.std(av_rewards)
        ax1.fill_between(x, (av_rewards - confidence_interval_rew), (av_rewards + confidence_interval_rew), alpha=0.2)
        ax1.set_title('Average reward on episode steps')
        ax1.legend(fontsize='x-small')

        ax2.plot(x, gaussian_filter1d(av_steps, 1))
        ax2.set_yscale('log')
        ax2.set_title('Actions needed to complete the episode')
        ax2.set_xlabel('Episode', fontsize=12)

    plt.savefig(f'{dir_results}/Average_comparison_{n_games}Games.pdf')
    plt.show()
