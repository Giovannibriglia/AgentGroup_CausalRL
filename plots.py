import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy.ndimage import gaussian_filter1d
fontsize = 12

def plot_av_rew_steps(dir_results, algorithms, n_games, n_episodes, rows, cols, n_enemies):
    targetPattern = fr"{dir_results}\*.npy"
    directories = glob.glob(targetPattern)
    # data_names = [os.path.basename(directories[s]) for s in range(len(directories))]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, dpi=500)
    fig.suptitle(f'Grid {rows}x{cols} - {n_enemies} enemy - Averaged over {n_games} games', fontsize=15)

    for alg in algorithms:
        filename_rewards = [s for s in directories if f'{alg}_rewards' in s]
        filename_steps = [s for s in directories if f'{alg}_steps' in s]

        av_rew = np.zeros(n_episodes)
        av_steps = np.zeros(n_episodes)
        for n_game in range(n_games):
            av_rew = np.sum([av_rew, np.load(filename_rewards[n_game])], axis=0)
            av_steps = np.sum([av_steps, np.load(filename_steps[n_game])], axis=0)

        av_steps = av_steps / n_games
        av_rewards = np.cumsum(av_rew, dtype=int)

        x = np.arange(0, len(av_rewards), 1)
        ax1.plot(x, av_rewards, label=f'{alg} = {round(np.mean(av_rew) / n_games, 3)}')
        confidence_interval_rew = np.std(av_rewards)
        ax1.fill_between(x, (av_rewards - confidence_interval_rew), (av_rewards + confidence_interval_rew), alpha=0.2)
        ax1.set_title('Average reward on episode steps')
        ax1.legend(fontsize='xx-small')

        ax2.plot(x, gaussian_filter1d(av_steps, 1))
        ax2.set_yscale('log')
        ax2.set_title('Actions needed to complete the episode')
        ax2.set_xlabel('Episode', fontsize=12)

    plt.savefig(f'{dir_results}/Av_rew_actions_comparison_{n_games}Games.pdf')
    plt.show()


def plot_av_computation_time(dir_results, algorithms, n_games, rows, cols, n_enemies):
    targetPattern = fr"{dir_results}\*.npy"
    directories = glob.glob(targetPattern)
    # data_names = [os.path.basename(directories[s]) for s in range(len(directories))]

    fig = plt.figure(dpi=500)
    fig.suptitle(f'Grid {rows}x{cols} - {n_enemies} enemy - Averaged over {n_games} games', fontsize=fontsize+3)

    data = []
    for alg in algorithms:
        filename_comp_time = [s for s in directories if f'{alg}_computation_time' in s]

        av_comp_time = 0
        data_to_app = []
        for n_game in range(n_games):
            av_comp_time += np.load(filename_comp_time[n_game])
            data_to_app.append(av_comp_time)
        av_comp_time = av_comp_time/n_games
        data.append(data_to_app)

        # plt.bar(alg, av_comp_time, label=f'{alg} = {round(av_comp_time, 3)}')
    plt.boxplot(data, vert=True, patch_artist=True, labels=algorithms)

    # plt.legend(loc='best')
    plt.xticks(rotation=90)
    plt.ylabel('Computation time [min]', fontsize=fontsize)
    plt.subplots_adjust(bottom=0.5)
    plt.grid()
    plt.savefig(f'{dir_results}/Average_comp_time_comparison_{n_games}Games.pdf')
    plt.show()
