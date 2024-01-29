import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy.ndimage import gaussian_filter1d

fontsize = 12


def is_number(element):
    return isinstance(element.item(), (int, float))


"""
def plot_av_rew_steps(dir_results, algorithms, n_games, n_episodes, rows, cols, n_enemies):
    targetPattern = fr"{dir_results}\*.npy"
    directories = glob.glob(targetPattern)
    # data_names = [os.path.basename(directories[s]) for s in range(len(directories))]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, dpi=500)
    fig.suptitle(f'Grid {rows}x{cols} - {n_enemies} enemy - Averaged over {n_games} games', fontsize=fontsize + 3)

    algorithms_checked = []
    for alg in algorithms:
        if all(is_number(np.load(f'{dir_results}/{alg}_computation_time_game{game}.npy')) for game in range(1, n_games + 1, 1)):
            algorithms_checked.append(alg)
        else:
            print(f'{alg} timeout occurred')

    for alg in algorithms_checked:
        filename_rewards = [s for s in directories if f'{alg}_rewards' in s]
        filename_steps = [s for s in directories if f'{alg}_steps' in s]

        av_rew = np.zeros(n_episodes)
        av_steps = np.zeros(n_episodes)

        for n_game in range(n_games):
            av_rew = np.sum([av_rew, np.load(filename_rewards[n_game])], axis=0)
            av_steps = np.sum([av_steps, np.load(filename_steps[n_game])], axis=0)

        av_rew = av_rew[:n_episodes]
        av_steps = av_steps[:n_episodes]

        av_steps = av_steps / n_games
        av_rewards = np.cumsum(av_rew, dtype=int)

        x = np.arange(0, len(av_rewards), 1)
        ax1.plot(x, av_rewards,
                 label=f'{alg} = {round(np.mean(av_rew) / n_games, 2)} \u00B1 {round(np.std(av_rew) / n_games, 2)}')
        confidence_interval_rew = np.std(av_rewards)
        ax1.fill_between(x, (av_rewards - confidence_interval_rew), (av_rewards + confidence_interval_rew), alpha=0.2)
        ax1.set_title('Cumulative average reward')
        ax1.legend(fontsize='xx-small')

        av_steps_gauss = gaussian_filter1d(av_steps, 4)
        ax2.plot(x, gaussian_filter1d(av_steps, 8))
        confidence_interval_steps = np.std(av_steps_gauss)
        ax2.fill_between(x, (np.min(av_steps_gauss)), (np.max(av_steps_gauss)),
                         alpha=0.2)
        ax2.set_yscale('log')
        ax2.set_title('Actions needed to complete the episode')
        ax2.set_xlabel('Episode', fontsize=12)

    # plt.savefig(f'{dir_results}/Av_rew_actions_comparison_{n_games}Games.pdf')
    plt.show()
"""


def plot_cumulative_average_rewards(dir_results, algorithms, n_games, n_episodes, rows, cols, n_enemies, dir_saving, kind_of_comparison):
    target_pattern = fr"{dir_results}\*.npy"
    directories = glob.glob(target_pattern)

    fig, ax1 = plt.subplots(dpi=1000)
    fig.suptitle(f'Grid {rows}x{cols} - {n_enemies} enemies - Averaged over {n_games} games', fontsize=fontsize + 3)

    algorithms_checked = []
    for alg in algorithms:
        if all(is_number(np.load(f'{dir_results}/{alg}_computation_time_game{game}.npy')) for game in
               range(1, n_games + 1, 1)):
            algorithms_checked.append(alg)

    for alg in algorithms_checked:
        filename_rewards = [s for s in directories if f'{alg}_rewards' in s]

        av_rew = np.zeros(n_episodes)

        for n_game in range(n_games):
            av_rew = np.sum([av_rew, np.load(filename_rewards[n_game])], axis=0)

        av_rew = av_rew[:n_episodes]
        av_rewards = np.cumsum(av_rew, dtype=int)

        x = np.arange(0, len(av_rewards), 1)
        ax1.plot(x, av_rewards,
                 label=f'{alg} = {round(np.mean(av_rew) / n_games, 2)} \u00B1 {round(np.std(av_rew) / n_games, 2)}')
        confidence_interval_rew = np.std(av_rewards)
        ax1.fill_between(x, (av_rewards - confidence_interval_rew), (av_rewards + confidence_interval_rew), alpha=0.2)
        ax1.set_title('Cumulative average reward')
        ax1.legend(fontsize='x-small')
    plt.savefig(f'{dir_saving}/cumulative_average_reward_comparison_{kind_of_comparison}.pdf')
    # plt.show()


def plot_average_rewards_episode(dir_results, algorithms, n_games, n_episodes, rows, cols, n_enemies, dir_saving, kind_of_comparison):
    target_pattern = fr"{dir_results}\*.npy"
    directories = glob.glob(target_pattern)

    fig, ax1 = plt.subplots(dpi=1000)
    fig.suptitle(f'Grid {rows}x{cols} - {n_enemies} enemies - Averaged over {n_games} games', fontsize=fontsize + 3)

    algorithms_checked = []
    for alg in algorithms:
        if all(is_number(np.load(f'{dir_results}/{alg}_computation_time_game{game}.npy')) for game in
               range(1, n_games + 1, 1)):
            algorithms_checked.append(alg)

    for alg in algorithms_checked:
        filename_rewards = [s for s in directories if f'{alg}_rewards' in s]

        av_rew = np.zeros(n_episodes)

        for n_game in range(n_games):
            av_rew = np.sum([av_rew, np.load(filename_rewards[n_game])], axis=0)

        av_rew = av_rew[:n_episodes]
        av_rewards = gaussian_filter1d(av_rew, 3)  # np.cumsum(av_rew, dtype=int)

        x = np.arange(0, len(av_rewards), 1)
        ax1.plot(x, av_rewards,
                 label=f'{alg} = {round(np.mean(av_rew) / n_games, 2)} \u00B1 {round(np.std(av_rew) / n_games, 2)}')
        confidence_interval_rew = np.std(av_rewards)
        ax1.fill_between(x, (av_rewards - confidence_interval_rew), (av_rewards + confidence_interval_rew), alpha=0.2)
        ax1.set_title('Average reward on each episode')
        ax1.legend(fontsize='x-small')
    plt.savefig(f'{dir_saving}/average_reward_episode_comparison_{kind_of_comparison}.pdf')
    # plt.show()


def plot_average_steps_episode(dir_results, algorithms, n_games, n_episodes, rows, cols, n_enemies, dir_saving, kind_of_comparison):
    target_pattern = fr"{dir_results}\*.npy"
    directories = glob.glob(target_pattern)

    fig, ax2 = plt.subplots(dpi=1000)
    fig.suptitle(f'Grid {rows}x{cols} - {n_enemies} enemies - Averaged over {n_games} games', fontsize=fontsize + 3)

    algorithms_checked = []
    for alg in algorithms:
        if all(is_number(np.load(f'{dir_results}/{alg}_computation_time_game{game}.npy')) for game in
               range(1, n_games + 1, 1)):
            algorithms_checked.append(alg)

    for alg in algorithms_checked:
        filename_steps = [s for s in directories if f'{alg}_steps' in s]

        av_steps = np.zeros(n_episodes)

        for n_game in range(n_games):
            av_steps = np.sum([av_steps, np.load(filename_steps[n_game])], axis=0)

        av_steps = av_steps[:n_episodes]
        av_steps_gauss = gaussian_filter1d(av_steps, 10)

        x = np.arange(0, len(av_steps), 1)
        ax2.plot(x, av_steps_gauss, label=f'{alg}')
        confidence_interval_steps = np.std(av_steps_gauss)
        ax2.fill_between(x, (av_steps_gauss - confidence_interval_steps), (av_steps_gauss + confidence_interval_steps), alpha=0.2)
        ax2.set_yscale('log')
        ax2.set_title('Actions needed to complete the episode')
        ax2.set_xlabel('Episode', fontsize=12)
        ax2.legend(fontsize='x-small')
    plt.savefig(f'{dir_saving}/steps_for_episode_comparison_{kind_of_comparison}.pdf')
    # plt.show()


def plot_average_computation_time(dir_results, algorithms, n_games, rows, cols, n_enemies, dir_saving, kind_of_comparison):
    targetPattern = fr"{dir_results}\*.npy"
    directories = glob.glob(targetPattern)
    # data_names = [os.path.basename(directories[s]) for s in range(len(directories))]

    fig, ax3 = plt.subplots(dpi=1000)
    fig.suptitle(f'Grid {rows}x{cols} - {n_enemies} enemies - Averaged over {n_games} games', fontsize=fontsize + 3)

    algorithms_checked = []
    for alg in algorithms:
        if all(is_number(np.load(f'{dir_results}/{alg}_computation_time_game{game}.npy')) for game in
               range(1, n_games + 1, 1)):
            algorithms_checked.append(alg)
        else:
            print(f'{alg} timeout occurred')

    data = []
    for alg in algorithms_checked:
        filename_comp_time = [s for s in directories if f'{alg}_computation_time' in s]

        av_comp_time = 0
        data_to_app = []
        for n_game in range(n_games):
            av_comp_time += np.load(filename_comp_time[n_game])
            data_to_app.append(av_comp_time)
        av_comp_time = av_comp_time / n_games
        data.append(data_to_app)

    ax3.boxplot(data, vert=True, patch_artist=True, labels=algorithms_checked)

    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=90)
    ax3.set_ylabel('Computation time [min]', fontsize=fontsize)
    ax3.set_xlabel('Algorithm', fontsize=fontsize)
    ax3.grid()
    ax3.set_title('Average computation time')
    fig.subplots_adjust(bottom=0.4)
    plt.savefig(f'{dir_saving}/average_comp_time_comparison_{kind_of_comparison}.pdf')
    # plt.show()
