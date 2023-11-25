import time
import matplotlib.pyplot as plt
import numpy as np
import env_game
import os
import models
from scipy.ndimage import gaussian_filter1d

algorithms = ['QL', 'CQL3', 'CQL4', 'DeepQNetwork', 'DeepQNetwork_Mod']
n_games = 1
vect_rows = [5]
vect_n_enemies = [1]
n_episodes = 1000
vect_if_maze = [False]
vect_if_same_enemies_actions = [False]
dir_start = 'Results'

os.makedirs(dir_start, exist_ok=True)
for if_maze in vect_if_maze:
    directory = dir_start
    if if_maze:
        directory += '/Maze'
    else:
        directory += '/Grid'
    os.makedirs(directory, exist_ok=True)
    for if_same_enemies_actions in vect_if_same_enemies_actions:
        if if_same_enemies_actions:
            directory += '/SameEnAct'
        else:
            directory += '/RandEnAct'
        os.makedirs(directory, exist_ok=True)
        for n_enemies in vect_n_enemies:
            directory += f'/{n_enemies}En'
            os.makedirs(directory, exist_ok=True)
            for rows in vect_rows:
                cols = rows
                directory += f'/{rows}x{cols}'
                os.makedirs(directory, exist_ok=True)

                for game_n in range(1, n_games+1, 1):
                    n_agents = 1
                    n_act_agents = 5
                    n_act_enemies = 1
                    n_goals = 1
                    env = env_game.CustomEnv(rows, cols, n_agents, n_act_agents, n_enemies, n_act_enemies, n_goals,
                                             if_maze, if_same_enemies_actions)

                    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, dpi=500)
                    fig.suptitle('Performance comparison', fontsize=15)

                    for alg in algorithms:
                        print(f'\n*** {alg} ****')
                        time.sleep(1)
                        env_for_alg = env
                        rewards = []
                        steps = []
                        # returned: reward for episode and steps for episode
                        if alg == 'QL':
                            rewards, steps = models.QL(env_for_alg, n_act_agents, n_episodes)
                        elif alg == 'CQL3':
                            rewards, steps = models.CQL3(env_for_alg, n_act_agents, n_episodes)
                        elif alg == 'CQL4':
                            rewards, steps = models.CQL4(env_for_alg, n_act_agents, n_episodes)
                        elif alg == 'DeepQNetwork':
                            rewards, steps = models.DeepQNetwork(env_for_alg, n_act_agents, n_episodes)
                        elif alg == 'DeepQNetwork_Mod':
                            rewards, steps = models.DeepQNetwork_Mod(env_for_alg, n_act_agents, n_episodes)

                        np.save(f"{directory}/{alg}_rewards_game{game_n}.npy", rewards)
                        np.save(f"{directory}/{alg}_steps_game{game_n}.npy", steps)

                        ax1.plot(gaussian_filter1d(rewards, 2), label=f'{alg} = {round(np.mean(rewards), 3)}')
                        ax1.set_title('Average reward on episode steps')
                        ax1.legend()  # fontsize='x-small'

                        ax2.plot(gaussian_filter1d(steps, 2))
                        ax2.set_yscale('log')
                        ax2.set_title('Steps needed to complete the episode')
                        ax2.set_xlabel('Episode', fontsize=12)

                    plt.show()


