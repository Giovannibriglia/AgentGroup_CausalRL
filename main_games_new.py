
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import new_env_game
import os
import new_models
from scipy.ndimage import gaussian_filter1d
import plots
import time

# 'QL_EG', 'QL_SA', 'QL_BM', 'QL_TS' + all 'causal'
# 'DQN' + 'causal'
algorithms = ['QL_EG', 'QL_SA', 'QL_BM', 'QL_TS',
              'QL_EG_causal', 'QL_SA_causal', 'QL_BM_causal', 'QL_TS_causal']
n_games = 10
vect_rows = [5, 10]
vect_n_enemies = [2, 5, 10]
n_episodes = 2500
vect_if_maze = [False]
vect_if_same_enemies_actions = [False]
dir_start = f'Comp2_OnlineDoCalculus'
who_moves_first = 'Enemy'  # 'Enemy' or 'Agent'


os.makedirs(dir_start, exist_ok=True)
for if_maze in vect_if_maze:
    if if_maze:
        env_name = 'Maze'
    else:
        env_name = 'Grid'
    directory = dir_start + f'/{env_name}'
    os.makedirs(directory, exist_ok=True)

    for if_same_enemies_actions in vect_if_same_enemies_actions:
        if if_same_enemies_actions:
            en_act = 'SameEnAct'
        else:
            en_act = 'RandEnAct'
        directory = dir_start + f'/{env_name}'+ f'/{en_act}'
        os.makedirs(directory, exist_ok=True)
        for n_enemies in vect_n_enemies:
            directory = dir_start + f'/{env_name}' + f'/{en_act}' + f'/{n_enemies}Enem'
            os.makedirs(directory, exist_ok=True)
            for rows in vect_rows:
                """if n_enemies > 2 * rows:
                    break"""
                if n_enemies == 2 and rows == 5:
                    break
                cols = rows
                directory = dir_start + f'/{env_name}' + f'/{en_act}' + f'/{n_enemies}Enem' + f'/{rows}x{cols}'
                os.makedirs(directory, exist_ok=True)

                for game_n in range(1, n_games + 1, 1):
                    n_agents = 1
                    n_act_agents = 5
                    n_act_enemies = 5
                    n_goals = 1
                    env = new_env_game.CustomEnv(rows, cols, n_agents, n_act_agents, n_enemies, n_act_enemies, n_goals,
                                             if_maze, if_same_enemies_actions)

                    """fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, dpi=500)
                    fig.suptitle(f'{env_name} {rows}x{cols} - {n_enemies} enemies - Game {game_n}/{n_games}',
                                 fontsize=15)"""

                    for alg in algorithms:
                        print(f'\n*** {alg} - Game {game_n}/{n_games} ****')
                        time.sleep(1)

                        start_time = time.time()

                        env_for_alg = env
                        rewards = []
                        steps = []

                        # returned: reward for episode and steps for episode
                        if 'QL' in alg:
                            rewards, steps = new_models.QL_causality_online(env_for_alg, n_act_agents, n_episodes,
                                                                    alg, who_moves_first)
                        else:
                            rewards, steps = new_models.DQN_variations(env_for_alg, n_act_agents, n_episodes,
                                                                      alg, who_moves_first)

                        computation_time = (time.time() - start_time)/60

                        np.save(f"{directory}/{alg}_rewards_game{game_n}.npy", rewards)
                        np.save(f"{directory}/{alg}_steps_game{game_n}.npy", steps)
                        np.save(f'{directory}/{alg}_computation_time_game{game_n}.npy', computation_time)

                        """cumulative_rewards = np.cumsum(rewards, dtype=int)
                        x = np.arange(0, n_episodes, 1)
                        ax1.plot(x, cumulative_rewards, label=f'{alg} = {round(np.mean(rewards), 3)}')
                        confidence_interval_rew = np.std(cumulative_rewards)
                        ax1.fill_between(x, (cumulative_rewards - confidence_interval_rew),
                                         (cumulative_rewards + confidence_interval_rew),
                                         alpha=0.2)
                        ax1.set_title('Cumulative reward')
                        ax1.legend(fontsize='x-small')

                        ax2.plot(x, gaussian_filter1d(steps, 5))
                        ax2.set_yscale('log')
                        ax2.set_title('Actions needed to complete the episode')
                        ax2.set_xlabel('Episode', fontsize=12)

                    plt.savefig(f'{directory}/Comparison_Game{game_n}.pdf')
                    plt.show()"""

                """if n_games > 1:
                    plots.plot_av_rew_steps(directory, algorithms, n_games, n_episodes, rows, cols, n_enemies)
                    plots.plot_av_computation_time(directory, algorithms, n_games, rows, cols, n_enemies)"""

