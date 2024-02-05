import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import new_env_game
import os
import new_models
from scipy.ndimage import gaussian_filter1d
import plots
import time
import random

seed_values = np.load('seed_values.npy')


def get_batch_episodes(n_enemies, rows):
    table = pd.read_pickle('TradeOff_causality_batch_episodes_enemies/results_tradeoff_online_causality.pkl')

    condition = (table['Grid Size'] == rows) & (table['Enemies'] == n_enemies) & (table['Suitable'] == 'yes')
    result_column = table.loc[condition, 'Episodes'].to_list()
    try:
        batch = min(result_column)
        if batch is not None:
            return batch
        else:
            return 500
    except:
        return 500


# 'QL_EG', 'QL_SA', 'QL_BM', 'QL_TS' + all 'causal' + 'offline'/'online'
# 'DQN' + 'causal'
algorithms = ['QL_EG_basic', 'QL_EG_causal_offline', 'QL_EG_causal_online',
              'QL_TS_basic', 'QL_TS_causal_offline', 'QL_TS_causal_online',
              'QL_SA_basic', 'QL_SA_causal_offline', 'QL_SA_causal_online',
              'QL_BM_basic', 'QL_BM_causal_offline', 'QL_BM_causal_online'
              ]

n_games = 5
vect_rows = [5, 10]
vect_n_enemies = [2, 5, 10]
n_episodes = 3000
vect_if_maze = [False]
vect_if_same_enemies_actions = [False]
dir_start = f'Results_Comparison123'
dir_start_env = f'Env_Comparison123'
who_moves_first = 'Enemy'  # 'Enemy' or 'Agent'

episodes_to_visualize = [0, int(n_episodes * 0.33), int(n_episodes * 0.66), n_episodes - 1]

os.makedirs(dir_start, exist_ok=True)
os.makedirs(dir_start_env, exist_ok=True)
for if_maze in vect_if_maze:

    if if_maze:
        env_name = 'Maze'
    else:
        env_name = 'Grid'
    directory = dir_start + f'/{env_name}'
    directory_env = dir_start_env + f'/{env_name}'
    os.makedirs(directory, exist_ok=True)
    os.makedirs(directory_env, exist_ok=True)

    for if_same_enemies_actions in vect_if_same_enemies_actions:
        if if_same_enemies_actions:
            en_act = 'SameEnAct'
        else:
            en_act = 'RandEnAct'
        directory = dir_start + f'/{env_name}' + f'/{en_act}'
        directory_env = dir_start_env + f'/{env_name}' + f'/{en_act}'
        os.makedirs(directory, exist_ok=True)
        os.makedirs(directory_env, exist_ok=True)

        for n_enemies in vect_n_enemies:
            directory = dir_start + f'/{env_name}' + f'/{en_act}' + f'/{n_enemies}Enem'
            directory_env = dir_start_env + f'/{env_name}' + f'/{en_act}' + f'/{n_enemies}Enem'
            os.makedirs(directory, exist_ok=True)
            os.makedirs(directory_env, exist_ok=True)
            for rows in vect_rows:
                if n_enemies > 2 * rows:
                    break

                cols = rows
                directory = dir_start + f'/{env_name}' + f'/{en_act}' + f'/{n_enemies}Enem' + f'/{rows}x{cols}'
                directory_env = dir_start_env + f'/{env_name}' + f'/{en_act}' + f'/{n_enemies}Enem' + f'/{rows}x{cols}'
                os.makedirs(directory, exist_ok=True)
                os.makedirs(directory_env, exist_ok=True)

                BATCH_EPISODES_UPDATE_BN = get_batch_episodes(n_enemies, rows)

                for game_n in range(1, n_games + 1, 1):
                    seed_value = seed_values[game_n]
                    np.random.seed(seed_value)
                    random.seed(seed_value)

                    n_agents = 1
                    n_act_agents = 5
                    n_act_enemies = 5
                    n_goals = 1

                    env = new_env_game.CustomEnv(rows, cols, n_agents, n_act_agents, n_enemies, n_act_enemies, n_goals,
                                                 if_maze, if_same_enemies_actions, directory, game_n, seed_value,
                                                 predefined_env=None)

                    np.save(f"{directory}/env_game{game_n}.npy", env.grid_for_game)
                    np.save(f"{directory_env}/env_game{game_n}.npy", env.grid_for_game)

                    for alg in algorithms:
                        print(f'\n*** {alg} - Game {game_n}/{n_games} ****')
                        time.sleep(1)

                        start_time = time.time()

                        env_for_alg = env
                        rewards = []
                        steps = []

                        # returned: reward for episode, actions for episode and the final Q-table
                        if 'QL' in alg:
                            if 'offline' in alg or 'basic' in alg:
                                rewards, steps, q_table = new_models.QL_causality_offline(env_for_alg, n_act_agents,
                                                                                          n_episodes,
                                                                                          alg, who_moves_first,
                                                                                          episodes_to_visualize,
                                                                                          seed_value)
                            else:
                                rewards, steps, q_table = new_models.QL_causality_online(env_for_alg, n_act_agents,
                                                                                         n_episodes,
                                                                                         alg, who_moves_first,
                                                                                         episodes_to_visualize,
                                                                                         seed_value,
                                                                                         BATCH_EPISODES_UPDATE_BN)

                        """else: TO FIX
                            rewards, steps, q_table = new_models.DQN_variations(env_for_alg, n_act_agents, n_episodes,
                                                                                alg, who_moves_first)"""

                        if len(rewards) == n_episodes:
                            computation_time = (time.time() - start_time) / 60  # minutes
                        else:
                            computation_time = 'timeout'

                        np.save(f"{directory}/{alg}_rewards_game{game_n}.npy", rewards)
                        np.save(f"{directory}/{alg}_steps_game{game_n}.npy", steps)
                        np.save(f'{directory}/{alg}_computation_time_game{game_n}.npy', computation_time)

                        if 'TS' in alg:
                            alpha, beta = q_table[0], q_table[1]
                            np.save(f'{directory}/{alg}_alpha_game{game_n}.npy', alpha)
                            np.save(f'{directory}/{alg}_beta_game{game_n}.npy', beta)
                        else:
                            np.save(f'{directory}/{alg}_q_table_game{game_n}.npy', q_table)

                # plots.plot_av_rew_steps(directory, algorithms, n_games, n_episodes, rows, cols, n_enemies)
                # plots.plot_av_computation_time(directory, algorithms, n_games, rows, cols, n_enemies)
