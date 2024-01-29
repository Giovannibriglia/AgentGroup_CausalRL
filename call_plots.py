import plots
import os

possible_algorithms = ['QL_TS_basic', 'QL_TS_causal_offline', 'QL_TS_causal_online',
                       'QL_EG_basic', 'QL_EG_causal_offline', 'QL_EG_causal_online',
                       'QL_SA_basic', 'QL_SA_causal_offline', 'QL_SA_causal_online',
                       'QL_BM_basic', 'QL_BM_causal_offline', 'QL_BM_causal_online'
                       ]

group_by_kind = ['basic', 'causal_online', 'causal_offline']
group_by_strategy = ['TS', 'EG', 'SA', 'BM']

combs_algorithms_by_kind = [[s for s in possible_algorithms if kind in s] for kind in group_by_kind]
combs_algorithms_by_strategy = [[s for s in possible_algorithms if strategy in s] for strategy in group_by_strategy]

# algorithms = ['QL_TS_basic']
n_games = 5
vect_rows = [5, 10]
vect_n_enemies = [2, 5, 10]
n_episodes = 3000
vect_if_maze = [False]
vect_if_same_enemies_actions = [False]
dir_start = f'Results_Comparison123'

for comb_algorithms in [combs_algorithms_by_strategy, combs_algorithms_by_kind]:
    for algorithms in comb_algorithms:
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
                directory = dir_start + f'/{env_name}' + f'/{en_act}'
                os.makedirs(directory, exist_ok=True)
                for n_enemies in vect_n_enemies:
                    directory = dir_start + f'/{env_name}' + f'/{en_act}' + f'/{n_enemies}Enem'
                    os.makedirs(directory, exist_ok=True)
                    for rows in vect_rows:
                        if n_enemies > 2 * rows:
                            break

                        cols = rows
                        directory = dir_start + f'/{env_name}' + f'/{en_act}' + f'/{n_enemies}Enem' + f'/{rows}x{cols}'

                        directory_for_saving = directory + '/plots'

                        os.makedirs(directory_for_saving, exist_ok=True)

                        plots.plot_cumulative_average_rewards(directory, algorithms, n_games, n_episodes, rows, cols,
                                                              n_enemies, directory_for_saving)
                        plots.plot_average_rewards_episode(directory, algorithms, n_games, n_episodes, rows, cols,
                                                           n_enemies, directory_for_saving)
                        plots.plot_average_steps_episode(directory, algorithms, n_games, n_episodes, rows, cols,
                                                         n_enemies, directory_for_saving)
                        plots.plot_average_computation_time(directory, algorithms, n_games, rows, cols, n_enemies,
                                                            directory_for_saving)

# plots.plot_cumulative_average_rewards('5x5', algorithms, 5, 3000, 5, 5, 2)
# plots.plot_average_rewards_episode('5x5', algorithms, 5, 3000, 5, 5, 2)
# plots.plot_average_steps_episode('5x5', algorithms, 5, 3000, 5, 5, 2)
# plots.plot_average_computation_time('5x5', algorithms, 5, 5, 5, 2)
