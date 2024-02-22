import pandas as pd
import plots
import os


def find_common_words(list1, list2):
    # Split each string into words using underscores and convert them to sets
    set1 = set(word for s in list1 for word in s.split('_'))
    set2 = set(word for s in list2 for word in s.split('_'))

    # Find the intersection of the sets to get common words
    common_words = set1.intersection(set2)
    common_words_str = '_'.join(common_words)

    return common_words_str

"""'QL_TS_basic', 'QL_TS_causal_offline', 'QL_TS_causal_online',
       'QL_EG_basic', 'QL_EG_causal_offline', 'QL_EG_causal_online',
       'QL_SA_basic', 'QL_SA_causal_offline', 'QL_SA_causal_online',
       'QL_BM_basic', 'QL_BM_causal_offline', 'QL_BM_causal_online'"""

possible_algorithms = ['QL_TS_basic', 'QL_TS_causal_offline', 'QL_TS_causal_online',
       'QL_EG_basic', 'QL_EG_causal_offline', 'QL_EG_causal_online',
       'QL_SA_basic', 'QL_SA_causal_offline', 'QL_SA_causal_online',
       'QL_BM_basic', 'QL_BM_causal_offline', 'QL_BM_causal_online']

group_by_kind = ['basic', 'causal_offline', 'causal_online']
group_by_strategy = ['TS', 'EG', 'BM', 'SA']

combs_algorithms_by_kind = [[s for s in possible_algorithms if kind in s] for kind in group_by_kind]
combs_algorithms_by_strategy = [[s for s in possible_algorithms if strategy in s] for strategy in group_by_strategy]


n_games = 5
vect_rows = [10, 5]
vect_n_enemies = [10, 5, 2]
n_episodes = 3000
vect_if_maze = [True]
vect_if_same_enemies_actions = [False]
dir_start = f'Results/Results_Comparison4_NoTF'
dir_saving_plots = f'Plots/Plots_Comparison4_NoTF'
dir_saving_resume_metrics = f'Resume_Metrics/Metrics_Comparison4_NoTF'


for comb_algorithms in [combs_algorithms_by_strategy, combs_algorithms_by_kind]:

    if comb_algorithms == combs_algorithms_by_strategy:
        kind_of_comparison_start = 'comparison_by_strategy'
    elif comb_algorithms == combs_algorithms_by_kind:
        kind_of_comparison_start = 'comparison_by_kind'

    for algorithms in comb_algorithms:
        if comb_algorithms == combs_algorithms_by_strategy:
            kind_of_comparison = kind_of_comparison_start + '_' + str(find_common_words(algorithms, group_by_strategy))
        elif comb_algorithms == combs_algorithms_by_kind:
            kind_of_comparison = kind_of_comparison_start + '_' + str(find_common_words(algorithms, group_by_kind))

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
                        if n_enemies >= 2 * rows:
                            print(f'No {n_enemies} enemies - {rows}x{rows} grid')
                        else:
                            # print('\n')
                            cols = rows
                            directory = dir_start + f'/{env_name}' + f'/{en_act}' + f'/{n_enemies}Enem' + f'/{rows}x{cols}'
                            directory_for_saving_plots = dir_saving_plots + f'/{env_name}' + f'/{en_act}' + f'/{n_enemies}Enem' + f'/{rows}x{cols}'
                            directory_for_saving_resume_results = dir_saving_resume_metrics + f'/{env_name}' + f'/{en_act}' + f'/{n_enemies}Enem' + f'/{rows}x{cols}'

                            os.makedirs(directory_for_saving_plots, exist_ok=True)
                            os.makedirs(directory_for_saving_resume_results, exist_ok=True)

                            resume_metrics_table = pd.DataFrame(columns=['Algorithm'])
                            resume_metrics_table['Algorithm'] = algorithms

                            dict_res = plots.plot_cumulative_average_rewards(directory, algorithms, n_games, n_episodes, rows, cols,
                                                                  n_enemies, directory_for_saving_plots, kind_of_comparison, env_name)
                            resume_metrics_table['Cumulative_average_reward'] = list(dict_res.values())

                            dict_res = plots.plot_average_rewards_episode(directory, algorithms, n_games, n_episodes, rows, cols,
                                                               n_enemies, directory_for_saving_plots, kind_of_comparison, env_name)
                            resume_metrics_table['Average_episode_reward'] = list(dict_res.values())

                            dict_res = plots.plot_average_steps_episode(directory, algorithms, n_games, n_episodes, rows, cols,
                                                             n_enemies, directory_for_saving_plots, kind_of_comparison, env_name)
                            resume_metrics_table['Average_episode_steps'] = list(dict_res.values())

                            dict_res = plots.plot_average_computation_time(directory, algorithms, n_games, rows, cols, n_enemies,
                                                                directory_for_saving_plots, kind_of_comparison, env_name)
                            resume_metrics_table['Average_computation_time'] = list(dict_res.values())

                            dict_res = plots.reports_timeout_info(directory, algorithms, n_games, n_episodes)
                            resume_metrics_table['Timeout_info'] = list(dict_res.values())

                            resume_metrics_table.to_pickle(f'{directory_for_saving_resume_results}/resume_metrics_{kind_of_comparison}.pkl')
                            resume_metrics_table.to_excel(f'{directory_for_saving_resume_results}/resume_metrics_{kind_of_comparison}.xlsx')
