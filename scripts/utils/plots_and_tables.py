import json
import os
from itertools import product
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import global_variables
from scripts.utils import others
from scripts.utils.others import extract_grid_size_and_n_enemies

fontsize = 20
ticksize = 15


def get_results(dir_results, group_kind_of_algs, group_kind_of_exps, if_table, if_plots,
                if_comp4=False, N_GAMES_PERFORMED=global_variables.N_SIMULATIONS_PAPER, if_paper=False):
    combinations_algs = list(product(group_kind_of_algs, group_kind_of_exps))

    dir_saving_plots_and_table = f'{global_variables.GLOBAL_PATH_REPO}/Plots_and_Tables/{dir_results}'
    os.makedirs(dir_saving_plots_and_table, exist_ok=True)

    dir_results = f'{global_variables.GLOBAL_PATH_REPO}/Results/{dir_results}'
    files_inside_main_folder = os.listdir(dir_results)

    def drop_characters_after_first_word(string: str, words_to_drop: str) -> str:
        for word in words_to_drop:
            if word in string:
                return string.split(word)[0]
        return string

    def extract_from_input(input_string: str, possible_matches: list[str]) -> str:
        # Iterate through each string in the possible matches list
        for match in possible_matches:
            # Check if the match is a substring of the input string
            if match in input_string:
                return match  # Return the first matching substring found
        return ""  # Return an empty string if no match is found

    col_average_reward = 'IQM_average_reward'
    col_average_cumulative_reward = 'IQM_average_cumulative_reward'
    col_average_actions_needed = 'IQM_average_actions_needed'
    col_average_computation_time = 'IQM_average_computation_time[min]'
    col_timeout = '#timeouts'
    columns_table_results = ['grid_size', 'n_enemies', 'algo', 'exploration', f'{col_average_reward}',
                             f'{col_average_cumulative_reward}',
                             f'{col_average_actions_needed}', f'{col_average_computation_time}', f'{col_timeout}']

    table_results = pd.DataFrame(columns=columns_table_results)

    name_rewards_series = 'rewards'
    name_actions_series = 'actions'
    name_computation_time_series = 'computation_time'
    name_timeout_series = 'timeout'

    for file_main_folder in files_inside_main_folder:

        file_main_folder = f'{dir_results}/{file_main_folder}'
        files_inside_second_folder = os.listdir(file_main_folder)

        dict_values = {f'{comb[0]}_{comb[1]}': {f'{name_rewards_series}': [],
                                                f'{name_actions_series}': [],
                                                f'{name_computation_time_series}': [],
                                                f'{name_timeout_series}': []} for comb in combinations_algs}

        grid_size, n_enemies = extract_grid_size_and_n_enemies(os.path.basename(file_main_folder))
        figures_subtitle = f'{os.path.basename(file_main_folder).replace("_", " ")} - Averaged over {N_GAMES_PERFORMED} games'

        print(f'\n*** Grid/Maze {grid_size} - {n_enemies} enemy/enemies')
        # for making order
        for algorithm in dict_values.keys():
            selected_elements = [string for string in files_inside_second_folder if algorithm in string]

            if len(selected_elements) != N_GAMES_PERFORMED:
                print(f'* {algorithm} - missed {N_GAMES_PERFORMED - len(selected_elements)} files')
            for element in selected_elements:
                with open(f'{file_main_folder}/{element}', 'r') as file:
                    series = json.load(file)

                dict_values[algorithm][f'{name_rewards_series}'].append(
                    series[global_variables.KEY_METRIC_REWARDS_EPISODE])
                dict_values[algorithm][f'{name_actions_series}'].append(
                    series[global_variables.KEY_METRICS_STEPS_EPISODE])

                time_series_int = [element * 60 for element in series[global_variables.KEY_METRIC_TIME_EPISODE]]
                dict_values[algorithm][f'{name_computation_time_series}'].append(time_series_int)

                dict_values[algorithm][f'{name_timeout_series}'].append(
                    series[global_variables.KEY_METRIC_TIMEOUT_CONDITION])

        if if_table:
            # for table
            for algorithm, series in dict_values.items():
                if series[f'{name_rewards_series}']:
                    tot_indexes = len(series[f'{name_timeout_series}'])
                    ok_indexes = [index for index, value in enumerate(series[f'{name_timeout_series}']) if
                                  value == False]

                    if len(ok_indexes) != tot_indexes:
                        print(f'{algorithm}: {tot_indexes - len(ok_indexes)}/{tot_indexes} timeouts')

                    """for n, sublist in enumerate(series[f'{name_rewards_series}']):
                        print(f'sim {n} - {len(sublist)} completed episodes')"""

                    rewards_series = others.list_average(series[f'{name_rewards_series}'], ok_indexes)
                    cumulative_rewards_series = others.cumulative_list(rewards_series)
                    actions_series = others.list_average(series[f'{name_actions_series}'], ok_indexes)
                    computation_time_series = others.list_average(series[f'{name_computation_time_series}'], ok_indexes)

                    dict_metrics = others.compute_metrics(rewards_series, cumulative_rewards_series, actions_series,
                                                          computation_time_series,
                                                          col_average_cumulative_reward, col_average_reward,
                                                          col_average_actions_needed, col_average_computation_time)

                    cumulative_rewards_value_to_save = dict_metrics[f'{col_average_cumulative_reward}']
                    actions_value_to_save = dict_metrics[f'{col_average_actions_needed}']
                    reward_value_to_save = dict_metrics[f'{col_average_reward}']
                    computation_time_value_to_save = dict_metrics[f'{col_average_computation_time}']

                    algo_str = extract_from_input(algorithm, group_kind_of_algs)
                    exploration_str = extract_from_input(algorithm, group_kind_of_exps)

                    new_row_dict = {'grid_size': f'{grid_size[0]}x{grid_size[1]}', 'n_enemies': n_enemies,
                                    'algo': f'{algo_str}',
                                    'exploration': f'{exploration_str}',
                                    f'{col_average_reward}': reward_value_to_save,
                                    f'{col_average_cumulative_reward}': cumulative_rewards_value_to_save,
                                    f'{col_average_actions_needed}': actions_value_to_save,
                                    f'{col_average_computation_time}': computation_time_value_to_save,
                                    f'{col_timeout}': f'{tot_indexes - len(ok_indexes)}/{tot_indexes}' if tot_indexes != len(
                                        ok_indexes) else 'no'}

                    new_row_df = pd.DataFrame([new_row_dict])
                    table_results = pd.concat([table_results, new_row_df], ignore_index=True)

        if if_plots:
            # for plots and tables
            grid_enemies_values_save = os.path.basename(file_main_folder)
            save_plot = f'{dir_saving_plots_and_table}/{grid_enemies_values_save}'
            os.makedirs(save_plot, exist_ok=True)

            if if_comp4:
                groups = [['EG'], group_kind_of_exps, group_kind_of_algs]
            else:
                groups = [group_kind_of_exps, group_kind_of_algs]
            for group_chosen in groups:

                for item_chosen in group_chosen:
                    algos_chosen_from_dict = {key: value for key, value in dict_values.items() if item_chosen in key}

                    count_alg = 0
                    fig_reward, ax_reward = plt.subplots(dpi=1000, figsize=(16, 9))
                    """ax_reward.set_title('Reward', fontsize=fontsize)
                    fig_reward.suptitle(f'{figures_subtitle}', fontsize=fontsize + 3)"""
                    if if_paper:
                        axin_rew = inset_axes(ax_reward, width="52%", height="52%", borderpad=1, loc='center right')

                    fig_cum_reward, ax_cumul_reward = plt.subplots(dpi=1000, figsize=(16, 9))
                    """ax_cumul_reward.set_title('Cumulative reward', fontsize=fontsize)
                    fig_cum_reward.suptitle(f'{figures_subtitle}', fontsize=fontsize + 3)"""

                    fig_actions, ax_actions = plt.subplots(dpi=1000, figsize=(16, 9))
                    """ax_actions.set_title('Actions to complete the episode', fontsize=fontsize)
                    fig_actions.suptitle(f'{figures_subtitle}', fontsize=fontsize + 3)"""
                    ax_actions.set_yscale('log')

                    fig_time, ax_time = plt.subplots(dpi=1000, figsize=(16, 9))
                    """ax_time.set_title('Computation time to complete the episode', fontsize=fontsize)
                    fig_time.suptitle(f'{figures_subtitle}', fontsize=fontsize + 3)"""
                    # fig_time.subplots_adjust(bottom=0.45)
                    ax_time.set_ylabel('Minutes', fontsize=fontsize)
                    ax_time.set_xlabel('Algorithms', fontsize=fontsize)

                    for algorithm, series in algos_chosen_from_dict.items():
                        if series[f'{name_rewards_series}']:

                            label_plot = algorithm

                            n_games_performed = len(series[f'{name_timeout_series}'])
                            ok_indexes = [index for index, value in enumerate(series[f'{name_timeout_series}']) if
                                          value == False]
                            n_games_ok = len(ok_indexes)

                            if n_games_ok < n_games_performed:
                                str_timeout = f'{n_games_ok}/{n_games_performed}'
                            elif n_games_ok == n_games_performed:
                                str_timeout = None
                            else:
                                print(
                                    f'*** Problem with timeout counter: n_games_performed: {n_games_performed} - n_games_ok: {n_games_ok}')
                                str_timeout = None

                            rewards_series = others.list_average(series[f'{name_rewards_series}'], ok_indexes)
                            cumulative_rewards_series = others.cumulative_list(rewards_series)
                            actions_series = others.list_average(series[f'{name_actions_series}'], ok_indexes)
                            computation_time_series = others.list_average(series[f'{name_computation_time_series}'],
                                                                          ok_indexes)

                            dict_metrics = others.compute_metrics(rewards_series, cumulative_rewards_series,
                                                                  actions_series,
                                                                  computation_time_series,
                                                                  col_average_cumulative_reward,
                                                                  col_average_reward, col_average_actions_needed,
                                                                  col_average_computation_time)

                            cumulative_rewards_value_to_save = dict_metrics[f'{col_average_cumulative_reward}']
                            actions_value_to_save = dict_metrics[f'{col_average_actions_needed}']
                            reward_value_to_save = dict_metrics[f'{col_average_reward}']
                            computation_time_value_to_save = dict_metrics[f'{col_average_computation_time}']

                            others.upload_fig(ax_reward, rewards_series, reward_value_to_save, label_plot, str_timeout,
                                              count_alg)
                            if if_paper:
                                others.upload_fig(axin_rew, rewards_series, reward_value_to_save, label_plot,
                                                  str_timeout,
                                                  count_alg, if_legend=False)
                            others.upload_fig(ax_cumul_reward, cumulative_rewards_series,
                                              cumulative_rewards_value_to_save,
                                              label_plot, str_timeout, count_alg)
                            others.upload_fig(ax_actions, actions_series, actions_value_to_save, label_plot,
                                              str_timeout,
                                              count_alg)
                            others.upload_fig_time(ax_time, computation_time_series, computation_time_value_to_save,
                                                   label_plot,
                                                   str_timeout,
                                                   count_alg)

                            count_alg += 1

                    if if_paper:
                        axin_rew.set_xlim(-10, 500)
                        axin_rew.set_ylim(-5, 1.2)
                        # axin_rew.set_yticks([])
                        ax_reward.indicate_inset_zoom(axin_rew)
                        ax_reward.legend(loc='lower left', fontsize=25)

                        ax_actions.legend(loc='upper right', fontsize=32)

                    ax_time.set_xticklabels(ax_time.get_xticklabels(), rotation=90)

                    fig_time.tight_layout()
                    fig_actions.tight_layout()
                    fig_cum_reward.tight_layout()
                    fig_reward.tight_layout()

                    if if_comp4 and item_chosen == 'EG':
                        add_save = f'ALL_TF'
                    else:
                        add_save = item_chosen
                    fig_time.savefig(f'{save_plot}/time_{grid_enemies_values_save}_{add_save}.pdf')
                    fig_actions.savefig(f'{save_plot}/actions_{grid_enemies_values_save}_{add_save}.pdf')
                    fig_cum_reward.savefig(
                        f'{save_plot}/cumulative_reward_{grid_enemies_values_save}_{add_save}.pdf')
                    fig_reward.savefig(f'{save_plot}/reward_{grid_enemies_values_save}_{add_save}.pdf')

                    # plt.show()
                    plt.close(fig_time)
                    plt.close(fig_actions)
                    plt.close(fig_reward)
                    plt.close(fig_cum_reward)

    if if_table:
        table_results.to_excel(f'{dir_saving_plots_and_table}/results.xlsx')
        table_results.to_pickle(f'{dir_saving_plots_and_table}/results.pkl')


if __name__ == "__main__":
    """dir_res = 'Comparison123'
    group_algs = global_variables.LIST_IMPLEMENTED_ALGORITHMS
    group_algs.remove(global_variables.LABEL_RANDOM_AGENT)
    group_exps = ['EG'] #global_variables.LIST_IMPLEMENTED_EXPLORATION_STRATEGIES"""

    dir_res = 'Comparison4'
    group_algs = ['QL_vanilla_EG', 'QL_causal_offline_EG', 'QL_causal_online_EG',
                  'DQN_vanilla_EG', 'DQN_causal_offline_EG', 'DQN_causal_online_EG']
    group_exps = ['TransferLearning', 'NoTL']

    get_results(dir_res, group_algs, group_exps, if_table=True, if_plots=False, if_comp4=True, if_paper=True)
