import os
from itertools import product
import pandas as pd
import global_variables
import matplotlib.pyplot as plt
import json
import numpy as np
from scipy.ndimage import gaussian_filter1d
from decimal import *

from scripts.utils import others
from scripts.utils.others import extract_grid_size_and_n_enemies

fontsize = 12
SIGMA_GAUSSIAN_FILTER = 3

dir_results = 'Comparison123'
N_GAMES_PERFORMED = global_variables.N_SIMULATIONS_PAPER

n_episodes = global_variables.N_TRAINING_EPISODES

group_exp_strategies = global_variables.LIST_IMPLEMENTED_EXPLORATION_STRATEGIES
group_kind_algs = global_variables.LIST_IMPLEMENTED_ALGORITHMS
group_kind_algs.remove(global_variables.LABEL_RANDOM_AGENT)

vet_enemies = global_variables.N_ENEMIES_CONSIDERED_PAPER
vet_grid_sizes = global_variables.GRID_SIZES_CONSIDERED_PAPER

combinations_grid_enemies = list(product(vet_enemies, vet_grid_sizes))
combinations_algs = list(product(group_kind_algs, group_exp_strategies))

dir_saving_plots_and_table = f'{global_variables.GLOBAL_PATH_REPO}/Plots_and_Tables/{dir_results}'
os.makedirs(dir_saving_plots_and_table, exist_ok=True)

dir_results = f'{global_variables.GLOBAL_PATH_REPO}/Results/{dir_results}'
files_inside_main_folder = os.listdir(dir_results)


def drop_characters_after_first_word(string: str, words_to_drop: str) -> str:
    for word in words_to_drop:
        if word in string:
            return string.split(word)[0]
    return string


def upload_fig(ax_n: plt.axes, values: list, value_to_display: str, label_series: str,
               str_timeout: str):
    # TODO: FIX COLORS
    color_algo = global_variables.COLORS_ALGORITHMS[label_series]
    series_smooth = gaussian_filter1d(values, SIGMA_GAUSSIAN_FILTER)
    x_data = np.arange(0, len(series_smooth), 1)
    if str_timeout is not None:
        ax_n.plot(x_data, series_smooth, color=color_algo,
                  label=f'{label_series}: {value_to_display} ({str_timeout})')
    else:
        ax_n.plot(x_data, series_smooth, color=color_algo,
                  label=f'{label_series}: {value_to_display}')

    mean_str, std_str = value_to_display.split(' ± ')
    confidence_interval = float(std_str)
    ax_n.fill_between(x_data, (series_smooth - confidence_interval), (series_smooth + confidence_interval),
                      color=color_algo, alpha=0.2)

    ax_n.legend(fontsize='small')


col_average_reward = 'IQM_average_reward'
col_average_cumulative_reward = 'IQM_average_cumulative_reward'
col_average_actions_needed = 'IQM_average_actions_needed'
col_average_computation_time = 'IQM_average_computation_time'
columns_table_results = ['grid_size', 'n_enemies', 'algo', f'{col_average_reward}', f'{col_average_cumulative_reward}',
                         f'{col_average_actions_needed}', f'{col_average_computation_time}']

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

    # for make order
    for algorithm in dict_values.keys():
        selected_elements = [string for string in files_inside_second_folder if algorithm in string]
        print(algorithm, len(selected_elements))
        for element in selected_elements:
            print(element)
            with open(f'{file_main_folder}/{element}', 'r') as file:
                series = json.load(file)

            label_plot = drop_characters_after_first_word(element, group_exp_strategies).replace("_", " ")[:-1]

            dict_values[algorithm][f'{name_rewards_series}'].append(series[global_variables.KEY_METRIC_REWARDS_EPISODE])
            dict_values[algorithm][f'{name_actions_series}'].append(series[global_variables.KEY_METRICS_STEPS_EPISODE])
            dict_values[algorithm][f'{name_computation_time_series}'].append(
                series[global_variables.KEY_METRIC_TIME_EPISODE])
            dict_values[algorithm][f'{name_timeout_series}'].append(
                series[global_variables.KEY_METRIC_TIMEOUT_CONDITION])

    # for table
    for algorithm, series in dict_values.items():
        if series[f'{name_rewards_series}']:
            rewards_series = others.list_average(series[f'{name_rewards_series}'])
            cumulative_rewards_series = others.cumulative_list(rewards_series)
            actions_series = others.list_average(series[f'{name_actions_series}'])
            computation_time_series = others.list_average(series[f'{name_computation_time_series}'])

            dict_metrics = others.compute_metrics(rewards_series, cumulative_rewards_series, actions_series,
                                                  computation_time_series,
                                                  col_average_cumulative_reward, col_average_reward,
                                                  col_average_actions_needed, col_average_computation_time)

            cumulative_rewards_value_to_save = dict_metrics[f'{col_average_cumulative_reward}']
            actions_value_to_save = dict_metrics[f'{col_average_actions_needed}']
            reward_value_to_save = dict_metrics[f'{col_average_reward}']
            computation_time_value_to_save = dict_metrics[f'{col_average_computation_time}']

            new_row_dict = {'grid_size': f'{grid_size[0]}x{grid_size[1]}', 'n_enemies': n_enemies, 'algo': algorithm,
                            f'{col_average_reward}': reward_value_to_save,
                            f'{col_average_cumulative_reward}': cumulative_rewards_value_to_save,
                            f'{col_average_actions_needed}': actions_value_to_save,
                            f'{col_average_computation_time}': computation_time_value_to_save}

            new_row_df = pd.DataFrame([new_row_dict])
            table_results = pd.concat([table_results, new_row_df], ignore_index=True)

    """# for plots and tables
    save_plot = f'{dir_saving_plots_and_table}/{file_main_folder}'
    os.makedirs(save_plot, exist_ok=True)
    
    for group_chosen in [group_exp_strategies, group_kind_algs]:

        for item_chosen in group_chosen:
            algos_chosen_from_dict = {key: value for key, value in dict_values.items() if item_chosen in key}

            fig_iqm_reward, ax_iqm_reward = plt.subplots(dpi=1000)
            ax_iqm_reward.set_title('Average reward')
            fig_iqm_reward.suptitle(f'{figures_subtitle}', fontsize=fontsize + 3)

            fig_cum_reward, ax_cum_reward = plt.subplots(dpi=1000)
            ax_cum_reward.set_title('Cumulative reward')
            fig_cum_reward.suptitle(f'{figures_subtitle}', fontsize=fontsize + 3)

            fig_actions, ax_actions = plt.subplots(dpi=1000)
            ax_actions.set_title('Actions needed to complete the episode')
            fig_actions.suptitle(f'{figures_subtitle}', fontsize=fontsize + 3)

            fig_time, ax_time = plt.subplots(dpi=1000)
            ax_time.set_title('Computation time needed to complete the episode')
            fig_time.suptitle(f'{figures_subtitle}', fontsize=fontsize + 3)

            for algorithm, series in algos_chosen_from_dict.items():
                print(algorithm)
                if series[f'{name_rewards_series}']:

                    label_plot = algorithm

                    n_games_performed = len(series[f'{name_timeout_series}'])
                    n_games_ok = series[f'{name_timeout_series}'].count(False)

                    if n_games_ok < n_games_performed:
                        str_timeout = f'{n_games_ok}/{n_games_performed}'
                    elif n_games_ok == n_games_performed:
                        str_timeout = None
                    else:
                        print(
                            f'*** Problem with timeout counter: n_games_performed: {n_games_performed} - n_games_ok: {n_games_ok}')
                        str_timeout = None

                    indices_to_remove = [i for i, value in enumerate(series[f'{name_timeout_series}']) if value]
                    for key in series:
                        if key != f'{name_timeout_series}':
                            series[key] = [sublist for i, sublist in enumerate(series[key]) if
                                           i not in indices_to_remove]

                    rewards_series = list_average(series[f'{name_rewards_series}'])
                    cumulative_rewards_series = cumulative_list(rewards_series)
                    actions_series = list_average(series[f'{name_actions_series}'])
                    computation_time_series = list_average(series[f'{name_computation_time_series}'])

                    dict_metrics = compute_metrics(rewards_series, cumulative_rewards_series, actions_series,
                                                   computation_time_series)

                    cumulative_rewards_value_to_save = dict_metrics[f'{col_average_cumulative_reward}']
                    actions_value_to_save = dict_metrics[f'{col_average_actions_needed}']
                    reward_value_to_save = dict_metrics[f'{col_average_reward}']
                    computation_time_value_to_save = dict_metrics[f'{col_average_computation_time}']

                    upload_fig(ax_iqm_reward, rewards_series, reward_value_to_save, label_plot, str_timeout)
                    upload_fig(ax_cum_reward, cumulative_rewards_series, cumulative_rewards_value_to_save, label_plot,
                               str_timeout)
                    upload_fig(ax_actions, actions_series, actions_value_to_save, label_plot, str_timeout)
                    upload_fig(ax_time, computation_time_series, computation_time_value_to_save, label_plot,
                               str_timeout)

            plt.show()"""

table_results.to_excel(f'{dir_saving_plots_and_table}/results.xlsx')
table_results.to_pickle(f'{dir_saving_plots_and_table}/results.pkl')
