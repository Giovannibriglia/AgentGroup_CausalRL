import os
from itertools import product
import global_variables
import matplotlib.pyplot as plt
import json
import numpy as np
from scipy.ndimage import gaussian_filter1d
from decimal import *

fontsize = 12
SIGMA_GAUSSIAN_FILTER = 4
getcontext().prec = 5

dir_results = 'Comparison123'

n_episodes = global_variables.N_TRAINING_EPISODES

group_exp_strategies = global_variables.LIST_IMPLEMENTED_EXPLORATION_STRATEGIES
group_kind_algs = global_variables.LIST_IMPLEMENTED_ALGORITHMS
group_kind_algs.remove(global_variables.LABEL_RANDOM_AGENT)

vet_enemies = global_variables.N_ENEMIES_CONSIDERED_PAPER
vet_grid_sizes = global_variables.GRID_SIZES_CONSIDERED_PAPER

combinations_grid_enemies = list(product(vet_enemies, vet_grid_sizes))
combinations_algs = list(product(group_kind_algs, group_exp_strategies))

dir_saving_plots = f'{global_variables.GLOBAL_PATH_REPO}/Plots/{dir_results}'
os.makedirs(dir_saving_plots, exist_ok=True)
dir_saving_tables = f'{global_variables.GLOBAL_PATH_REPO}/Tables/{dir_results}'
os.makedirs(dir_saving_tables, exist_ok=True)

dir_results = f'{global_variables.GLOBAL_PATH_REPO}/Results/{dir_results}'
files_inside_main_folder = os.listdir(dir_results)


def drop_characters_after_first_word(string: str, words_to_drop: str) -> str:
    for word in words_to_drop:
        if word in string:
            return string.split(word)[0]
    return string


def IQM_mean(data):
    # Sort the data
    sorted_data = np.sort(data)

    # Calculate quartiles
    Q1 = np.percentile(sorted_data, 25)
    Q3 = np.percentile(sorted_data, 75)

    # Calculate IQR
    IQR = Q3 - Q1

    # Find indices of data within 1.5*IQR from Q1 and Q3
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    within_iqr_indices = np.where((sorted_data >= lower_bound) & (sorted_data <= upper_bound))[0]

    # Calculate IQM
    iq_mean = np.mean(sorted_data[within_iqr_indices])

    return iq_mean


def upload_fig(ax_n: plt.axes, values: list, mean_to_display: Decimal, std_to_display: Decimal, label_series: str):
    # TODO: COLORS, TIMEOUTS AND STD
    series_smooth = gaussian_filter1d(values[0], SIGMA_GAUSSIAN_FILTER)
    confidence_interval = Decimal(np.std(series_smooth)).quantize(Decimal('.01'))
    x_data = np.arange(0, len(series_smooth), 1)
    ax_n.plot(x_data, series_smooth,
              label=f'{label_series}: {mean_to_display} \u00B1 {std_to_display}')
    ax_n.fill_between(x_data, (series_smooth - confidence_interval), (series_smooth + confidence_interval),
                      alpha=0.2)
    ax_n.legend(fontsize='small')


for file_main_folder in files_inside_main_folder:

    save_plot = f'{dir_saving_plots}/{file_main_folder}'
    save_table = f'{dir_saving_tables}/{file_main_folder}'

    os.makedirs(save_plot, exist_ok=True)
    os.makedirs(save_table, exist_ok=True)

    file_main_folder = f'{dir_results}/{file_main_folder}'
    files_inside_second_folder = os.listdir(file_main_folder)

    dict_values = {f'{comb[0]}_{comb[1]}': {'rewards': [],
                                            'actions': [],
                                            'computation_time': [],
                                            'timeout': []} for comb in combinations_algs}
    # TODO: EXTRACT {n_games_performed}
    figures_subtitle = f'{os.path.basename(file_main_folder).replace("_", " ")} - Averaged over {10} games'

    for algorithm in dict_values.keys():
        selected_elements = [string for string in files_inside_second_folder if algorithm in string]

        for element in selected_elements:
            with open(f'{file_main_folder}/{element}', 'r') as file:
                series = json.load(file)

            label_plot = drop_characters_after_first_word(element, group_exp_strategies).replace("_", " ")[:-1]

            dict_values[algorithm]['rewards'].append(series[global_variables.KEY_METRIC_REWARDS_EPISODE])
            dict_values[algorithm]['actions'].append(series[global_variables.KEY_METRICS_STEPS_EPISODE])
            dict_values[algorithm]['computation_time'].append(series[global_variables.KEY_METRIC_TIME_EPISODE])
            dict_values[algorithm]['timeout'].append(series[global_variables.KEY_METRIC_TIMEOUT_CONDITION])

    for group_chosen in [group_exp_strategies, group_kind_algs]:

        for item_chosen in group_chosen:
            algs_chosen_from_dict = {key: value for key, value in dict_values.items() if item_chosen in key}

            fig_iqm_reward, ax_iqm_reward = plt.subplots(dpi=1000)
            ax_iqm_reward.set_title('IQM average reward')
            fig_iqm_reward.suptitle(f'{figures_subtitle}', fontsize=fontsize + 3)

            """fig_cum_reward, ax_cum_reward = plt.subplots(dpi=1000)
            ax_cum_reward.set_title('Cumulative reward')
            fig_cum_reward.suptitle(f'{figures_subtitle}', fontsize=fontsize + 3)

            fig_actions, ax_actions = plt.subplots(dpi=1000)
            ax_actions.set_title('Actions needed to complete the episode')
            fig_actions.suptitle(f'{figures_subtitle}', fontsize=fontsize + 3)

            fig_time, ax_time = plt.subplots(dpi=1000)
            ax_time.set_title('Computation time needed to complete the episode')
            fig_time.suptitle(f'{figures_subtitle}', fontsize=fontsize + 3)"""

            for algorithm, series in algs_chosen_from_dict.items():
                print(algorithm)
                if series['rewards']:

                    label_plot = algorithm

                    n_games_performed = len(series['timeout'])
                    n_games_ok = series['timeout'].count(False)

                    indices_to_remove = [i for i, value in enumerate(series['timeout']) if value]
                    for key in series:
                        if key != 'timeout':
                            series[key] = [sublist for i, sublist in enumerate(series[key]) if
                                           i not in indices_to_remove]

                    rewards_series = series['rewards']
                    actions_series = series['actions']
                    computation_time_series = series['computation_time']

                    x_data = np.arange(0, len(rewards_series[0]), 1)

                    cumulative_reward_series = [sum(values) / n_games_ok for values in zip(*rewards_series)]
                    IQM_cumulative_reward_series = Decimal(IQM_mean(cumulative_reward_series))

                    IQM_reward_series = Decimal(IQM_mean(rewards_series)).quantize(Decimal('.01'))
                    conf_interval_reward_series = Decimal(np.std(rewards_series)).quantize(Decimal('.01'))
                    upload_fig(ax_iqm_reward, rewards_series, IQM_reward_series, conf_interval_reward_series,
                               label_plot)

                    IQM_actions_needed = Decimal(IQM_mean(actions_series)).quantize(Decimal('.01'))

                    IQM_computation_time = Decimal(IQM_mean(computation_time_series)).quantize(Decimal('.01'))
            plt.show()

