import os
from itertools import product
import global_variables
import matplotlib.pyplot as plt
import json
import numpy as np

dir_results = 'Comparison123'

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


def drop_characters_after_first_word(string, words_to_drop):
    for word in words_to_drop:
        if word in string:
            return string.split(word)[0]
    return string


for file_main_folder in files_inside_main_folder:

    save_plot = f'{dir_saving_plots}/{file_main_folder}'
    save_table = f'{dir_saving_tables}/{file_main_folder}'

    os.makedirs(save_plot, exist_ok=True)
    os.makedirs(save_table, exist_ok=True)

    file_main_folder = f'{dir_results}/{file_main_folder}'
    files_inside_second_folder = os.listdir(file_main_folder)

    fig1 = plt.figure()
    fig2 = plt.figure()
    fig3 = plt.figure()

    for file_inside_second_folder in files_inside_second_folder:

        with open(f'{file_main_folder}/{file_inside_second_folder}', 'r') as file:
            series = json.load(file)

        label_plot = drop_characters_after_first_word(file_inside_second_folder, group_exp_strategies).replace("_", " ")[:-1]

        rewards_series = series[global_variables.KEY_METRIC_REWARDS_EPISODE]
        steps_for_episode = series[global_variables.KEY_METRICS_STEPS_EPISODE]
        time_for_episode = series[global_variables.KEY_METRIC_TIME_EPISODE]
        if_timeout = series[global_variables.KEY_METRIC_TIMEOUT_CONDITION]

        x_data = np.arange(0, len(rewards_series), 1)
        # Plot on each figure
        plt.figure(fig1.number)  # Switch to fig1
        plt.plot(x_data, rewards_series, label=label_plot)
        # Customize other plot attributes as needed

        plt.figure(fig2.number)  # Switch to fig2
        plt.plot(x_data, steps_for_episode, label=label_plot)
        # Customize other plot attributes as needed

        plt.figure(fig3.number)  # Switch to fig3
        plt.plot(x_data, time_for_episode, label=label_plot)
        # Customize other plot attributes as needed

        plt.show()


