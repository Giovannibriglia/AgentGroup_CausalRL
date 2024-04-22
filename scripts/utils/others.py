import os
import json
from decimal import Decimal
from typing import Tuple
import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d
import seaborn as sns

import global_variables


# TODO: SISTEMARE COME DIO COMANDA
def create_next_alg_folder(base_dir: str, core_word_path: str) -> str:
    # Ensure base directory exists
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Find existing folders with the given core word path
    alg_folders = [folder for folder in os.listdir(base_dir) if folder.startswith(core_word_path)]

    # Extract numbers from existing folders
    numbers = [int(folder.replace(core_word_path, "")) for folder in alg_folders if
               folder[len(core_word_path):].isdigit()]

    if numbers:
        next_number = max(numbers) + 1
    else:
        next_number = 1

    while True:
        new_folder_name = f"{core_word_path}{next_number}"
        new_folder_path = os.path.join(base_dir, new_folder_name)
        try:
            os.makedirs(new_folder_path)
            return new_folder_path
        except FileExistsError:
            next_number += 1


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert numpy array to list
        return json.JSONEncoder.default(self, obj)


def compare_causal_graphs(ground_truth: list, other_list: list) -> bool:
    # Convert ground_truth to a set of tuples for faster membership check
    ground_truth_set = sorted(set(tuple(item) for item in ground_truth))

    other_list_sorted = sorted(set(tuple(item) for item in other_list))

    in_other_list_not_in_ground_truth = []
    in_ground_truth_not_in_other_list = []

    for item in other_list_sorted:
        if item not in ground_truth_set:
            in_other_list_not_in_ground_truth.append(item)

    for item in ground_truth_set:
        if item not in other_list_sorted:
            in_ground_truth_not_in_other_list.append(item)

    if len(in_other_list_not_in_ground_truth) > 0:
        # print('In other list but not in ground truth: ', in_other_list_not_in_ground_truth, len(in_other_list_not_in_ground_truth))
        pass
    if len(in_ground_truth_not_in_other_list):
        # print('In ground truth but not in list: ', in_ground_truth_not_in_other_list, len(in_ground_truth_not_in_other_list))
        pass

    return len(in_other_list_not_in_ground_truth) == 0 and len(in_ground_truth_not_in_other_list) == 0


def get_batch_episodes(n_enemies: int, n_rows: int, n_cols: int, table_batch: pd.DataFrame) -> int:
    condition = ((table_batch['n_enemies'] == n_enemies) &
                 (table_batch['grid_size'] == f'{n_rows}x{n_cols}') &
                 (table_batch['suitable'] == 'yes'))

    filtered_df = table_batch[condition]

    if not filtered_df.empty:
        return filtered_df['n_episodes'].min()
    else:
        return 500


def extract_grid_size_and_n_enemies(input_string: str) -> Tuple[tuple, int]:
    match = re.match(r'(results_|)(grid|maze|TorGrid|Grid|Maze)(\d+)x(\d+)_(\d+)(enemies|enemy)', input_string)
    if match:
        grid_size = (int(match.group(3)), int(match.group(4)))
        n_enemies = int(match.group(5))
        return grid_size, n_enemies
    else:
        return (None, None), None


def IQM_mean(data: list) -> Decimal:
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
    iq_mean = Decimal(np.mean(sorted_data[within_iqr_indices])).quantize(Decimal('0.01'))

    return iq_mean


"""def list_average(list_of_lists: list[list], ok_indexes: list[int]) -> list:
    list_length = len(list_of_lists[0])
    averages = []
    for i in range(list_length):
        total = sum(sublist[i] for sublist in list_of_lists[ok_indexes])
        averages.append(total / len(list_of_lists))
    return averages"""


def list_average(list_of_lists: list[list], ok_indexes: list[int]) -> list:
    if not list_of_lists or not ok_indexes or len(ok_indexes) == 0:
        return []  # Return an empty list if there are no sublists or no ok indexes

    list_length = len(list_of_lists[ok_indexes[0]])
    averages = []
    for i in range(list_length):
        # Select only the sublist specified by ok_indexes and ensure the index is within the sublist length
        total = sum(list_of_lists[idx][i] for idx in ok_indexes if i < len(list_of_lists[idx]))
        count = sum(1 for idx in ok_indexes if i < len(list_of_lists[idx]))  # Count only the valid contributions
        averages.append(total / count if count > 0 else 0)  # Handle division by zero if no valid sublists
    return averages


def cumulative_list(input_list: list) -> list:
    cumulative_result = []
    cumulative_sum = 0
    for item in input_list:
        cumulative_sum += item
        cumulative_result.append(cumulative_sum)
    return cumulative_result


def compute_my_confidence_interval(data: list) -> Decimal:
    data_series = pd.Series(data)

    # Calculate the 25th and 75th percentiles
    q25 = data_series.quantile(0.25)
    q75 = data_series.quantile(0.75)

    # Filter the series to values between these percentiles
    filtered_series = data_series[(data_series >= q25) & (data_series <= q75)]

    value = Decimal(filtered_series.std()).quantize(Decimal('.01'))

    return value


def compute_metrics(rewards: list, cumulative_rewards: list, actions: list, computation_times: list,
                    col_average_cumulative_reward: str, col_average_reward: str, col_average_actions_needed: str,
                    col_average_computation_time: str) -> dict:
    dict_out = {}
    if len(cumulative_rewards) > 0:
        IQM_cumulative_reward_value = cumulative_rewards[-1]
        confidence_interval_cumulative_reward_series = compute_my_confidence_interval(rewards) * len(rewards)
        dict_out[
            f'{col_average_cumulative_reward}'] = f'{round(IQM_cumulative_reward_value, 2)} \u00B1 {round(confidence_interval_cumulative_reward_series, 2)}'
    else:
        dict_out[
            f'{col_average_cumulative_reward}'] = None

    if len(rewards) > 0:
        IQM_reward_value = IQM_mean(rewards)
        confidence_interval_reward_series = compute_my_confidence_interval(rewards)
        dict_out[f'{col_average_reward}'] = f'{IQM_reward_value} \u00B1 {confidence_interval_reward_series}'
    else:
        dict_out[f'{col_average_reward}'] = None

    if len(actions) > 0:
        IQM_actions_needed = IQM_mean(actions)
        confidence_interval_actions_needed_series = compute_my_confidence_interval(actions)
        dict_out[
            f'{col_average_actions_needed}'] = f'{IQM_actions_needed} \u00B1 {confidence_interval_actions_needed_series}'
    else:
        dict_out[f'{col_average_actions_needed}'] = None

    if len(computation_times) > 0:
        IQM_computation_time = IQM_mean(computation_times)
        confidence_interval_computation_time_series = compute_my_confidence_interval(computation_times)
        dict_out[
            f'{col_average_computation_time}'] = f'{IQM_computation_time} \u00B1 {confidence_interval_computation_time_series}'
    else:
        dict_out[f'{col_average_computation_time}'] = None

    return dict_out


SIGMA_GAUSSIAN_FILTER = 3


def upload_fig(ax_n: plt.axes, values: list, value_to_display: str, label_series: str,
               str_timeout: str, algo_number: int):
    colors = sns.color_palette("Set2")
    color_algo = colors[algo_number]

    series_smooth = gaussian_filter1d(values, SIGMA_GAUSSIAN_FILTER)
    x_data = np.arange(0, len(series_smooth), 1)
    if str_timeout is not None:
        ax_n.plot(x_data, series_smooth, color=color_algo,
                  label=f'{label_series}: {value_to_display} ({str_timeout})')
    else:
        ax_n.plot(x_data, series_smooth, color=color_algo,
                  label=f'{label_series}: {value_to_display}')

    data_series_pandas = pd.Series(values)
    rolling_max = data_series_pandas.rolling(window=25).max()
    rolling_min = data_series_pandas.rolling(window=25).min()
    ax_n.fill_between(x_data, (rolling_min), (rolling_max),
                      color=color_algo,
                      alpha=0.2)

    ax_n.legend(fontsize='small')
