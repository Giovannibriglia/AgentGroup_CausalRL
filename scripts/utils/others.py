import os
import json
from typing import Tuple
import re
import numpy as np
import pandas as pd
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
    condition = ((table_batch['#Enemies'] == n_enemies) &
                 (table_batch['Grid Size'] == f'{n_rows}x{n_cols}') &
                 (table_batch['Suitable'] == 'yes'))

    filtered_df = table_batch[condition]

    if not filtered_df.empty:
        return filtered_df['n_episodes'].min()
    else:
        return 500


def extract_grid_size_and_n_enemies(input_string: str) -> Tuple[tuple, int]:
    match = re.match(r'(Grid|Maze)(\d+)x(\d+)_(\d+)(enemies|enemy)', input_string)
    if match:
        grid_size = (int(match.group(2)), int(match.group(3)))
        n_enemies = int(match.group(4))
        return grid_size, n_enemies
    else:
        return (None, None), None


def extract_grid_size(input_string: str) -> tuple:
    match1 = re.match(r'(Grid|Maze)(\d+)x(\d+)', input_string)
    match2 = re.match(r'(TorGrid|TorMaze)(\d+)x(\d+)', input_string)
    if match1:
        grid_size = (int(match1.group(2)), int(match1.group(3)))
        return grid_size
    elif match2:
        grid_size = (int(match2.group(2)), int(match2.group(3)))
        return grid_size
    else:
        return (None, None)
