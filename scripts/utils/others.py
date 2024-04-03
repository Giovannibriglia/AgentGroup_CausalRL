import os
import json
import numpy as np
import pandas as pd

import global_variables


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


def compare_causal_graphs(cg1: list, cg2: list) -> bool:
    # cg1 and cg2 are lists of lists
    list1 = cg1.copy()
    list2 = cg2.copy()

    sorted_list1 = [sorted(sublist) for sublist in list1]
    sorted_list2 = [sorted(sublist) for sublist in list2]

    sorted_list1.sort()
    sorted_list2.sort()

    return sorted_list1 == sorted_list2


def get_batch_episodes(n_enemies: int, n_rows: int, n_cols: int) -> int:
    df_batch_episodes_online_CD = pd.read_pickle(f'{global_variables.PATH_RESULTS_BATCH_EPISODES_ONLINE_CD}')

    condition = ((df_batch_episodes_online_CD['n_enemies'] == n_enemies) &
                 (df_batch_episodes_online_CD['grid_size'] == (n_rows, n_cols)) &
                 (df_batch_episodes_online_CD['suitable'] == True))

    filtered_df = df_batch_episodes_online_CD[condition]

    if filtered_df:
        print('DEBUGGAREEEEE ', filtered_df['n_episodes'].min())
        return filtered_df['n_episodes'].min()
    else:
        print('DEBUGGARE')
        return 1000