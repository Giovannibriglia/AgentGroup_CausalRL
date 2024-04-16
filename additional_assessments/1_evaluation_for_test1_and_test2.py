import numpy as np

import global_variables
import json
import os
from scripts.utils.merge_causal_graphs import MergeCausalGraphs
from scripts.utils.others import extract_grid_size, compare_causal_graphs
from scripts.utils.test_causal_table import TestCausalTable
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


with open(f'{global_variables.PATH_CAUSAL_GRAPH_OFFLINE}', 'r') as file:
    GROUND_TRUTH_CAUSAL_GRAPH = json.load(file)

N_ENEMIES = 1
N_TRAINING_EPISODE = global_variables.N_TRAINING_EPISODES
NAME_DIR_RESULTS = f'{global_variables.GLOBAL_PATH_REPO}/Results/Test2'
files_inside_main_folder = os.listdir(NAME_DIR_RESULTS)

col_n_checks_ok = 'n_checks_ok'
col_suitable = 'suitable'
cols_table_results = ['grid_size', 'n_enemies', 'n_episodes', 'n_simulations', f'{col_n_checks_ok}', f'{col_suitable}']
# table_results = pd.DataFrame(columns=cols_table_results)

for file_inside_main_folder in files_inside_main_folder:
    print('\n', file_inside_main_folder)
    dict_new_row = {}
    """grid_size = extract_grid_size(file_inside_main_folder)
    dict_new_row['grid_size'] = f'{grid_size[0]}x{grid_size[1]}'
    dict_new_row['n_enemies'] = N_ENEMIES
    dict_new_row['n_episodes'] = N_TRAINING_EPISODE"""

    n_simulations, n_checks_ok = 0, 0
    with open(f'{NAME_DIR_RESULTS}/{file_inside_main_folder}', 'r') as file:
        dict_results = json.load(file)

        merging = MergeCausalGraphs(dict_results=dict_results)
        merging.start_merging()
        out_causal_graph = merging.get_merged_causal_graph()

        merging.start_cd()
        out_causal_table = merging.get_causal_table()

        test = TestCausalTable(out_causal_table, global_variables.get_possible_actions)
        test.do_check()

        """dict_new_row['n_simulations'] = n_simulations
        dict_new_row[f'{col_n_checks_ok}'] = n_checks_ok
        dict_new_row[f'{col_suitable}'] = n_checks_ok > int(n_simulations/2)
    
        new_row_df = pd.DataFrame([dict_new_row])
        table_results = pd.concat([table_results, new_row_df], ignore_index=True)
"""
# print(table_results)
# table_results.to_excel(f'{NAME_DIR_RESULTS}/results_analysis.xlsx')
