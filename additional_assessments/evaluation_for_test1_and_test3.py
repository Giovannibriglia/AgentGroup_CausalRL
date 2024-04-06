import global_variables
import json
import os
from scripts.utils.others import extract_grid_size, compare_causal_graphs
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


with open(f'{global_variables.PATH_CAUSAL_GRAPH_OFFLINE}', 'r') as file:
    GROUND_TRUTH_CAUSAL_GRAPH = json.load(file)

N_ENEMIES = 1
N_TRAINING_EPISODE = global_variables.N_TRAINING_EPISODES
NAME_DIR_RESULTS = f'{global_variables.GLOBAL_PATH_REPO}/Results/Test3'
files_inside_main_folder = os.listdir(NAME_DIR_RESULTS)

col_n_checks_ok = 'n_checks_ok'
col_suitable = 'suitable'
cols_table_results = ['grid_size', 'n_enemies', 'n_episodes', 'n_simulations', f'{col_n_checks_ok}', f'{col_suitable}']
table_results = pd.DataFrame(columns=cols_table_results)

for file_inside_main_folder in files_inside_main_folder:

    dict_new_row = {}
    grid_size = extract_grid_size(file_inside_main_folder)
    dict_new_row['grid_size'] = f'{grid_size[0]}x{grid_size[1]}'
    dict_new_row['n_enemies'] = N_ENEMIES
    dict_new_row['n_episodes'] = N_TRAINING_EPISODE

    files_inside_second_folder = os.listdir(f'{NAME_DIR_RESULTS}/{file_inside_main_folder}')

    n_simulations, n_checks_ok = 0, 0
    for file_inside_second_folder in [s for s in files_inside_second_folder if 'causal_structure.json' in s]:
        with open(f'{NAME_DIR_RESULTS}/{file_inside_main_folder}/{file_inside_second_folder}', 'r') as file:
            causal_graph = json.load(file)

        n_simulations += 1
        if compare_causal_graphs(GROUND_TRUTH_CAUSAL_GRAPH, causal_graph):
            n_checks_ok += 1

    dict_new_row['n_simulations'] = n_simulations
    dict_new_row[f'{col_n_checks_ok}'] = n_checks_ok
    dict_new_row[f'{col_suitable}'] = n_checks_ok > int(n_simulations/2)

    new_row_df = pd.DataFrame([dict_new_row])
    table_results = pd.concat([table_results, new_row_df], ignore_index=True)

print(table_results)
# table_results.to_excel(f'{NAME_DIR_RESULTS}/results_analysis.xlsx')

