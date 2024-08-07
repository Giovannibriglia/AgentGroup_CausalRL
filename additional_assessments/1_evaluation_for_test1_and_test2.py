import json
import os
import warnings

import global_variables
from scripts.utils.merge_causal_graphs import MergeCausalGraphs
from scripts.utils.test_causal_table import TestCausalTable

warnings.filterwarnings("ignore")


with open(f'{global_variables.PATH_CAUSAL_GRAPH_OFFLINE}', 'r') as file:
    GROUND_TRUTH_CAUSAL_GRAPH = json.load(file)

N_ENEMIES = 1
N_TRAINING_EPISODE = global_variables.N_TRAINING_EPISODES
NAME_DIR_RESULTS = f'{global_variables.GLOBAL_PATH_REPO}/Results/Test1'
files_inside_main_folder = os.listdir(NAME_DIR_RESULTS)

for file_inside_main_folder in files_inside_main_folder:
    print('\n', file_inside_main_folder)
    dict_new_row = {}

    n_simulations, n_checks_ok = 0, 0
    with open(f'{NAME_DIR_RESULTS}/{file_inside_main_folder}', 'r') as file:
        dict_results = json.load(file)
        try:
            merging = MergeCausalGraphs(dict_results=dict_results)
            merging.start_merging()
            out_causal_graph = merging.get_merged_causal_graph()

            rows, cols = dict_results['grid_size']
            # merging._generate_plot(out_causal_graph, f'Average causal graph, grid {rows}x{cols}', True)

            merging.start_cd()
            out_causal_table = merging.get_causal_table()

            test = TestCausalTable(out_causal_table, global_variables.get_possible_actions)
            test.do_check()
        except:
            print('no')

