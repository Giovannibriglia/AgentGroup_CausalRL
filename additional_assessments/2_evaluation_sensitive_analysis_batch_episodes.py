import os
import networkx as nx
from causalnex.structure import StructureModel
from matplotlib import pyplot as plt
import global_variables
from scripts.utils.merge_causal_graphs import MergeCausalGraphs
from scripts.utils.test_causal_table import TestCausalTable
import json
import pandas as pd


# TODO: vedi se va bene il count separato e se valutare il merge
def generate_plot(edges: list, title, if_arrows):
    sm_true = StructureModel()
    sm_true.add_edges_from(edges)

    fig = plt.figure(dpi=1000)
    plt.title(f'{title}', fontsize=20)
    nx.draw(sm_true, with_labels=True, font_size=7, arrowsize=30, arrows=if_arrows,
            edge_color='orange', node_size=1000, font_weight='bold', pos=nx.circular_layout(sm_true))
    # plt.savefig(f'{title}.png')
    # plt.show()
    plt.close(fig)


N_SIMULATIONS_CONSIDERED = global_variables.N_SIMULATIONS_PAPER
N_EPISODES_ANALYSIS = global_variables.N_EPISODES_CONSIDERED_FOR_SENSITIVE_ANALYSIS_PAPER
DIR_RESULTS = 'Sensitive_Analysis_Batch_Episodes'

DIR_RESULTS = f'{global_variables.GLOBAL_PATH_REPO}/Results/{DIR_RESULTS}'
list_rows = []
for file_main_folder in os.listdir(DIR_RESULTS):
    with open(f'{DIR_RESULTS}/{file_main_folder}', 'r') as file:
        series = json.load(file)

        n_enemies = series['n_enemies']
        rows, cols = series['grid_size']
        n_episodes = series['n_episodes']
        dicts_dfs_track = series['dfs_track']
        # envs = series['envs']
        causal_graphs = series['causal_graphs']
        # dicts_causal_tables = series['causal_tables']
        print(f'*** {rows}x{cols} - {n_enemies} enemies - {n_episodes} episodes')
        list_results = [causal_graphs, dicts_dfs_track]
        merging = MergeCausalGraphs(list_results=list_results)
        merging.start_merging()
        merging.start_cd()
        out_causal_table = merging.get_causal_table()

        test = TestCausalTable(out_causal_table, global_variables.get_possible_actions)

        dict_row_results = {'grid_size': f'{rows}x{cols}',
                            'n_enemies': n_enemies,
                            'n_episodes': n_episodes,
                            'suitable': test.do_check()
                            }

        """
        count_ok = 0
            for dict_causal_table in dicts_causal_tables:
            causal_table = pd.DataFrame(dict_causal_table)
            print(len(causal_table))
            test = TestCausalTable(causal_table, global_variables.get_possible_actions)
            if test.do_check():
                count_ok += 1

        dict_row_results = {'grid_size': f'{rows}x{cols}',
                            'n_enemies': n_enemies,
                            'n_episodes': n_episodes,
                            f'checks_over_{len(dicts_causal_tables)}': count_ok,
                            'suitable': count_ok > int(len(dicts_causal_tables)/2)
                            }"""

        list_rows.append(dict_row_results)

out_table_results = pd.DataFrame(list_rows)
"""out_table_results.to_pickle(f'{global_variables.PATH_RESULTS_BATCH_EPISODES_ONLINE_CD}')
out_table_results.to_pickle(f'{global_variables.GLOBAL_PATH_REPO}/Results/Sensitive_Analysis_Batch_Episodes2/res.pkl')
out_table_results.to_pickle(f'{global_variables.GLOBAL_PATH_REPO}/Results/Sensitive_Analysis_Batch_Episodes2/res.xlsx')"""
out_table_results.to_excel(f'{global_variables.GLOBAL_PATH_REPO}/batch_episodes_online.xlsx')
