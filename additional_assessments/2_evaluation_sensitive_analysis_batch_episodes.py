import itertools
import networkx as nx
from causalnex.structure import StructureModel
from matplotlib import pyplot as plt
import global_variables
from scripts.utils.test_causal_table import TestCausalTable
import json
import pandas as pd


# TODO: valuta il mergato se passa il test
def generate_plot(edges: list, title, if_arrows):
    sm_true = StructureModel()
    sm_true.add_edges_from(edges)

    plt.figure(dpi=1000)
    plt.title(f'{title}', fontsize=20)
    nx.draw(sm_true, with_labels=True, font_size=7, arrowsize=30, arrows=if_arrows,
            edge_color='orange', node_size=1000, font_weight='bold', pos=nx.circular_layout(sm_true))
    plt.savefig(f'{title}.png')
    # plt.show()


N_SIMULATIONS_CONSIDERED = global_variables.N_SIMULATIONS_PAPER
path_list_values = f'{global_variables.GLOBAL_PATH_REPO}/Results/Sensitive_Analysis_Batch_Episodes/batch_episodes_for_online_cd_values.json'
with open(f'{path_list_values}', 'r') as file:
    list_values = json.load(file)

unique_values = {}

# Iterate over each dictionary in the list
for d in list_values:
    # Iterate over each key-value pair in the dictionary
    for key, value in d.items():
        # Exclude "causal_graph" key
        if key != "causal_graph":
            # If the key is not already in the unique_values dictionary, add it
            if key not in unique_values:
                unique_values[key] = set()
            # Convert lists to tuples before adding to set
            if isinstance(value, list):
                value = tuple(value)
            # Add the value to the set of unique values for the key
            unique_values[key].add(value)

combinations = []
for values in itertools.product(*unique_values.values()):
    combinations.append({
        "n_enemies": values[0],
        "n_episodes": values[1],
        "grid_size": values[2]
    })

list_rows = []
for new_dict_comb in combinations:
    n_enemies = new_dict_comb['n_enemies']
    grid_size = new_dict_comb['grid_size']
    n_episodes = new_dict_comb['n_episodes']

    target_key_values = {
        "n_enemies": n_enemies,
        "grid_size": list(grid_size),
        'n_episodes': n_episodes
    }

    filtered_dicts = [d for d in list_values if all(d.get(key) == value for key, value in target_key_values.items())][0]

    vet_causal_graphs = filtered_dicts['causal_graph']
    n_checks = 0
    for sim_n in range(N_SIMULATIONS_CONSIDERED):
        causal_graph = vet_causal_graphs[sim_n]
        generate_plot(causal_graph, f'{grid_size} - {n_enemies} enemies - {n_episodes} ep - {sim_n}', True)
        # print('ok' if len(causal_graph) == 6 else None)
        # if compare_causal_graphs(causal_graph, GROUND_TRUTH_CAUSAL_GRAPH):
        # TODO: test
        # n_checks += 1

    new_dict_comb[f'checks_over_{N_SIMULATIONS_CONSIDERED}simulations'] = n_checks
    new_dict_comb['suitable'] = n_checks > int(N_SIMULATIONS_CONSIDERED / 2)  # better than the random case

    list_rows.append(new_dict_comb)

out_table_results = pd.DataFrame(list_rows)
out_table_results.to_pickle(f'{global_variables.PATH_RESULTS_BATCH_EPISODES_ONLINE_CD}')
out_table_results.to_pickle(f'{global_variables.GLOBAL_PATH_REPO}/Results/Sensitive_Analysis_Batch_Episodes/res.pkl')
out_table_results.to_pickle(f'{global_variables.GLOBAL_PATH_REPO}/Results/Sensitive_Analysis_Batch_Episodes/res.xlsx')
out_table_results.to_excel(f'{global_variables.GLOBAL_PATH_REPO}/batch_episodes_online.xlsx')
