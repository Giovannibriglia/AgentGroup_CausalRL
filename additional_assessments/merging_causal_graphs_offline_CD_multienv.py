import json
from itertools import product
import networkx as nx
import pandas as pd
from causalnex.inference import InferenceEngine
from causalnex.network import BayesianNetwork
from matplotlib import pyplot as plt
import global_variables
import os
from collections import Counter
from causalnex.structure import StructureModel
from scripts.utils.others import compare_causal_graphs

DIR_GET_RESULTS = 'OfflineCD_MultiEnv'

dir_get_results = f'{global_variables.GLOBAL_PATH_REPO}/Results/{DIR_GET_RESULTS}'
causal_graphs_json = [s for s in os.listdir(dir_get_results) if 'causal_graph' in s and '.json' in s]
df_tracks = [s for s in os.listdir(dir_get_results) if 'df_track' in s and '.pkl' in s]


def generate_plot(edges: list, title):
    sm_true = StructureModel()
    sm_true.add_edges_from(edges)

    plt.figure(dpi=1000)
    plt.title(f'{title}', fontsize=20)
    nx.draw(sm_true, with_labels=True, font_size=7, arrowsize=30, arrows=True,
            edge_color='orange', node_size=1000, font_weight='bold', pos=nx.circular_layout(sm_true))
    plt.show()


combined_list = []
for single_file_graph in causal_graphs_json:
    with open(f'{dir_get_results}/{single_file_graph}', 'r') as file:
        causal_graph = json.load(file)
    combined_list += causal_graph

tuple_lists = [tuple(sublist) for sublist in combined_list]
element_counts = Counter(tuple_lists)

edges = []
for element, count in element_counts.items():
    if count > int(len(causal_graphs_json) / 2):
        # print(f"Edge: {element}, Occurrences: {count}")
        edges.append(element)

generate_plot(edges, 'Average causal graph')

features = set()
for sublist in edges:
    features.update(sublist)
features = list(features)

# set dependents and independents variables
independents_features, dependents_features = [], []
for edge in edges:
    check = True
    for edge2 in edges:
        if edge[1] in edge2[0]:
            check = False
    if check:
        independents_features.append(edge[1])
        dependents_features.append(edge[0])
    else:
        dependents_features.append(edge[1])
        dependents_features.append(edge[0])
independents_features = list(set(independents_features))
dependents_features = list(set(dependents_features))
print('*** Independents variables: ', independents_features)
print('*** Dependents variables: ', dependents_features)

# concat df
concatenated_df = pd.DataFrame()
for single_file_df in df_tracks:
    df = pd.read_pickle(f'{dir_get_results}/{single_file_df}')
    concatenated_df = pd.concat([concatenated_df, df], ignore_index=True)
print('Length concatenated df: ', len(concatenated_df))

# structure model causal nex, bayesian network and inference engine
sm = StructureModel()
sm.add_edges_from(edges)

bn = BayesianNetwork(sm)
bn = bn.fit_node_states_and_cpds(concatenated_df)

ie = InferenceEngine(bn)

# do-interventions
table = pd.DataFrame(columns=features)

# DO-CALCULUS ON DEPENDENTS VARIABLES
arrays = []
for feat_dep in dependents_features:
    arrays.append(concatenated_df[feat_dep].unique())
var_combinations = list(product(*arrays))

for n, comb_n in enumerate(var_combinations):
    # print('\n')
    for var_dep in range(len(dependents_features)):
        try:
            ie.do_intervention(dependents_features[var_dep], int(comb_n[var_dep]))
            # print(f'{dependents_features[var_dep]} = {int(comb_n[var_dep])}')
            table.at[n, f'{dependents_features[var_dep]}'] = int(comb_n[var_dep])
        except:
            # print(f'Do-operation not possible for {dependents_features[var_dep]} = {int(comb_n[var_dep])}')
            table.at[n, f'{dependents_features[var_dep]}'] = pd.NA

    after = ie.query()
    for var_ind in independents_features:
        # print(f'Distribution of {var_ind}: {after[var_ind]}')
        max_key, max_value = max(after[var_ind].items(), key=lambda x: x[1])
        if round(max_value, 4) > round(1 / len(after[var_ind]), 4):
            table.at[n, f'{var_ind}'] = int(max_key)
            # print(f'{var_ind} -> {int(max_key)}: {round(max_value, 2)}')
        else:
            table.at[n, f'{var_ind}'] = pd.NA
            # print(f'{var_ind}) -> unknown')

    for var_dep in range(len(dependents_features)):
        ie.reset_do(dependents_features[var_dep])

# clean
table.dropna(inplace=True)
table.reset_index(drop=True, inplace=True)

# save
table.to_pickle(f'{global_variables.GLOBAL_PATH_REPO}/analysis_averaged.pkl')

print(table)

