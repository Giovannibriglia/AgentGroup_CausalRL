import json
import networkx as nx
from matplotlib import pyplot as plt
import global_variables
import os
from collections import Counter
from causalnex.structure import StructureModel
from scripts.utils.others import compare_causal_graphs

DIR_GET_RESULTS = 'OfflineCD_MultiEnv'

dir_get_results = f'{global_variables.GLOBAL_PATH_REPO}/Results/{DIR_GET_RESULTS}'
causal_graphs_json = [s for s in os.listdir(dir_get_results) if 'causal_graph' in s and '.json' in s]

combined_list = []
for single_file in causal_graphs_json:
    with open(f'{dir_get_results}/{single_file}', 'r') as file:
        causal_graph = json.load(file)
    combined_list += causal_graph

tuple_lists = [tuple(sublist) for sublist in combined_list]
element_counts = Counter(tuple_lists)

edges = []
for element, count in element_counts.items():
    if count > int(len(causal_graphs_json) / 2):
        # print(f"Edge: {element}, Occurrences: {count}")
        edges.append(element)


def generate_plot(edges: list, title):
    sm_true = StructureModel()
    sm_true.add_edges_from(edges)

    plt.figure(dpi=1000)
    plt.title(f'{title}', fontsize=20)
    # Draw the graph with custom labels
    nx.draw(sm_true, with_labels=True, font_size=9, arrowsize=30, arrows=True,
            edge_color='orange', node_size=1000, font_weight='bold', pos=nx.circular_layout(sm_true))
    plt.show()


sm = StructureModel()
sm.add_edges_from(edges)

plt.figure(dpi=1000)
plt.title(f'Average causal graph', fontsize=20)
# Draw the graph with custom labels
nx.draw(sm, with_labels=True, font_size=9, arrowsize=30, arrows=True,
        edge_color='orange', node_size=1000, font_weight='bold', pos=nx.circular_layout(sm))
plt.show()

with open(f'{global_variables.PATH_CAUSAL_GRAPH_OFFLINE}', 'r') as file:
    GROUND_TRUTH_CAUSAL_GRAPH = json.load(file)

edges_ground_truth = []
for element in GROUND_TRUTH_CAUSAL_GRAPH:
    edges_ground_truth.append(element)

sm_true = StructureModel()
sm_true.add_edges_from(edges_ground_truth)

plt.figure(dpi=1000)
plt.title(f'True causal graph', fontsize=20)
# Draw the graph with custom labels
nx.draw(sm_true, with_labels=True, font_size=9, arrowsize=30, arrows=True,
        edge_color='orange', node_size=1000, font_weight='bold', pos=nx.circular_layout(sm_true))
plt.show()

print('\n', compare_causal_graphs(edges_ground_truth, edges))

print(f'\nGround truth edges: {len(edges_ground_truth)} - Average causal graph: {len(edges)}')
