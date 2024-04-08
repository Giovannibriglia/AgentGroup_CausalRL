import json
import networkx as nx
from matplotlib import pyplot as plt
import global_variables
import os
from collections import Counter
from causalnex.structure import StructureModel

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
    if count > int(len(causal_graphs_json)/2):
        print(f"Element: {element}, Occurrences: {count}")
        edges.append(element)

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

edges2 = []
for element in GROUND_TRUTH_CAUSAL_GRAPH:
    edges2.append(element)

sm2 = StructureModel()
sm2.add_edges_from(edges2)

plt.figure(dpi=1000)
plt.title(f'True causal graph', fontsize=20)
# Draw the graph with custom labels
nx.draw(sm2, with_labels=True, font_size=9, arrowsize=30, arrows=True,
        edge_color='orange', node_size=1000, font_weight='bold', pos=nx.circular_layout(sm2))
plt.show()
