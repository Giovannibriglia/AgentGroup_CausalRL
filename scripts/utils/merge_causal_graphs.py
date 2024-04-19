from itertools import product
import networkx as nx
import pandas as pd
from causalnex.inference import InferenceEngine
from causalnex.network import BayesianNetwork
from matplotlib import pyplot as plt
import os
from collections import Counter
from causalnex.structure import StructureModel
import json


class MergeCausalGraphs:
    def __init__(self, dir_results: str = None, dict_results: dict = None, list_results: list = None):
        if dir_results is not None:
            self.dir_results = f'{dir_results}'
            self.dict_results = None
            self.list_results = None
        elif dict_results is not None:
            self.dict_results = dict_results
            self.dir_results = None
            self.list_results = None
        elif list_results is not None:
            self.list_results = list_results
            self.dict_results = None
            self.dir_results = None
        else:
            raise AssertionError('you must specify at least one between dir_results and dict_results')

    def start_merging(self):
        self.list_causal_graphs = []
        if self.dir_results is not None:
            for s in os.listdir(self.dir_results):
                if 'causal_graph' in s and ".json" in s:
                    with open(f'{self.dir_results}/{s}', 'r') as file:
                        graph = json.load(file)
                        self.list_causal_graphs.append(graph)
        elif self.dict_results is not None:
            for graph in self.dict_results['causal_graph']:
                self.list_causal_graphs.append(graph)
        elif self.list_results[0] is not None:
            for graph in self.list_results[0]:
                self.list_causal_graphs.append(graph)
        else:
            raise AssertionError('there are no causal graphs to merge')

        combined_list = []
        for single_causal_graph in self.list_causal_graphs:
            if single_causal_graph is not None:
                combined_list += single_causal_graph

        tuple_lists = [tuple(sublist) for sublist in combined_list]
        element_counts = Counter(tuple_lists)

        self.edges = []
        for element, count in element_counts.items():
            if count > int(len(self.list_causal_graphs) / 2):
                # print(f"Edge: {element}, Occurrences: {count}")
                self.edges.append(element)

        self._generate_plot(self.edges, 'Average causal graph', True)

    def get_merged_causal_graph(self):
        return self.edges

    def start_cd(self):
        self.df_tracks = []
        if self.dir_results is not None:
            self.df_tracks = [pd.read_pickle(f'{self.dir_results}/{s}') for s in os.listdir(self.dir_results) if 'df_track' in s]
        elif self.dict_results is not None:
            for df_as_dict in self.dict_results['df_track']:
                self.df_tracks.append(pd.DataFrame(df_as_dict))
        elif self.list_results[1] is not None:
            for df_as_dict in self.list_results[1]:
                self.df_tracks.append(pd.DataFrame(df_as_dict))

        # concat df
        concatenated_df = pd.DataFrame()
        for single_df in self.df_tracks:
            concatenated_df = pd.concat([concatenated_df, single_df], ignore_index=True)
        print('Length concatenated df: ', len(concatenated_df))

        features = set()
        for sublist in self.edges:
            features.update(sublist)
        features = list(features)

        # set dependents and independents variables
        independents_features, dependents_features = [], []
        for edge in self.edges:
            check = True
            for edge2 in self.edges:
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

        # structure model causal nex, bayesian network and inference engine
        sm = StructureModel()
        sm.add_edges_from(self.edges)

        bn = BayesianNetwork(sm)
        bn = bn.fit_node_states_and_cpds(concatenated_df)

        ie = InferenceEngine(bn)

        # do-interventions
        self.causal_table = pd.DataFrame(columns=features)

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
                    self.causal_table.at[n, f'{dependents_features[var_dep]}'] = int(comb_n[var_dep])
                except:
                    # print(f'Do-operation not possible for {dependents_features[var_dep]} = {int(comb_n[var_dep])}')
                    self.causal_table.at[n, f'{dependents_features[var_dep]}'] = pd.NA

            after = ie.query()
            for var_ind in independents_features:
                # print(f'Distribution of {var_ind}: {after[var_ind]}')
                max_key, max_value = max(after[var_ind].items(), key=lambda x: x[1])
                if round(max_value, 4) > round(1 / len(after[var_ind]), 4):
                    self.causal_table.at[n, f'{var_ind}'] = int(max_key)
                    # print(f'{var_ind} -> {int(max_key)}: {round(max_value, 2)}')
                else:
                    self.causal_table.at[n, f'{var_ind}'] = pd.NA
                    # print(f'{var_ind}) -> unknown')

            for var_dep in range(len(dependents_features)):
                ie.reset_do(dependents_features[var_dep])

        # clean
        self.causal_table.dropna(inplace=True)
        self.causal_table.reset_index(drop=True, inplace=True)

    def get_causal_table(self):
        return self.causal_table

    def _generate_plot(self, edges: list, title, if_arrows):
        sm_true = StructureModel()
        sm_true.add_edges_from(edges)

        fig = plt.figure(dpi=1000)
        plt.title(f'{title}', fontsize=20)
        nx.draw(sm_true, with_labels=True, font_size=7, arrowsize=30, arrows=if_arrows,
                edge_color='orange', node_size=1000, font_weight='bold', pos=nx.circular_layout(sm_true))
        plt.show()
        plt.close(fig)
