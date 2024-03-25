import random
import re
import time
import warnings
from itertools import product
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from causalnex.inference import InferenceEngine
from causalnex.network import BayesianNetwork
from causalnex.structure.notears import from_pandas
from tqdm import tqdm
import os

import global_variables

warnings.filterwarnings("ignore")


# TODO: multi-agent and multi-goal settings
class CausalInference:

    def __init__(self, df):
        self.df = self._process_df(df)
        self.features_names = self.df.columns.to_list()

        self.structureModel = None
        self.bn = None
        self.ie = None
        self.independents_var = None
        self.dependents_var = None

    def _process_df(self, df_start):
        start_columns = df_start.columns.to_list()
        n_enemies_columns = [s for s in start_columns if global_variables.LABEL_ENEMY_CAUSAL_TABLE in s]
        if n_enemies_columns == 1:
            return df_start
        else:
            df_only_nearbies = df_start[n_enemies_columns]

            new_column = []
            for episode in range(len(df_start)):
                single_row = df_only_nearbies.loc[episode].tolist()

                if df_start.loc[episode, global_variables.COL_REWARD] == global_variables.VALUE_REWARD_LOSER_PAPER:
                    enemy_nearbies_true = [s for s in single_row if s != global_variables.VALUE_ENTITY_FAR]
                    action_agent = df_start.loc[episode, global_variables.COL_ACTION]

                    if action_agent in enemy_nearbies_true:
                        new_column.append(action_agent)
                    else:
                        new_column.append(global_variables.VALUE_ENTITY_FAR)
                else:
                    new_column.append(random.choice(single_row))

            df_out = df_start.drop(columns=n_enemies_columns)

            df_out[global_variables.COL_NEARBY_ENEMY] = new_column

            return df_out

    def training(self):
        print(f'structuring model...', len(self.df))
        self.structureModel = from_pandas(self.df)
        self.structureModel.remove_edges_below_threshold(0.2)

        print(f'training bayesian network...')
        self.bn = BayesianNetwork(self.structureModel)
        self.bn = self.bn.fit_node_states_and_cpds(self.df)

        bad_nodes = [node for node in self.bn.nodes if not re.match("^[0-9a-zA-Z_]+$", node)]
        if bad_nodes:
            print('Bad nodes: ', bad_nodes)

        self.ie = InferenceEngine(self.bn)

        print('do-calculus-1...')
        # understand who influences whom
        self._identify_ind_dep_variables()

        print('do-calculus-2...')
        # resume results in a table
        table = self._get_causal_table()

        return table

    def _identify_ind_dep_variables(self):
        self.dependents_var = []
        self.independents_var = []

        before = self.ie.query()
        for var in self.features_names:
            count_var = 0
            for value in self.df[var].unique():
                try:
                    self.ie.do_intervention(var, int(value))
                    after = self.ie.query()
                    features = [s for s in self.features_names if s not in var]
                    count_var_value = 0
                    for feat in features:
                        best_key_before, max_value_before = max(before[feat].items(), key=lambda x: x[1])
                        best_key_after, max_value_after = max(after[feat].items(), key=lambda x: x[1])

                        if best_key_after != best_key_before and round(max_value_after, 4) != round(
                                1 / len(after[feat]), 4):
                            count_var_value += 1
                    self.ie.reset_do(var)

                    count_var += count_var_value
                except:
                    pass

            if count_var > 0:
                # print(f'{var} --> {count_var} changes, internally caused ')
                self.dependents_var.append(var)
            else:
                # print(f'{var} --> externally caused')
                self.independents_var.append(var)

        print(f'**Independents vars: {self.independents_var}')
        print(f'**Dependents vars: {self.dependents_var}')

    def _get_causal_table(self):
        table = pd.DataFrame(columns=self.features_names)

        arrays = []
        for feat_ind in self.dependents_var:
            arrays.append(self.df[feat_ind].unique())
        var_combinations = list(product(*arrays))

        for n, comb_n in enumerate(var_combinations):
            # print('\n')
            for var_dep in range(len(self.dependents_var)):
                try:
                    self.ie.do_intervention(self.dependents_var[var_dep], int(comb_n[var_dep]))
                    # print(f'{self.dependents_var[var_dep]} = {int(comb_n[var_dep])}')
                    table.at[n, f'{self.dependents_var[var_dep]}'] = int(comb_n[var_dep])
                except:
                    # print(f'no {self.dependents_var[var_dep]} = {int(comb_n[var_dep])}')
                    table.at[n, f'{self.dependents_var[var_dep]}'] = pd.NA

            after = self.ie.query()
            for var_ind in self.independents_var:
                # print(f'{var_ind}) {after[var_ind]}')
                max_key, max_value = max(after[var_ind].items(), key=lambda x: x[1])
                if round(max_value, 4) != round(1 / len(after[var_ind]), 4):
                    table.at[n, f'{var_ind}'] = int(max_key)
                    # print(f'{var_ind}) -> {max_key}: {max_value}')
                else:
                    table.at[n, f'{var_ind}'] = pd.NA
                    # print(f'{var_ind}) -> unknown')

            for var_dep in range(len(self.dependents_var)):
                self.ie.reset_do(self.dependents_var[var_dep])

        return table
