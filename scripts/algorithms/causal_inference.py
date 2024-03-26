import random
import re
import time
import warnings
from itertools import product
from typing import Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
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

    def __init__(self, df: pd.DataFrame, n_agents: int, n_enemies: int, n_goals: int) -> pd.DataFrame:

        self.n_agents = n_agents
        self.n_enemies = n_enemies
        self.n_goals = n_goals

        if not (self.n_enemies == 1 and self.n_goals == 1):
            self.df = self._process_df(df)
        else:
            self.df = df

        # to speed up causal inference
        for col in self.df.columns:
            self.df[str(col)] = self.df[str(col)].astype(str).str.replace(',', '').astype(float)

        self.features_names = self.df.columns.to_list()

        self.structureModel = None
        self.bn = None
        self.ie = None
        self.independents_var = None
        self.dependents_var = None
        self.table = None

        self._training()

    def _process_df(self, df_start: pd.DataFrame) -> pd.DataFrame:
        """ "This class is designed to enhance the efficiency of the causal inference process. It consolidates the
        features Enemy_Nearby and Goal_Nearby, which correspond in number to the `n_enemies` and `n_agents`
         respectively, into two distinct classes. These classes store only the relevant values for the specified action.

        For instance, if `Enemy0_Nearby_Agent0 = 1` and `Enemy1_Nearby_Agent0 = 3`, and the `Action_Agent0 = 1`,
        then we store the value 1 in `Enemy_Nearby_Agent0`. The same principle applies to the feature
         `Goal0_Nearby_Agent0`. This approach is applicable to multi-agent systems as well." """

        def __generate_empty_list(X: int, data_type) -> list:
            return [data_type() for _ in range(X)]

        n_steps = len(df_start)
        start_columns = df_start.columns.to_list()

        action_agents = [s for s in start_columns if global_variables.LABEL_COL_ACTION in s]
        dict_new_cols = {}

        if self.n_enemies > 1:
            enemy_nearby_features = [s for s in start_columns if global_variables.LABEL_NEARBY_CAUSAL_TABLE in s
                                     and global_variables.LABEL_ENEMY_CAUSAL_TABLE in s]
            new_cols_nearby_enemy = [
                f'{global_variables.LABEL_ENEMY_CAUSAL_TABLE}0_{global_variables.LABEL_NEARBY_CAUSAL_TABLE}_{global_variables.LABEL_AGENT_CAUSAL_TABLE}{s}'
                for s in range(self.n_agents)]
            for agent in range(self.n_agents):
                dict_new_cols[f'{new_cols_nearby_enemy[agent]}'] = __generate_empty_list(n_steps, int)

        if self.n_goals > 1:
            goal_nearby_features = [s for s in start_columns if global_variables.LABEL_NEARBY_CAUSAL_TABLE in s
                                    and global_variables.LABEL_GOAL_CAUSAL_TABLE in s]
            new_cols_nearby_goal = [
                f'{global_variables.LABEL_GOAL_CAUSAL_TABLE}0_{global_variables.LABEL_NEARBY_CAUSAL_TABLE}_{global_variables.LABEL_AGENT_CAUSAL_TABLE}{s}'
                for s in range(self.n_agents)]
            for agent in range(self.n_agents):
                dict_new_cols[f'{new_cols_nearby_goal[agent]}'] = __generate_empty_list(n_steps, int)

        for step_n in range(n_steps):
            for agent in range(self.n_agents):

                if self.n_enemies > 1:
                    out1 = global_variables.VALUE_ENTITY_FAR
                    for enemy_nearby_feature in enemy_nearby_features:
                        if df_start.loc[step_n, enemy_nearby_feature] == df_start.loc[step_n, action_agents[agent]]:
                            out1 = df_start.loc[step_n, enemy_nearby_feature]
                    if out1 != global_variables.VALUE_ENTITY_FAR:
                        random_feature = random.choice(enemy_nearby_features)
                        out1 = df_start.loc[step_n, random_feature]
                    dict_new_cols[f'{new_cols_nearby_enemy[agent]}'][step_n] = out1

                if self.n_goals > 1:
                    out2 = global_variables.VALUE_ENTITY_FAR
                    for goal_nearby_feature in goal_nearby_features:
                        if df_start.loc[step_n, goal_nearby_feature] == df_start.loc[step_n, action_agents[agent]]:
                            out2 = df_start.loc[step_n, goal_nearby_feature]
                    if out2 != global_variables.VALUE_ENTITY_FAR:
                        random_feature = random.choice(goal_nearby_features)
                        out2 = df_start.loc[step_n, random_feature]
                    dict_new_cols[f'{new_cols_nearby_goal[agent]}'][step_n] = out2

        if self.n_enemies > 1:
            df_start.drop(columns=enemy_nearby_features, inplace=True)
        if self.n_goals > 1:
            df_start.drop(columns=goal_nearby_features, inplace=True)

        new_data = pd.DataFrame(dict_new_cols)
        df_start = pd.concat([df_start, new_data], axis=1)

        return df_start

    def _training(self):
        print(f'structuring model through NOTEARS... {len(self.df)} actions')
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
        self.__identify_ind_dep_variables()

        print('do-calculus-2...')
        # resume results in a table
        self.table = self.__apply_causal_inference()

    def __identify_ind_dep_variables(self):
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

    def __apply_causal_inference(self) -> pd.DataFrame:
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

    def return_causal_table(self):
        return self.table

    """
    def binarize_dataframe(self):

        new_df = pd.DataFrame()

        for col in self.df.columns.to_list():
            if len(self.df[col].unique()) <= 2:
                col_ok = self.df[col]
                new_df.insert(loc=0, column=col_ok.name, value=col_ok)
            else:
                bins = self.df[col].unique().tolist()
                bins = sorted(bins)
                bins.insert(0, -100000)
                labels = []
                for bin in bins[1:]:
                    if bin >= 0:
                        labels.append('Value' + str(bin))
                    else:
                        labels.append('Value_' + str(abs(bin)))

                binary_df = pd.get_dummies(pd.cut(self.df[col], bins=bins, labels=labels), prefix=f'{col}')

                new_df = pd.concat([new_df, binary_df], axis=1)

        return new_df"""
