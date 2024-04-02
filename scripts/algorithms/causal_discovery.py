import random
import re
import warnings
from itertools import product
import networkx as nx
import pandas as pd
from causalnex.inference import InferenceEngine
from causalnex.network import BayesianNetwork
from causalnex.structure.notears import from_pandas
from matplotlib import pyplot as plt
from causalnex.structure import StructureModel
import global_variables
import json

warnings.filterwarnings("ignore")

FONT_SIZE_NODE_GRAPH = 7.5
ARROWS_SIZE_NODE_GRAPH = 30
NODE_SIZE_GRAPH = 1000


class CausalDiscovery:

    def __init__(self, df: pd.DataFrame, n_agents: int, n_enemies: int, n_goals: int, dir_saving_graphs: str = None,
                 name_saving_graphs: str = None):

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

        self.dir_saving = dir_saving_graphs
        self.name_save = name_saving_graphs

        # self._modify_action_values()

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
        print(f'structuring model through NOTEARS... {len(self.df)} timesteps')
        self.structureModel = from_pandas(self.df)
        self.structureModel.remove_edges_below_threshold(0.2)

        if self.dir_saving is not None and self.name_save is not None:
            self._plot_and_save_causal_graph(self.structureModel, False)

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

        if self.dir_saving is not None and self.name_save is not None:
            sm = StructureModel()
            edges = self.causal_relationships
            sm.add_edges_from(edges)
            self._plot_and_save_causal_graph(sm, True)

        print('do-calculus-2...')
        # resume results in a table
        self.table = self.__perform_interventions()

    def __identify_ind_dep_variables(self):
        self.dependents_var = []
        self.independents_var = []
        self.causal_relationships = []

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

                        uniform_probability_value = round(1 / len(after[feat]), 4)
                        if max_value_after > uniform_probability_value * 1.01 and best_key_after != best_key_before:
                            # print(f'{var} depends on {feat}')
                            self.causal_relationships.append((feat, var))

                            """if var == 'Action_Agent0':
                                print(f'\n{var} is classified as dependent because a consistent probability '
                                      f'distribution occurred when do({feat} = {value})')
                                print(feat, ' before intervention: ', before[feat])
                                print(feat, ' after intervention: ', after[feat])
                                print('Uniform probability value: ', round(1 / len(after[feat]), 4))
                                print(
                                    f'Best key before: {best_key_before}, max value before: {round(max_value_before, 4)}')
                                print(
                                    f'Best key after: {best_key_after}, max value after: {round(max_value_after, 4)}\n')"""

                            count_var_value += 1
                    self.ie.reset_do(var)

                    count_var += count_var_value
                except:
                    pass

            if count_var > 0:
                # print(f'*{var} --> {count_var} changes, internally caused -> dependent')
                self.dependents_var.append(var)
            else:
                # print(f'*{var} --> externally caused -> independent')
                self.independents_var.append(var)

        print(f'**Independents vars: {self.independents_var}')
        print(f'**Dependents vars: {self.dependents_var}')

    def __perform_interventions(self) -> pd.DataFrame:
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
        self.table.dropna(inplace=True)
        self.table.reset_index(drop=True, inplace=True)
        return self.table

    def _modify_action_values(self):

        columns_deltaX = [s for s in self.features_names if global_variables.LABEL_COL_DELTAX in s]
        columns_deltaY = [s for s in self.features_names if global_variables.LABEL_COL_DELTAY in s]
        columns_actions = [s for s in self.features_names if global_variables.LABEL_COL_ACTION in s]

        for agent in range(self.n_agents):
            col_deltaX = [s for s in columns_deltaX if f'{global_variables.LABEL_AGENT_CAUSAL_TABLE}{agent}' in s][0]
            col_deltaY = [s for s in columns_deltaY if f'{global_variables.LABEL_AGENT_CAUSAL_TABLE}{agent}' in s][0]
            col_action = [s for s in columns_actions if f'{global_variables.LABEL_AGENT_CAUSAL_TABLE}{agent}' in s][0]

            condition = (self.df[f'{col_deltaX}'] == 0) & (
                    self.df[f'{col_deltaY}'] == 0) & (self.df[f'{col_action}'] != 0)

            self.df.loc[condition, f'{col_action}'] = 0

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

    def _plot_and_save_causal_graph(self, sm: StructureModel, if_causal: bool):

        fig = plt.figure(dpi=1000)
        if if_causal:
            plt.title(f'Causal graph', fontsize=16)
        else:
            plt.title(f'NOTEARS graph', fontsize=16)
        nx.draw(sm, with_labels=True, font_size=FONT_SIZE_NODE_GRAPH,
                arrowsize=ARROWS_SIZE_NODE_GRAPH, arrows=if_causal,
                edge_color='orange', node_size=NODE_SIZE_GRAPH, font_weight='bold',
                pos=nx.circular_layout(sm))
        # plt.show()
        structure_to_save = [x for x in sm.edges]

        if if_causal:
            plt.savefig(f'{self.dir_saving}/{self.name_save}_causal_graph.png')

            with open(f'{self.dir_saving}/{self.name_save}_notears_structure.json', 'w') as json_file:
                json.dump(structure_to_save, json_file)
        else:
            plt.savefig(f'{self.dir_saving}/{self.name_save}_notears_graph.png')

            with open(f'{self.dir_saving}/{self.name_save}_causal_structure.json', 'w') as json_file:
                json.dump(structure_to_save, json_file)

        plt.close(fig)
