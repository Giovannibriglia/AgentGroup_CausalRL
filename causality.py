import random
import os
import re
import time
from itertools import combinations, product, permutations
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from causalnex.inference import InferenceEngine
from causalnex.network import BayesianNetwork
from causalnex.structure import StructureModel
from causalnex.structure.notears import from_pandas
import warnings
from scipy.stats import chi2_contingency, cramervonmises_2samp, pearsonr

warnings.filterwarnings("ignore")

visualization = False


class MiniGame:

    def __init__(self, n_goals):
        self.rows = 3
        self.cols = 3
        self.n_agents = 1
        self.n_actions = 5
        self.n_enemies = 1
        self.n_goals = n_goals

        # predefined action state
        self.predefined_delta_state = 50

        # rewards
        self.reward_alive = 0
        self.reward_winner = 1
        self.reward_gameover = -1

        # Action_Agent,  DeltaX_Agent, DeltaY_Agent, Alive_Agent, GameOver_Agent, Winner_Agent, Enemy_Nearby_Agent
        self.cols_df = []
        for agent in range(self.n_agents):
            self.cols_df.append(f'Action_Agent{agent}')
            self.cols_df.append(f'DeltaX_Agent{agent}')
            self.cols_df.append(f'DeltaY_Agent{agent}')
            if self.n_enemies > 0:
                self.cols_df.append(f'GameOver_Agent{agent}')
                self.cols_df.append(f'Alive_Agent{agent}')
            if self.n_goals > 0:
                self.cols_df.append(f'Winner_Agent{agent}')
            if self.n_goals > 0 or self.n_enemies > 0:
                self.cols_df.append(f'Reward_Agent{agent}')
            for enemy in range(self.n_enemies):
                self.cols_df.append(f'Enemy{enemy}_Nearby_Agent{agent}')
            for goal in range(self.n_goals):
                self.cols_df.append(f'Goal{goal}_Nearby_Agent{agent}')

        self.df = pd.DataFrame(columns=self.cols_df)

    def create_df(self, n_episodes, if_binary_df=True):

        print('creating dataframe...')
        time.sleep(2)

        counter = 0
        for _ in tqdm(range(n_episodes)):
            done = False
            agents_coord, enemies_coord, goals_coord = self.create_env()
            while not done:

                # enemies movement
                for enemy in range(self.n_enemies):
                    lastX_en = enemies_coord[enemy][0]
                    lastY_en = enemies_coord[enemy][1]
                    newX_en, newY_en, _, _, _ = self.get_action(lastX_en, lastY_en)
                    enemies_coord[enemy][0] = newX_en
                    enemies_coord[enemy][1] = newY_en

                # enemies nearbies direction
                for enemy in range(self.n_enemies):
                    x_en = enemies_coord[enemy][0]
                    y_en = enemies_coord[enemy][1]
                    for agent in range(self.n_agents):
                        x_ag = agents_coord[agent][0]
                        y_ag = agents_coord[agent][1]

                        nearby_distance = self.get_direction(x_ag, y_ag, x_en, y_en)
                        self.df.at[counter, f'Enemy{enemy}_Nearby_Agent{agent}'] = nearby_distance

                # goal nearbies direction
                for goal in range(self.n_goals):
                    x_goal = goals_coord[goal][0]
                    y_goal = goals_coord[goal][1]
                    for agent in range(self.n_agents):
                        x_ag = agents_coord[agent][0]
                        y_ag = agents_coord[agent][1]

                        nearby_distance = self.get_direction(x_ag, y_ag, x_goal, y_goal)
                        self.df.at[counter, f'Goal{goal}_Nearby_Agent{agent}'] = nearby_distance

                # agents movement
                for agent in range(self.n_agents):
                    lastX_ag = agents_coord[agent][0]
                    lastY_ag = agents_coord[agent][1]
                    newX_ag, newY_ag, action_ag, deltaX_ag, deltaY_ag = self.get_action(lastX_ag, lastY_ag)
                    agents_coord[agent][0] = newX_ag
                    agents_coord[agent][1] = newY_ag

                    self.df.at[counter, f'Action_Agent{agent}'] = action_ag
                    self.df.at[counter, f'DeltaX_Agent{agent}'] = deltaX_ag
                    self.df.at[counter, f'DeltaY_Agent{agent}'] = deltaY_ag

                # check game over, winner, nothing
                done = self.check_gameover_or_winner(agents_coord, enemies_coord, goals_coord, counter)
                counter += 1

        indep_var_names = []
        for feat1 in self.cols_df:
            p_values = []
            for feat2 in self.cols_df:
                if feat1.split('_')[0] != feat2.split('_')[0]:
                    res = cramervonmises_2samp(self.df[feat1], self.df[feat2])
                    p_values.append(res.pvalue)
            av = np.mean(p_values)

            if -0.05 < av < 0.05:
                indep_var_names.append(feat1)

        dep_var_names = [s for s in self.cols_df if s not in indep_var_names]
        print(f'Dependents features: {dep_var_names}')
        print(f'Independents features: {indep_var_names}')
        print('\n')

        if if_binary_df:
            df_bin = self.binarize_dataframe()
            for col in df_bin.columns:
                df_bin[str(col)] = df_bin[str(col)].astype(str).str.replace(',', '').astype(float)
            return df_bin, dep_var_names, indep_var_names
        else:
            for col in self.df.columns:
                self.df[str(col)] = self.df[str(col)].astype(str).str.replace(',', '').astype(float)
            return self.df, dep_var_names, indep_var_names

    def create_env(self):
        """ 0.0  1.0  2.0
            0.1  1.1  2.1 """
        agents_coord = []
        for agent in range(self.n_agents):
            x_ag = random.randint(0, self.cols - 1)
            y_ag = random.randint(0, self.rows - 1)
            agents_coord.append([x_ag, y_ag])

        enemies_coord = []
        for enemy in range(self.n_enemies):
            x_en = random.randint(0, self.cols - 1)
            y_en = random.randint(0, self.rows - 1)
            enemies_coord.append([x_en, y_en])

        goals_coord = []
        for goal in range(self.n_goals):
            x_goal = random.randint(0, self.cols - 1)
            y_goal = random.randint(0, self.rows - 1)
            goals_coord.append([x_goal, y_goal])

        return agents_coord, enemies_coord, goals_coord

    def get_direction(self, x1, y1, x2, y2):
        deltaX = x2 - x1
        deltaY = y2 - y1

        if deltaX == 0 and deltaY == 0:  # stop
            direction_1_respect_2 = 0
        elif deltaX == 1 and deltaY == 0:  # right
            direction_1_respect_2 = 1
        elif deltaX == -1 and deltaY == 0:  # left
            direction_1_respect_2 = 2
        elif deltaX == 0 and deltaY == 1:  # up
            direction_1_respect_2 = 3
        elif deltaX == 0 and deltaY == -1:  # down
            direction_1_respect_2 = 4
        elif deltaX == 1 and deltaY == 1 and self.n_actions > 5:  # diag up right
            direction_1_respect_2 = 5
        elif deltaX == 1 and deltaY == -1 and self.n_actions > 5:  # diag down right
            direction_1_respect_2 = 6
        elif deltaX == -1 and deltaY == 1 and self.n_actions > 5:  # diag up left
            direction_1_respect_2 = 7
        elif deltaX == -1 and deltaY == -1 and self.n_actions > 5:  # diag down left
            direction_1_respect_2 = 8
        else:  # otherwise
            direction_1_respect_2 = 50

        # print([x_ag, y_ag], [x_en, y_en], direction_1_respect_2)

        return direction_1_respect_2

    def get_action(self, last_stateX, last_stateY):

        action = random.randint(0, self.n_actions - 1)

        if action == 0:  # stop
            new_stateX = last_stateX
            new_stateY = last_stateY
            deltaX = 0
            deltaY = 0
        elif action == 1:  # right
            sub_action = 1
            if 0 <= sub_action + last_stateX < self.cols:
                new_stateX = sub_action + last_stateX
                deltaX = sub_action
            else:
                new_stateX = last_stateX
                deltaX = 0
                action = 0
            new_stateY = last_stateY
            deltaY = 0
        elif action == 2:  # left
            sub_action = -1
            if 0 <= sub_action + last_stateX < self.cols:
                new_stateX = sub_action + last_stateX
                deltaX = sub_action
            else:
                new_stateX = last_stateX
                deltaX = 0
                action = 0
            new_stateY = last_stateY
            deltaY = 0
        elif action == 3:  # up
            new_stateX = last_stateX
            deltaX = 0
            sub_action = 1
            if 0 <= sub_action + last_stateY < self.rows:
                new_stateY = sub_action + last_stateY
                deltaY = sub_action
            else:
                new_stateY = last_stateY
                deltaY = 0
                action = 0
        elif action == 4:  # down
            new_stateX = last_stateX
            deltaX = 0
            sub_action = -1
            if 0 <= sub_action + last_stateY < self.rows:
                new_stateY = sub_action + last_stateY
                deltaY = sub_action
            else:
                new_stateY = last_stateY
                deltaY = 0
                action = 0

        return new_stateX, new_stateY, action, deltaX, deltaY

    def check_gameover_or_winner(self, agents_coord, enemies_coord, goals_coord, row_number):
        done = False
        for agent in range(self.n_agents):
            x_ag = agents_coord[agent][0]
            y_ag = agents_coord[agent][1]

            if self.n_goals > 0:
                for goal in range(self.n_goals):
                    x_goal = goals_coord[goal][0]
                    y_goal = goals_coord[goal][1]

                    if [x_ag, y_ag] == [x_goal, y_goal]:
                        done = True
                        self.df.at[row_number, f'Winner_Agent{agent}'] = 1

            if done:
                self.df.at[row_number, f'Reward_Agent{agent}'] = self.reward_winner
                if self.n_enemies > 0:
                    self.df.at[row_number, f'GameOver_Agent{agent}'] = 0
                    self.df.at[row_number, f'Alive_Agent{agent}'] = 0
            else:
                if self.n_goals > 0:
                    self.df.at[row_number, f'Winner_Agent{agent}'] = 0
                if self.n_enemies > 0:

                    for enemy in range(self.n_enemies):
                        x_en = enemies_coord[enemy][0]
                        y_en = enemies_coord[enemy][1]

                        if [x_ag, y_ag] == [x_en, y_en]:
                            done = True

                    if done:
                        self.df.at[row_number, f'GameOver_Agent{agent}'] = 1
                        self.df.at[row_number, f'Reward_Agent{agent}'] = self.reward_gameover
                        self.df.at[row_number, f'Alive_Agent{agent}'] = 0
                    else:
                        self.df.at[row_number, f'GameOver_Agent{agent}'] = 0
                        self.df.at[row_number, f'Reward_Agent{agent}'] = self.reward_alive
                        self.df.at[row_number, f'Alive_Agent{agent}'] = 1

        return done

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

        return new_df


class Causality:

    def __init__(self, df, dep_var_names=None, indep_var_names=None):
        self.df = df
        self.features_names = self.df.columns.to_list()

        if dep_var_names != None:
            self.dependent_vars = []
            for col in self.df.columns.to_list():
                for name in dep_var_names:
                    if name in col:
                        self.dependent_vars.append(col)
        if indep_var_names != None:
            self.independent_vars = []
            for col in self.df.columns.to_list():
                for name in indep_var_names:
                    if name in col:
                        self.independent_vars.append(col)

        self.structureModel = None
        self.bn = None
        self.ie = None

    def training(self):
        print(f'structuring model...')
        self.structureModel = from_pandas(self.df)
        self.structureModel.remove_edges_below_threshold(0.1)

        " Plot structure "
        plt.figure(dpi=500)
        plt.title(f'Initial {self.structureModel}')
        # nx.draw(self.structureModel, pos=networkx.circular_layout(self.structureModel), with_labels=True, font_size=4, edge_color='orange')
        nx.draw(self.structureModel, with_labels=True, font_size=4, edge_color='orange')
        plt.show()

        print(f'training bayesian network...')
        self.bn = BayesianNetwork(self.structureModel)
        self.bn = self.bn.fit_node_states_and_cpds(self.df)

        bad_nodes = [node for node in self.bn.nodes if not re.match("^[0-9a-zA-Z_]+$", node)]
        if bad_nodes:
            print('Bad nodes: ', bad_nodes)

        self.ie = InferenceEngine(self.bn)

    def counterfactual(self):

        print('do-calculus...')
        before = self.ie.query()
        print(before)

        self.deps = [s for s in self.features_names if 'Nearby' in s or 'Reward' in s or 'Delta' in s or 'Alive' in s]
        self.inds = [s for s in self.features_names if s not in self.deps]

        arrays = []
        for feat_ind in self.inds:
            arrays.append(self.df[feat_ind].unique())
        comb_inds = list(product(*arrays))

        for comb_n in comb_inds:
            print('\n')
            for var_ind in range(len(self.inds)):
                try:
                    self.ie.do_intervention(self.inds[var_ind], int(comb_n[var_ind]))
                    print(f'{self.inds[var_ind]} = {int(comb_n[var_ind])}')
                except:
                    print(f'no {self.inds[var_ind]} = {int(comb_n[var_ind])}')
                    pass

            after = self.ie.query()
            for var_dep in self.deps:
                # print(f'{var_dep}) {after[var_dep]}')
                max_key, max_value = max(after[var_dep].items(), key=lambda x: x[1])
                print(f'{var_dep}) -> {max_key}: {max_value}')

            for var_ind in range(len(self.inds)):
                self.ie.reset_do(self.inds[var_ind])


def get_CausaltablesXY(causal_table, rows, cols):
    rows_table = causal_table.index.to_list()
    cols_agents_actions = [s for s in causal_table.columns.to_list() if 'Agent' in s]
    actions = len(cols_agents_actions)
    Causal_TableX = np.zeros((rows, actions))
    Causal_TableY = np.zeros((cols, actions))

    rows_agents_actionsX_consequences = [s for s in rows_table if 'StateX' in s and 'Agent' in s]
    rows_agents_actionsY_consequences = [s for s in rows_table if 'StateY' in s and 'Agent' in s]

    for stateX in range(rows):
        for stateY in range(cols):
            for col_agent_action in cols_agents_actions:
                if rows_agents_actionsX_consequences != []:
                    value_most_probX = 0
                    for row_actX in rows_agents_actionsX_consequences:
                        valX = causal_table.loc[row_actX, col_agent_action]
                        if valX >= value_most_probX:
                            value_most_probX = valX
                            ind_for_cutting = row_actX.index('val')
                            action_most_probX_in = row_actX.replace(row_actX[:ind_for_cutting + 3], "")
                            if action_most_probX_in.find('_') != -1:
                                action_most_probX = -int(action_most_probX_in.replace('_', ''))
                            else:
                                action_most_probX = int(action_most_probX_in)
                else:
                    action_most_probX = 0

                if 0 <= stateX + action_most_probX < cols - 1:
                    update_stateX = stateX + action_most_probX
                else:
                    update_stateX = stateX

                if rows_agents_actionsY_consequences != []:
                    value_most_probY = 0
                    for row_actY in rows_agents_actionsY_consequences:
                        valY = causal_table.loc[row_actY, col_agent_action]
                        if valY >= value_most_probY:
                            value_most_probY = valY
                            ind_for_cutting = row_actY.index('val')
                            action_most_probY = row_actY.replace(row_actY[:ind_for_cutting + 3], "")
                            if action_most_probY.find('_') != -1:
                                action_most_probY = -int(action_most_probY.replace('_', ''))
                            else:
                                action_most_probY = int(action_most_probY)
                else:
                    action_most_probY = 0

                if 0 <= stateY + action_most_probY < rows - 1:
                    update_stateY = stateY + action_most_probY
                else:
                    update_stateY = stateY

                Causal_TableX[stateX, cols_agents_actions.index(col_agent_action)] = update_stateX
                Causal_TableY[stateY, cols_agents_actions.index(col_agent_action)] = update_stateY

    return Causal_TableX, Causal_TableY


""" ************************************************************************************************************* """
" Dataframe "
n_goals = 0
obj_minigame = MiniGame(n_goals=n_goals)  # grid 3x3, one agent, one enemy
df, dep_var_names, indep_var_names = obj_minigame.create_df(n_episodes=2000, if_binary_df=False)
df.to_pickle(f'sample_df_causality_{n_goals}goals.pkl')

" Causal Model "
# df = pd.read_pickle(f'sample_df_causality_{n_goals}goals.pkl')
df = df.drop('Reward_Agent0', axis=1)
df = df.drop('Alive_Agent0', axis=1)
causal_model = Causality(df, None, None)
causal_model.training()
causal_model.counterfactual()




