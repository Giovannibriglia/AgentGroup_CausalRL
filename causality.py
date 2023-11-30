import random
import os
import re
import time
import operator

from pyAgrum import pyAgrum
from tqdm import tqdm
import dowhy
import matplotlib.pyplot as plt
import networkx
import numpy as np
import pandas as pd
from causalnex.inference import InferenceEngine
from causalnex.network import BayesianNetwork
from causalnex.structure import StructureModel
from causalnex.structure.notears import from_pandas
import warnings
from scipy.stats import chi2_contingency, cramervonmises_2samp

warnings.filterwarnings("ignore")

visualization = False


class CreateDf:

    def __init__(self, n_goals):
        self.rows = 4
        self.cols = 4
        self.n_agents = 1
        self.n_actions = 5
        self.n_enemies = 1
        self.n_goals = n_goals

        # predefined action state
        self.predefined_delta_state = 50

        # Action_Agent..,  DeltaX_Agent.., DeltaY_Agent.., GameOver_Agent.., Winner_Agent.., Enemy.._Nearby_Agent..
        self.cols_df = []
        for agent in range(self.n_agents):
            self.cols_df.append(f'Action_Agent{agent}')
            self.cols_df.append(f'DeltaX_Agent{agent}')
            self.cols_df.append(f'DeltaY_Agent{agent}')
            if self.n_enemies > 0:
                self.cols_df.append(f'GameOver_Agent{agent}')
            if self.n_goals > 0:
                self.cols_df.append(f'Winner_Agent{agent}')
            for enemy in range(self.n_enemies):
                self.cols_df.append(f'Enemy{enemy}_Nearby_Agent{agent}')
            for goal in range(self.n_goals):
                self.cols_df.append(f'Goal{goal}_Nearby_Agent{agent}')

        self.df = pd.DataFrame(columns=self.cols_df)

    def create_df(self, n_episodes, if_binary_df=True):

        print('creating dataframe...')
        time.sleep(2)

        for e in tqdm(range(n_episodes)):
            done = False
            while not done:
                if e == 0:
                    agents_coord, enemies_coord, goals_coord = self.create_env()

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
                        self.df.at[e, f'Enemy{enemy}_Nearby_Agent{agent}'] = nearby_distance

                # goal nearbies direction
                for goal in range(self.n_goals):
                    x_goal = goals_coord[goal][0]
                    y_goal = goals_coord[goal][1]
                    for agent in range(self.n_agents):
                        x_ag = agents_coord[agent][0]
                        y_ag = agents_coord[agent][1]

                        nearby_distance = self.get_direction(x_ag, y_ag, x_goal, y_goal)
                        self.df.at[e, f'Goal{goal}_Nearby_Agent{agent}'] = nearby_distance

                # agents movement
                for agent in range(self.n_agents):
                    lastX_ag = agents_coord[agent][0]
                    lastY_ag = agents_coord[agent][1]
                    newX_ag, newY_ag, action_ag, deltaX_ag, deltaY_ag = self.get_action(lastX_ag, lastY_ag)
                    agents_coord[agent][0] = newX_ag
                    agents_coord[agent][1] = newY_ag

                    self.df.at[e, f'Action_Agent{agent}'] = action_ag
                    self.df.at[e, f'DeltaX_Agent{agent}'] = deltaX_ag
                    self.df.at[e, f'DeltaY_Agent{agent}'] = deltaY_ag

                # check game over, winner, nothing
                done = self.check_gameover_or_winner(agents_coord, enemies_coord, goals_coord, e)

        if if_binary_df:
            df_bin = self.binarize_dataframe()
            for col in df_bin.columns:
                df_bin[str(col)] = df_bin[str(col)].astype(str).str.replace(',', '').astype(float)
            return df_bin
        else:
            for col in self.df.columns:
                self.df[str(col)] = self.df[str(col)].astype(str).str.replace(',', '').astype(float)
            return self.df

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

        direction_ag_en = -1

        if deltaX == 0 and deltaY == 0:  # stop
            direction_ag_en = 0
        elif deltaX == 1 and deltaY == 0:  # right
            direction_ag_en = 1
        elif deltaX == -1 and deltaY == 0:  # left
            direction_ag_en = 2
        elif deltaX == 0 and deltaY == 1:  # up
            direction_ag_en = 3
        elif deltaX == 0 and deltaY == -1:  # down
            direction_ag_en = 4
        elif deltaX == 1 and deltaY == 1 and self.n_actions > 5:  # diag up right
            direction_ag_en = 5
        elif deltaX == 1 and deltaY == -1 and self.n_actions > 5:  # diag down right
            direction_ag_en = 6
        elif deltaX == -1 and deltaY == 1 and self.n_actions > 5:  # diag up left
            direction_ag_en = 7
        elif deltaX == -1 and deltaY == -1 and self.n_actions > 5:  # diag down left
            direction_ag_en = 8
        else:  # otherwise
            direction_ag_en = 50

        # print([x_ag, y_ag], [x_en, y_en], direction_ag_en)

        return direction_ag_en

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

    def check_gameover_or_winner(self, agents_coord, enemies_coord, goals_coord, e):
        done = False
        for agent in range(self.n_agents):
            x_ag = agents_coord[agent][0]
            y_ag = agents_coord[agent][1]

            if self.n_goals > 0:
                for goal in range(self.n_goals):
                    x_goal = goals_coord[goal][0]
                    y_goal = goals_coord[goal][1]

                    if [x_ag, y_ag] == [x_goal, y_goal]:
                        self.df.at[e, f'Winner_Agent{agent}'] = 1
                        done = True
                        break
                    else:
                        self.df.at[e, f'Winner_Agent{agent}'] = 0

            if self.n_enemies > 0:
                for enemy in range(self.n_enemies):
                    x_en = enemies_coord[enemy][0]
                    y_en = enemies_coord[enemy][1]

                    if [x_ag, y_ag] == [x_en, y_en]:
                        self.df.at[e, f'GameOver_Agent{agent}'] = 1
                        done = True
                        break
                    else:
                        self.df.at[e, f'GameOver_Agent{agent}'] = 0

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
                    if bin > 0:
                        labels.append('Value' + str(bin))
                    else:
                        labels.append('Value_' + str(abs(bin)))

                binary_df = pd.get_dummies(pd.cut(self.df[col], bins=bins, labels=labels), prefix=f'{col}')

                new_df = pd.concat([new_df, binary_df], axis=1)

        return new_df


class Causality:

    def __init__(self, df):
        self.df = df
        self.features_names = self.df.columns.to_list()
        self.children_names = ['Action', 'Nearby']
        self.structureModel = None
        self.bn = None
        self.ie = None


    def training(self):
        print(f'structuring model...')

        children = []
        for col in self.df.columns.to_list():
            for par_name in self.children_names:
                if par_name in col:
                    children.append(col)

        self.structureModel = from_pandas(self.df, tabu_child_nodes=children)
        self.structureModel.remove_edges_below_threshold(0.3)

        " Plot structure "
        plt.figure(dpi=500)
        plt.title(f'Initial {self.structureModel}')
        networkx.draw(self.structureModel, pos=networkx.spring_layout(self.structureModel), with_labels=True, font_size=4, edge_color='orange')
        # networkx.draw(self.structureModel, with_labels=True, font_size=4, edge_color='orange')
        plt.show()

        features_work = self.structureModel.nodes

        print(f'training bayesian network...')
        self.bn = BayesianNetwork(self.structureModel)
        self.bn = self.bn.fit_node_states_and_cpds(self.df)

        bad_nodes = [node for node in self.bn.nodes if not re.match("^[0-9a-zA-Z_]+$", node)]
        if bad_nodes:
            print('Bad nodes: ', bad_nodes)

        """evidence = {'Node1': 0, 'Node2': 1}

        # Create an inference engine
        ie = pyAgrum.LazyPropagation(self.bn)

        # Set evidence
        for node, value in evidence.items():
            ie.setEvidence(self.bn.idFromName(node), value)

        # Perform inference
        ie.makeInference()

        # Get the posterior probabilities
        posterior_probs = ie.posterior(self.bn.idFromName('TargetNode'))"""

        print(self.bn.cpds)

        self.ie = InferenceEngine(self.bn)

    def counterfactual(self):
        print('do-calculus...')
        print(self.ie.query())
        print(self.ie.query()['GameOver_Agent0'])

        # self.ie.do_intervention('Enemy0_Nearby_Agent0_Value1', True)

        self.ie.do_intervention('Action_Agent0_Value1', True)

        print(self.ie.query()['GameOver_Agent0'])
        return 0


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
"""create = CreateDf(n_goals=1)  # grid 4x4, one agent, one enemy, no goal
df = create.create_df(n_episodes=10000)
df.to_pickle('sample_df_causality_1goals.pkl')"""

" Causal Model "
df = pd.read_pickle('sample_df_causality_0goals.pkl')
# print(df.columns)
causal_model = Causality(df)
causal_model.training()
causal_model.counterfactual()

"""cols = df.columns.to_list()
contacts = [s for s in cols if 'Contact' in s]
actions = [s for s in cols if 'Action' in s]
states = [s for s in cols if 'State' in s]
nearbies = [s for s in cols if 'Nearby' in s]
others = [s for s in cols if s not in contacts and s not in actions and s not in states and s not in nearbies]

vets = [contacts, actions, states, nearbies, others]
deps = []
inds = []
for vet in vets:
    for feat1 in vet:
        pvalues = []
        cols_2 = [s for s in cols if s not in contacts]
        for feat2 in cols_2:
            res = cramervonmises_2samp(df[feat1], df[feat2])
            pvalues.append(res.pvalue)
        av = np.mean(pvalues)
        if av < 0.05:
            # print(f'dep) {feat1}: {av}')
            deps.append(feat1)
        else:
            # print(f'ind) {feat1}: {av}')
            inds.append(feat1)

print(f'Dependent features: {deps}')
print(f'Indipendent features: {inds}')

for var_dep in deps:
    cols_new = [s for s in cols if s not in deps]
    for feat3 in cols_new:
        res = cramervonmises_2samp(df[var_dep], df[feat3])
        if res.pvalue > 0.05:
            print(f'{var_dep} -- {feat3}: {res.pvalue}')"""
