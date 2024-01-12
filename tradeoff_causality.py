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

warnings.filterwarnings("ignore")


class MiniGame:

    def __init__(self, rows, cols, n_agents, n_enemies, n_goals):
        self.rows = rows
        self.cols = cols
        self.n_agents = n_agents
        self.n_actions = 5
        self.n_enemies = n_enemies
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

    def create_df(self, n_episodes, if_binary_df=False):

        print('\ncreating dataframe...')
        time.sleep(1)

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

        self.df = self.df.drop(['Alive_Agent0', 'GameOver_Agent0', 'Winner_Agent0'], axis=1)

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

    def __init__(self, df):
        self.df = df
        self.features_names = self.df.columns.to_list()

        self.structureModel = None
        self.bn = None
        self.ie = None
        self.independents_var = None
        self.dependents_var = None

    def training(self):
        print(f'structuring model...', len(self.df))
        self.structureModel = from_pandas(self.df)
        self.structureModel.remove_edges_below_threshold(0.2)

        " Plot structure "
        """plt.figure(dpi=500)
        plt.title(f'{self.structureModel} - {n_episodes} episodes - Grid {cols}x{rows}')
        nx.draw(self.structureModel, pos=nx.circular_layout(self.structureModel), with_labels=True, font_size=6,
                edge_color='orange')
        # nx.draw(self.structureModel, with_labels=True, font_size=4, edge_color='orange')
        plt.show()"""

        print(f'training bayesian network...')
        self.bn = BayesianNetwork(self.structureModel)
        self.bn = self.bn.fit_node_states_and_cpds(self.df)

        bad_nodes = [node for node in self.bn.nodes if not re.match("^[0-9a-zA-Z_]+$", node)]
        if bad_nodes:
            print('Bad nodes: ', bad_nodes)

        self.ie = InferenceEngine(self.bn)

        self.dependents_var = []
        self.independents_var = []

        print('do-calculus-1...')
        " understand who influences whom "
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

        # print(f'**Externally caused: {self.independents_var}')
        # print(f'**Externally influenced: {self.dependents_var}')
        print('do-calculus-2...')
        causal_table = pd.DataFrame(columns=self.features_names)

        arrays = []
        for feat_ind in self.dependents_var:
            arrays.append(self.df[feat_ind].unique())
        var_combinations = list(product(*arrays))

        for n, comb_n in enumerate(var_combinations):
            # print('\n')
            for var_ind in range(len(self.dependents_var)):
                try:
                    self.ie.do_intervention(self.dependents_var[var_ind], int(comb_n[var_ind]))
                    # print(f'{self.dependents_var[var_ind]} = {int(comb_n[var_ind])}')
                    causal_table.at[n, f'{self.dependents_var[var_ind]}'] = int(comb_n[var_ind])
                except:
                    # print(f'no {self.dependents_var[var_ind]} = {int(comb_n[var_ind])}')
                    causal_table.at[n, f'{self.dependents_var[var_ind]}'] = pd.NA

            after = self.ie.query()
            for var_dep in self.independents_var:
                # print(f'{var_dep}) {after[var_dep]}')
                max_key, max_value = max(after[var_dep].items(), key=lambda x: x[1])
                if round(max_value, 4) != round(1 / len(after[var_dep]), 4):
                    causal_table.at[n, f'{var_dep}'] = int(max_key)
                    # print(f'{var_dep}) -> {max_key}: {max_value}')
                else:
                    causal_table.at[n, f'{var_dep}'] = pd.NA
                    # print(f'{var_dep}) -> unknown')

            for var_ind in range(len(self.dependents_var)):
                self.ie.reset_do(self.dependents_var[var_ind])

        return causal_table


def process_df(df_start):
    start_columns = df_start.columns.to_list()
    n_enemies_columns = [s for s in start_columns if 'Enemy' in s]
    if n_enemies_columns == 1:
        return df_start
    else:
        df_only_nearbies = df_start[n_enemies_columns]

        new_column = []
        for episode in range(len(df_start)):
            single_row = df_only_nearbies.loc[episode].tolist()

            if df_start.loc[episode, 'Reward_Agent0'] == -1:
                enemy_nearbies_true = [s for s in single_row if s != 50]
                action_agent = df_start.loc[episode, 'Action_Agent0']

                if action_agent in enemy_nearbies_true:
                    new_column.append(action_agent)
                else:
                    new_column.append(50)
            else:
                new_column.append(random.choice(single_row))

        df_out = df_start.drop(columns=n_enemies_columns)

        df_out['Enemy0_Nearby_Agent0'] = new_column

        return df_out


""" ************************************************************************************************************* """
" EVALUATION ENVIRONMENT AND NUMBER OF EPISODES NEEDED"
" Dataframe "
path_save = 'TradeOff_BatchEpisodesEnemies_Causality'
os.makedirs(path_save, exist_ok=True)


def are_dataframes_equal(df1, df2):
    # Sort DataFrames by values
    sorted_df1 = df1.sort_values(by=['Enemy0_Nearby_Agent0', 'Goal0_Nearby_Agent0', 'Action_Agent0']).reset_index(
        drop=True)
    sorted_df2 = df2.sort_values(by=['Enemy0_Nearby_Agent0', 'Goal0_Nearby_Agent0', 'Action_Agent0']).reset_index(
        drop=True)

    # Check if the sorted DataFrames are equal
    return sorted_df1.equals(sorted_df2)


n_simulations = 10
official_causal_table = pd.read_pickle('heuristic_table.pkl')
vector_episodes = [250, 500, 1000]
vector_grid_size = [5, 10]
vector_n_enemies = [2, 5, 10]
columns = ['Grid Size', 'Episodes', 'Suitable']
result = pd.DataFrame(columns=columns)

df_row = 0
for n_episodes in vector_episodes:
    for rows in vector_grid_size:
        for n_enemies in vector_n_enemies:
            if n_enemies > rows*2:
                break
            cols = rows
            n_oks = 0
            print(f'\n Grid size: {rows}x{cols} - {n_episodes} episodes')
            for sim_n in range(n_simulations):
                obj_minigame = MiniGame(rows=rows, cols=cols, n_agents=1, n_enemies=n_enemies, n_goals=1)
                df = obj_minigame.create_df(n_episodes=n_episodes)

                " Causal Model "
                df = process_df(df)
                causality = Causality(df)
                causal_table = causality.training()
                causal_table.dropna(axis=0, how='any', inplace=True)

                if are_dataframes_equal(causal_table, official_causal_table):
                    n_oks += 1
                    print('ok')
                else:
                    causal_table.to_excel(f'{path_save}\\{rows}x{cols}_{n_enemies}enemies_{n_episodes}episodes_{sim_n}.xlsx')
                    print('no')

            result.at[df_row, 'Grid Size'] = rows
            result.at[df_row, 'Episodes'] = n_episodes
            result.at[df_row, 'Enemies'] = n_enemies

            if n_oks > int(n_simulations/2):
                result.at[df_row, 'Suitable'] = 'yes'
            else:
                result.at[df_row, 'Suitable'] = 'no'

            df_row += 1

result.to_excel(f'{path_save}\\comparison_causality.xlsx')

""" ************************************************************************************************************* """
" Grid size and number of episodes already chosen"
" Dataframe "
"""obj_minigame = MiniGame(rows=3, cols=3, n_agents=1, n_enemies=1, n_goals=1)
df = obj_minigame.create_df(n_episodes=2000)

causality = Causality(df)
causal_table = causality.training()
causal_table.dropna(axis=0, how='any', inplace=True)
causal_table.to_pickle('heuristic_table.pkl')"""
