import numpy as np
import random
import global_variables
import os
import json
from scripts.utils.environment import CustomEnv
from scripts.utils.train_models import Training
from scripts.algorithms.causal_discovery import CausalDiscovery

"""
We develop this assessment to validate our hypothesis concerning the extraction of causal relationships. Our 
hypothesis posits that there arises an issue when an agent, operating in both offline and online contexts, 
selects an action that leads it to collide with a wall or boundary of the environment. While this aspect does not 
directly impact reinforcement learning (RL) algorithms, it significantly affects the causal discovery process. Here, 
the action value itself is not the stopping action, but rather, the variables DeltaX and DeltaY both have a 
value of zero. Consequently, this introduces erroneous information into both the causal graph and the causal table.

We believe this problem is environment-dependent. In wider environments, this issue exists but holds lesser 
significance, whereas in smaller environments, it becomes considerably more relevant.

As a result of our simulations, we present final causal graphs and causal tables across various environmental 
configurations; all simulations are conducted with 1 agent, 1 enemy and 1 goal in a grid-like world environments of 
different sizes. A high number of training episodes has been selected for this assessment to ensure thorough 
exploration of the entire environment by the agent and to guarantee the comprehensive development of the causal 
table, capturing all possible dependencies within the specified episodes.

Additionally, if we consider a square grid with dimensions rows * cols, the number of 'central' cells is (rows-2)^2. 
For clarity, consider the following example:
    - grid 3x3 --> 1 central cell, 8 boundary cells 
    - grid 4x4 --> 4 central cells, 12 boundary cells 
    - grid 8x8 --> 36 central cells, 28 boundary cells
"""

NAME_DIR_RESULTS_GRID = f'{global_variables.GLOBAL_PATH_REPO}/Results/Test1'
NAME_DIR_RESULTS_TOR_GRID = f'{global_variables.GLOBAL_PATH_REPO}/Results/Test2_new'

N_SIMULATIONS = global_variables.N_SIMULATIONS_PAPER
N_TRAINING_EPISODE = global_variables.N_TRAINING_EPISODES
N_AGENTS = 1
N_ENEMIES = 1
N_GOALS = 1
GRID_SIZES = [(3, 3), (4, 4), (5, 5), (8, 8), (10, 10)]

label_kind_of_alg = global_variables.LABEL_RANDOM_AGENT
label_exploration_strategy = global_variables.LABEL_RANDOM_AGENT

os.makedirs(NAME_DIR_RESULTS_GRID, exist_ok=True)
DIR_SAVE_RESULTS_GRID = f'{NAME_DIR_RESULTS_GRID}'

os.makedirs(NAME_DIR_RESULTS_TOR_GRID, exist_ok=True)
DIR_SAVE_RESULTS_TOR_GRID = f'{NAME_DIR_RESULTS_TOR_GRID}'


def generate_empty_list(X: int, data_type) -> list:
    return [data_type() for _ in range(X)]


def make_toroidal(df_input, n_agents, n_enemies, n_goals):
    new_df = df_input.copy()
    cols_new_df = new_df.columns.to_list()
    cols_actions, cols_DeltaX, cols_DeltaY, cols_GoalNearby, cols_EnemyNearby, cols_rewards = [], [], [], [], [], []
    for ag in range(n_agents):
        cols_actions.append([s for s in cols_new_df if
                             global_variables.LABEL_COL_ACTION in s and f'{global_variables.LABEL_AGENT_CAUSAL_TABLE}{ag}' in s][
                                0])
        cols_DeltaX.append([s for s in cols_new_df if
                            global_variables.LABEL_COL_DELTAX in s and f'{global_variables.LABEL_AGENT_CAUSAL_TABLE}{ag}' in s][
                               0])
        cols_DeltaY.append([s for s in cols_new_df if
                            global_variables.LABEL_COL_DELTAY in s and f'{global_variables.LABEL_AGENT_CAUSAL_TABLE}{ag}' in s][
                               0])

        cols_rewards.append([s for s in cols_new_df if
                             global_variables.LABEL_COL_REWARD in s and f'{global_variables.LABEL_AGENT_CAUSAL_TABLE}{ag}' in s][
                                0])

        for goal in range(n_goals):
            cols_GoalNearby.append([s for s in cols_new_df if f'Goal{goal}' in s and
                                    global_variables.LABEL_NEARBY_CAUSAL_TABLE in s and f'{global_variables.LABEL_AGENT_CAUSAL_TABLE}{ag}' in s][
                                       0])
        for enemy in range(n_enemies):
            cols_EnemyNearby.append([s for s in cols_new_df if f'Enemy{enemy}' in s and
                                     global_variables.LABEL_NEARBY_CAUSAL_TABLE in s and f'{global_variables.LABEL_AGENT_CAUSAL_TABLE}{ag}' in s][
                                        0])

    for index, row in new_df.iterrows():
        for ag in range(n_agents):
            if row[cols_actions[ag]] != 0 and (row[cols_DeltaX[ag]] == 0 and row[cols_DeltaY[ag]] == 0):
                #print('\n*****Past', row)

                for en in range(n_enemies):
                    if row[cols_rewards[ag]] == global_variables.VALUE_REWARD_LOSER_PAPER:
                        new_df.at[index, cols_EnemyNearby[en]] = row[cols_actions[ag]]
                    else:
                        new_df.at[index, cols_EnemyNearby[en]] = global_variables.VALUE_ENTITY_FAR

                for goal in range(n_goals):
                    if row[cols_rewards[ag]] == global_variables.VALUE_REWARD_WINNER_PAPER:
                        new_df.at[index, cols_GoalNearby[goal]] = row[cols_actions[ag]]
                    else:
                        new_df.at[index, cols_GoalNearby[goal]] = global_variables.VALUE_ENTITY_FAR

                deltaY, deltaX = global_variables.DICT_IMPLEMENTED_ACTIONS[row[cols_actions[ag]]]
                new_df.at[index, cols_DeltaX[ag]] = deltaX
                new_df.at[index, cols_DeltaY[ag]] = deltaY

                #print('New: ', new_df.loc[index, :])
    return new_df


for rows, cols in GRID_SIZES:

    dict_to_save_grid = {'grid_size': (rows, cols), 'n_enemies': N_ENEMIES, 'n_episodes': N_TRAINING_EPISODE,
                         'env': generate_empty_list(N_SIMULATIONS, list),
                         'causal_graph': generate_empty_list(N_SIMULATIONS, list),
                         'df_track': generate_empty_list(N_SIMULATIONS, list)}

    dict_to_save_tor_grid = {'grid_size': (rows, cols), 'n_enemies': N_ENEMIES, 'n_episodes': N_TRAINING_EPISODE,
                             'env': generate_empty_list(N_SIMULATIONS, list),
                             'causal_graph': generate_empty_list(N_SIMULATIONS, list),
                             'df_track': generate_empty_list(N_SIMULATIONS, list)}

    for simulation_n in range(N_SIMULATIONS):

        seed_value = global_variables.seed_values[simulation_n]
        np.random.seed(seed_value)
        random.seed(seed_value)

        dict_learning_params = global_variables.DICT_LEARNING_PARAMETERS_PAPER
        dict_other_params = global_variables.DICT_OTHER_PARAMETERS_PAPER

        dict_env_params = {'rows': rows, 'cols': cols, 'n_agents': N_AGENTS, 'n_enemies': N_ENEMIES, 'n_goals': N_GOALS,
                           'n_actions': global_variables.N_ACTIONS_PAPER, 'if_maze': False,
                           'value_reward_alive': global_variables.VALUE_REWARD_ALIVE_PAPER,
                           'value_reward_winner': global_variables.VALUE_REWARD_WINNER_PAPER,
                           'value_reward_loser': global_variables.VALUE_REWARD_LOSER_PAPER,
                           'seed_value': seed_value, 'enemies_actions': 'random', 'env_type': 'numpy',
                           'predefined_env': None}

        dict_other_params['N_EPISODES'] = N_TRAINING_EPISODE

        env = CustomEnv(dict_env_params)
        env_to_save = np.vectorize(lambda x: env.number_names_grid.get(x, str(x)))(env.grid_for_game)
        dict_to_save_grid['env'][simulation_n] = env_to_save.tolist()
        dict_to_save_tor_grid['env'][simulation_n] = env_to_save.tolist()

        class_train = Training(dict_env_params, dict_learning_params, dict_other_params,
                               f'{label_kind_of_alg}',
                               f'{label_exploration_strategy}')
        class_train.start_train(env, batch_update_df_track=1000)
        df_track = class_train.get_df_track()

        for if_tor_grid in [True]:

            if if_tor_grid:
                toroidal_df_track = make_toroidal(df_track, n_agents=N_AGENTS, n_enemies=N_ENEMIES, n_goals=N_GOALS)
                dict_to_save_tor_grid['df_track'][simulation_n] = toroidal_df_track.to_dict(orient='records')

                cd = CausalDiscovery(toroidal_df_track, N_AGENTS, N_ENEMIES, N_GOALS)
                causal_graph = cd.return_causal_graph()
                dict_to_save_tor_grid['causal_graph'][simulation_n] = causal_graph

            else:
                dict_to_save_grid['df_track'][simulation_n] = df_track.to_dict(orient='records')

                cd = CausalDiscovery(df_track, N_AGENTS, N_ENEMIES, N_GOALS)
                causal_graph = cd.return_causal_graph()
                dict_to_save_grid['causal_graph'][simulation_n] = causal_graph

    #with open(f'{DIR_SAVE_RESULTS_GRID}/results_Grid{rows}x{cols}_{N_ENEMIES}enemies_{N_TRAINING_EPISODE}episodes.json',
              #'w') as json_file:
        #json.dump(dict_to_save_grid, json_file)

    with open(
            f'{DIR_SAVE_RESULTS_TOR_GRID}/results_TorGrid{rows}x{cols}_{N_ENEMIES}enemies_{N_TRAINING_EPISODE}episodes.json',
            'w') as json_file:
        json.dump(dict_to_save_tor_grid, json_file)
