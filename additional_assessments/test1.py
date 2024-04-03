import numpy as np
import global_variables
import os
import json
from scripts.algorithms.causal_discovery import CausalDiscovery
from scripts.utils.environment import CustomEnv
from scripts.utils.train_models import Training

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

NAME_DIR_RESULTS = f'{global_variables.GLOBAL_PATH_REPO}/Results/Test1'

N_SIMULATIONS = global_variables.N_SIMULATIONS_PAPER
N_TRAINING_EPISODE = global_variables.N_TRAINING_EPISODES
N_AGENTS = 1
N_ENEMIES = 1
N_GOALS = 1
GRID_SIZES = [(3, 3), (4, 4), (6, 6), (8, 8), (10, 10)]

label_kind_of_alg = global_variables.LABEL_RANDOM_AGENT
label_exploration_strategy = global_variables.LABEL_RANDOM_AGENT

# SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
os.makedirs(NAME_DIR_RESULTS, exist_ok=True)
DIR_SAVE_RESULTS = f'{NAME_DIR_RESULTS}'

for rows, cols in GRID_SIZES:
    dir_save = f'{DIR_SAVE_RESULTS}/Grid{rows}x{cols}'
    os.makedirs(dir_save, exist_ok=True)

    for simulation_n in range(N_SIMULATIONS):
        name_save = f'{rows}x{cols}_game{simulation_n}'

        seed_value = global_variables.seed_values[simulation_n]

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
        with open(f'{dir_save}/env_{name_save}.json', 'w') as json_file:
            json.dump(env_to_save.tolist(), json_file)

        class_train = Training(dict_env_params, dict_learning_params, dict_other_params,
                               f'{label_kind_of_alg}',
                               f'{label_exploration_strategy}')
        class_train.start_train(env, batch_update_df_track=1000)
        df_track = class_train.get_df_track()

        out_causal_table = CausalDiscovery(df_track, N_AGENTS, N_ENEMIES, N_GOALS, dir_save,
                                           f'graph_{name_save}').return_causal_table()

        if out_causal_table is not None:
            out_causal_table.to_excel(f'{dir_save}/causal_table_{name_save}.xlsx')
            out_causal_table.to_pickle(f'{dir_save}/causal_table_{name_save}.pkl')
