import os
import random
import json
from itertools import product
import numpy as np
import pandas as pd
import global_variables
from scripts.algorithms.causal_discovery import CausalDiscovery
from scripts.utils.environment import CustomEnv
from scripts.utils.others import compare_causal_graphs
from scripts.utils.train_models import Training

""" 
The objective of this simulation is to address the inherent tradeoff within the algorithm involving online causal
inference. Specifically, our aim is to determine the optimal number of episodes needed to reliably generate the 
correct causal table for varying numbers of enemies and grid sizes. 
 
The simulation will yield a recommended set of 'batch' episodes for utilization in algorithms incorporating online 
causal discovery. This quantity will vary depending on factors such as grid size and the count of adversaries within 
the environment."""


def generate_empty_list(X: int, data_type) -> list:
    return [data_type() for _ in range(X)]


"""
def prepare_df_for_comparison(df1: pd.DataFrame) -> pd.DataFrame:
    col_df1 = df1.columns.to_list()
    # Sort DataFrames by values
    sorted_df1 = df1.sort_index(axis=0).sort_index(axis=1).reset_index(drop=True)

    new_col_df1_to_drop = [s for s in sorted_df1.columns.to_list() if s not in col_df1]

    check_df1 = sorted_df1.drop(columns=new_col_df1_to_drop)

    return check_df1


GROUND_TRUTH_CAUSAL_TABLE = prepare_df_for_comparison(pd.read_pickle(f'{global_variables.PATH_CAUSAL_TABLE_OFFLINE}'))
"""
with open(f'{global_variables.PATH_CAUSAL_GRAPH_OFFLINE}', 'r') as file:
    GROUND_TRUTH_CAUSAL_GRAPH = json.load(file)

N_SIMULATIONS_CONSIDERED = global_variables.N_SIMULATIONS_PAPER
N_ENEMIES_CONSIDERED = global_variables.N_ENEMIES_CONSIDERED_PAPER
N_EPISODES_CONSIDERED = global_variables.N_EPISODES_CONSIDERED_FOR_SENSITIVE_ANALYSIS_PAPER
GRID_SIZES_CONSIDERED = global_variables.GRID_SIZES_CONSIDERED_PAPER
n_agents = 1
n_goals = 1

dict_learning_params = global_variables.DICT_LEARNING_PARAMETERS_PAPER
dict_other_params = global_variables.DICT_OTHER_PARAMETERS_PAPER

label_kind_of_alg = global_variables.LABEL_RANDOM_AGENT
label_exploration_strategy = 'random'

" First part: simulations "
combinations_enemies_episodes_grid = list(product(N_ENEMIES_CONSIDERED, N_EPISODES_CONSIDERED, GRID_SIZES_CONSIDERED))
list_combinations_for_simulations = [{'n_enemies': item[0], 'n_episodes': item[1], 'grid_size': item[2]} for item in
                                     combinations_enemies_episodes_grid]

for dict_comb in list_combinations_for_simulations:
    n_enemies = dict_comb['n_enemies']
    n_episodes = dict_comb['n_episodes']
    rows, cols = dict_comb['grid_size']
    print(f'\n *** Grid size: {rows}x{cols} - {n_episodes} episodes - {n_enemies} enemies ***')

    # dict_comb['df'] = generate_empty_list(N_SIMULATIONS_CONSIDERED, pd.DataFrame)
    # dict_comb['causal_table'] = generate_empty_list(N_SIMULATIONS_CONSIDERED, pd.DataFrame)
    dict_comb['causal_graph'] = generate_empty_list(N_SIMULATIONS_CONSIDERED, list)

    for sim_n in range(N_SIMULATIONS_CONSIDERED):
        seed_value = global_variables.seed_values[sim_n]
        np.random.seed(seed_value)
        random.seed(seed_value)
        dict_env_params = {'rows': rows, 'cols': cols, 'n_agents': n_agents, 'n_enemies': n_enemies,
                           'n_goals': n_goals,
                           'n_actions': global_variables.N_ACTIONS_PAPER,
                           'if_maze': False,
                           'value_reward_alive': global_variables.VALUE_REWARD_ALIVE_PAPER,
                           'value_reward_winner': global_variables.VALUE_REWARD_WINNER_PAPER,
                           'value_reward_loser': global_variables.VALUE_REWARD_LOSER_PAPER,
                           'seed_value': seed_value, 'enemies_actions': 'random', 'env_type': 'numpy',
                           'predefined_env': None}

        dict_other_params['N_EPISODES'] = n_episodes

        env = CustomEnv(dict_env_params)

        class_train = Training(dict_env_params, dict_learning_params, dict_other_params,
                               f'{label_kind_of_alg}',
                               f'{label_exploration_strategy}')

        class_train.start_train(env, batch_update_df_track=1000)

        df_track = class_train.get_df_track()
        # dict_comb['df'][sim_n] = df_track

        cd = CausalDiscovery(df_track, n_agents, n_enemies, n_goals)
        out_causal_graph = cd.return_causal_graph()
        # out_causal_table = cd.return_causal_table()

        # dict_comb['causal_table'][sim_n] = out_causal_table
        dict_comb['causal_graph'][sim_n] = out_causal_graph

os.makedirs(f'{global_variables.GLOBAL_PATH_REPO}/Results/Sensitive_Analysis_Batch_Episodes', exist_ok=True)
with open(f'{global_variables.GLOBAL_PATH_REPO}/Results/Sensitive_Analysis_Batch_Episodes/batch_episodes_for_online_cd_values.json', 'w') as json_file:
    json.dump(list_combinations_for_simulations, json_file)


