import json
import os
import random
from itertools import product

import numpy as np
import pandas as pd

import global_variables
from scripts.algorithms.causal_discovery import CausalDiscovery
from scripts.utils.environment import CustomEnv
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


def select_df(dict_for_df: dict, n_ep) -> pd.DataFrame:
    df = pd.DataFrame(dict_for_df)
    mask = df['episode'] == n_ep

    df = df.drop('episode', axis=1)
    if mask.any():
        first_occurrence_index = mask.idxmax()
        selected_df = df.loc[:first_occurrence_index]
        return selected_df
    else:
        print(f'{n_ep} not founded')
        return df


DIR_SAVING = f'{global_variables.GLOBAL_PATH_REPO}/Results/Sensitive_Analysis_Batch_Episodes_2'
N_SIMULATIONS_CONSIDERED = 1 #global_variables.N_SIMULATIONS_PAPER
N_ENEMIES_CONSIDERED = [2] #global_variables.N_ENEMIES_CONSIDERED_PAPER
N_EPISODES_CONSIDERED = global_variables.N_EPISODES_CONSIDERED_FOR_SENSITIVE_ANALYSIS_PAPER
N_EPISODES_CONSIDERED = [s-1 for s in N_EPISODES_CONSIDERED]
MAX_N_EPISODES = [max(N_EPISODES_CONSIDERED)]
GRID_SIZES_CONSIDERED = [(5, 5)] #global_variables.GRID_SIZES_CONSIDERED_PAPER
n_agents = 1
n_goals = 1

dict_learning_params = global_variables.DICT_LEARNING_PARAMETERS_PAPER
dict_other_params = global_variables.DICT_OTHER_PARAMETERS_PAPER

label_kind_of_alg = f'{global_variables.LABEL_Q_LEARNING}_{global_variables.LABEL_VANILLA}'
label_exploration_strategy = f'{global_variables.LABEL_EPSILON_GREEDY}'

" First part: create df with maximum number of episodes "
combinations_enemies_episodes_grid = list(product(N_ENEMIES_CONSIDERED, MAX_N_EPISODES, GRID_SIZES_CONSIDERED))
list_combinations_for_simulations = [{'n_enemies': item[0], 'n_episodes': item[1], 'grid_size': item[2]} for item in
                                     combinations_enemies_episodes_grid]
os.makedirs(f'{DIR_SAVING}', exist_ok=True)
list_dicts = []
for dict_comb in list_combinations_for_simulations:
    n_enemies = dict_comb['n_enemies']
    n_episodes = dict_comb['n_episodes']
    rows, cols = dict_comb['grid_size']
    print(f'\n *** Grid size: {rows}x{cols} - {n_episodes} episodes - {n_enemies} enemies ***')

    dict_simulations = {'grid_size': (rows, cols),
                        'n_enemies': n_enemies,
                        'n_episodes': n_episodes,
                        'n_simulations': N_SIMULATIONS_CONSIDERED,
                        'envs': generate_empty_list(N_SIMULATIONS_CONSIDERED, list),
                        'dfs_track': generate_empty_list(N_SIMULATIONS_CONSIDERED, pd.DataFrame)}

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

        dict_other_params['N_EPISODES'] = n_episodes+1

        env = CustomEnv(dict_env_params)
        env_to_save = np.vectorize(lambda x: env.number_names_grid.get(x, str(x)))(env.grid_for_game)
        dict_simulations['envs'][sim_n] = env_to_save

        class_train = Training(dict_env_params, dict_learning_params, dict_other_params,
                               f'{label_kind_of_alg}',
                               f'{label_exploration_strategy}')

        class_train.start_train(env, batch_update_df_track=MAX_N_EPISODES[0])

        df_track = class_train.df_track
        dict_simulations['dfs_track'][sim_n] = df_track.to_dict(orient='records')

    list_dicts.append(dict_simulations)

" Second part: perform causal discovery until the episode required and store results "

for single_dict in list_dicts:
    n_enemies = single_dict['n_enemies']
    max_n_episodes_df = single_dict['n_episodes']
    rows, cols = single_dict['grid_size']
    envs = single_dict['envs']
    dfs_track = single_dict['dfs_track']

    for n_episodes_cd in N_EPISODES_CONSIDERED:
        dict_to_save = {'n_enemies': n_enemies,
                        'grid_size': (rows, cols),
                        'n_episodes': n_episodes_cd,
                        'dfs_track': generate_empty_list(N_SIMULATIONS_CONSIDERED, pd.DataFrame),
                        'envs': generate_empty_list(N_SIMULATIONS_CONSIDERED, list),
                        'causal_graphs': generate_empty_list(N_SIMULATIONS_CONSIDERED, list),
                        'causal_tables': generate_empty_list(N_SIMULATIONS_CONSIDERED, pd.DataFrame)}

        for sim_n in range(N_SIMULATIONS_CONSIDERED):
            df_track_cd = select_df(dfs_track[sim_n], n_episodes_cd)
            cd = CausalDiscovery(df_track_cd, n_agents, n_enemies, n_goals)
            out_causal_graph = cd.return_causal_graph()
            out_causal_table = cd.return_causal_table()
            print(out_causal_graph.columns)
            print(list(set(out_causal_graph)))
            dict_to_save['dfs_track'][sim_n] = df_track_cd.to_dict(orient='records')
            dict_to_save['envs'][sim_n] = envs[sim_n].tolist()
            dict_to_save['causal_graphs'][sim_n] = out_causal_graph
            dict_to_save['causal_tables'][sim_n] = out_causal_table.to_dict(orient='records')

        with open(f'{DIR_SAVING}/results_grid{rows}x{cols}_{n_enemies}enemies_{int(n_episodes_cd+1)}episodes.json',
                  'w') as json_file:
            json.dump(dict_to_save, json_file)
