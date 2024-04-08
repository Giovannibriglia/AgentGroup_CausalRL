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


def prepare_df_for_comparison(df1: pd.DataFrame) -> pd.DataFrame:
    col_df1 = df1.columns.to_list()
    # Sort DataFrames by values
    sorted_df1 = df1.sort_index(axis=0).sort_index(axis=1).reset_index(drop=True)

    new_col_df1_to_drop = [s for s in sorted_df1.columns.to_list() if s not in col_df1]

    check_df1 = sorted_df1.drop(columns=new_col_df1_to_drop)

    return check_df1


GROUND_TRUTH_CAUSAL_TABLE = prepare_df_for_comparison(pd.read_pickle(f'{global_variables.PATH_CAUSAL_TABLE_OFFLINE}'))

with open(f'{global_variables.PATH_CAUSAL_GRAPH_OFFLINE}', 'r') as file:
    GROUND_TRUTH_CAUSAL_GRAPH = json.load(file)

N_SIMULATIONS_CONSIDERED = global_variables.N_SIMULATIONS_PAPER
N_ENEMIES_CONSIDERED = [1]  # global_variables.N_ENEMIES_CONSIDERED_PAPER
N_EPISODES_CONSIDERED = [500]  # global_variables.N_EPISODES_CONSIDERED_FOR_SENSITIVE_ANALYSIS_PAPER
GRID_SIZES_CONSIDERED = [(5, 5)]  # global_variables.GRID_SIZES_CONSIDERED_PAPER
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

with open(f'{global_variables.PATH_LIST_OF_DICTS_BATCH_EPISODES_ANALYSIS}', 'w') as json_file:
    json.dump(list_combinations_for_simulations, json_file, indent=4)

" Second part: comparisons "
list_combinations_df_results = [{'n_enemies': item[0], 'n_episodes': item[1], 'grid_size': item[2]} for item in
                                combinations_enemies_episodes_grid]

for new_dict_comb in list_combinations_df_results:
    n_enemies = new_dict_comb['n_enemies']
    grid_size = new_dict_comb['grid_size']
    n_episodes = new_dict_comb['n_episodes']

    filtered_dict = [value for value in list_combinations_for_simulations if
                     value['n_enemies'] == n_enemies and value['grid_size'] == grid_size and value[
                         'n_episodes'] == n_episodes][0]

    vet_causal_graphs = filtered_dict['causal_graph']
    n_checks = 0
    for sim_n in range(N_SIMULATIONS_CONSIDERED):
        causal_graph = vet_causal_graphs[sim_n]

        if compare_causal_graphs(causal_graph, GROUND_TRUTH_CAUSAL_GRAPH):
            n_checks += 1

    new_dict_comb[f'checks_over_{N_SIMULATIONS_CONSIDERED}simulations'] = n_checks
    new_dict_comb['suitable'] = n_checks > int(N_SIMULATIONS_CONSIDERED / 2)  # better than the random case

out_table_results = pd.DataFrame(list_combinations_df_results)
out_table_results.to_pickle(f'{global_variables.PATH_RESULTS_BATCH_EPISODES_ONLINE_CD}')
out_table_results.to_excel(f'{global_variables.GLOBAL_PATH_REPO}/batch_episodes_online.xlsx')
