import random

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


def prepare_df_for_comparison(df1: pd.DataFrame) -> pd.DataFrame:
    col_df1 = df1.columns.to_list()
    # Sort DataFrames by values
    sorted_df1 = df1.sort_index(axis=0).sort_index(axis=1).reset_index(drop=True)

    new_col_df1_to_drop = [s for s in sorted_df1.columns.to_list() if s not in col_df1]

    check_df1 = sorted_df1.drop(columns=new_col_df1_to_drop)

    return check_df1


GROUND_TRUTH_CAUSAL_TABLE = prepare_df_for_comparison(pd.read_pickle(f'{global_variables.PATH_CAUSAL_TABLE_OFFLINE}'))

N_SIMULATIONS_CONSIDERED = 6      # global_variables.N_SIMULATIONS_PAPER
N_ENEMIES_CONSIDERED = [1, 2]     # global_variables.N_ENEMIES_CONSIDERED_PAPER
N_EPISODES_CONSIDERED = [100]     # global_variables.N_EPISODES_CONSIDERED_FOR_ANALYSIS_PAPER
GRID_SIZES_CONSIDERED = [(5, 5)]  # global_variables.GRID_SIZES_CONSIDERED_PAPER
n_agents = 1
n_goals = 1

dict_learning_params = global_variables.DICT_LEARNING_PARAMETERS_PAPER
dict_other_params = global_variables.DICT_OTHER_PARAMETERS_PAPER

label_kind_of_alg = global_variables.LABEL_RANDOM_AGENT
label_exploration_strategy = 'random'

n_tries = N_SIMULATIONS_CONSIDERED * len(N_ENEMIES_CONSIDERED) * len(N_EPISODES_CONSIDERED) * len(GRID_SIZES_CONSIDERED)

dict_storing_data = {'grid_size': generate_empty_list(n_tries, tuple),
                     'n_enemies': generate_empty_list(n_tries, int),
                     'n_episodes': generate_empty_list(n_tries, int),
                     'df': generate_empty_list(n_tries, pd.DataFrame),
                     'causal_table_generated': generate_empty_list(n_tries, pd.DataFrame)}

# TODO: FINISH HERE ONCE WE HAVE THE OFFLINE CAUSAL TABLE


try_n = 0
for n_enemies in N_ENEMIES_CONSIDERED:
    seed_value = global_variables.seed_values[try_n]
    np.random.seed(seed_value)
    random.seed(seed_value)

    for rows, cols in GRID_SIZES_CONSIDERED:
        for n_episodes in N_EPISODES_CONSIDERED:
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

            # Create an environment
            env = CustomEnv(dict_env_params)

            class_train = Training(dict_env_params, dict_learning_params, dict_other_params,
                                   f'{label_kind_of_alg}',
                                   f'{label_exploration_strategy}')

            for simulation_n in range(N_SIMULATIONS_CONSIDERED):
                class_train.start_train(env, df_track=True)

                df_track = class_train.get_df_track()

                dict_storing_data['grid_size'][try_n] = (rows, cols)
                dict_storing_data['n_enemies'][try_n] = n_enemies
                dict_storing_data['n_episodes'][try_n] = n_episodes
                dict_storing_data['df'][try_n] = df_track
                out_causal_table = CausalDiscovery(df_track, n_agents, n_enemies, n_goals).return_causal_table()
                dict_storing_data['causal_table_generated'][try_n] = out_causal_table

                try_n += 1

len_df = int(n_tries / N_SIMULATIONS_CONSIDERED)
dict_rows = {'n_enemies': generate_empty_list(len_df, int),
             'grid_size': generate_empty_list(len_df, tuple),
             'n_episodes': generate_empty_list(len_df, int),
             f'checks_over_{N_SIMULATIONS_CONSIDERED}': generate_empty_list(len_df, int),
             'suitable': generate_empty_list(len_df, bool)}

try_n = 0
count_n = 0
for n_enemies in N_ENEMIES_CONSIDERED:
    for rows, cols in GRID_SIZES_CONSIDERED:
        checks = 0
        for n_episodes in N_EPISODES_CONSIDERED:
            for simulation_n in range(N_SIMULATIONS_CONSIDERED):
                df = dict_storing_data['causal_table_generated'][count_n]
                df = prepare_df_for_comparison(df)
                count_n += 1

                if df.equals(GROUND_TRUTH_CAUSAL_TABLE):
                    checks += 1

            dict_rows['n_enemies'] = n_enemies
            dict_rows['grid_size'] = (rows, cols)
            dict_rows['n_episodes'] = n_episodes
            dict_rows['checks_over_{N_SIMULATIONS_CONSIDERED}'] = checks
            dict_rows['suitable'][try_n] = checks > int(N_SIMULATIONS_CONSIDERED / 2)  # better random case

            try_n += 1

out_table_results = pd.DataFrame(dict_rows)
out_table_results.to_excel(f'{global_variables.GLOBAL_PATH_REPO}/res_batch_episodes.pkl')
