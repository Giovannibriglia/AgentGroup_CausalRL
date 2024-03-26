import pandas as pd
import global_variables
from scripts.algorithms.causal_inference import CausalInference
from scripts.environment import CustomEnv
from scripts.train_models import Training

""" The objective of this simulation is to address the inherent tradeoff within the algorithm involving online causal
 inference. Specifically, our aim is to determine the optimal number of episodes needed to reliably generate the 
 correct causal table for varying numbers of enemies and grid sizes."""


def generate_empty_list(X: int, data_type) -> list:
    return [data_type() for _ in range(X)]


N_SIMULATIONS = 3
N_ENEMIES_CONSIDERED = [2]  # [2, 5, 10]
N_EPISODES_CONSIDERED = [100]  # [100, 250, 500, 100]
GRID_SIZE = [(5, 5)]  # [(5, 5), (10, 10)]
n_agents = 1
n_goals = 1

dict_learning_params = global_variables.DICT_LEARNING_PARAMETERS_PAPER
dict_other_params = global_variables.DICT_OTHER_PARAMETERS_PAPER

label_kind_of_alg = global_variables.LABEL_RANDOM_AGENT
label_exploration_strategy = 'random'

dict_storing_data = {'n_simulations': N_SIMULATIONS, 'grid_size': generate_empty_list(N_SIMULATIONS, tuple),
                     'n_enemies': generate_empty_list(N_SIMULATIONS, int),
                     'n_episodes': generate_empty_list(N_SIMULATIONS, int),
                     'df': generate_empty_list(N_SIMULATIONS, pd.DataFrame),
                     'causal_table_generated': generate_empty_list(N_SIMULATIONS, pd.DataFrame)}

for simulation_n in range(N_SIMULATIONS):
    seed_value = global_variables.seed_values[simulation_n]

    for n_enemies in N_ENEMIES_CONSIDERED:

        for rows, cols in GRID_SIZE:
            dict_env_params = {'rows': rows, 'cols': cols, 'n_agents': n_agents, 'n_enemies': n_enemies,
                               'n_goals': n_goals,
                               'n_actions': global_variables.N_ACTIONS_PAPER,
                               'if_maze': False,
                               'value_reward_alive': global_variables.VALUE_REWARD_ALIVE_PAPER,
                               'value_reward_winner': global_variables.VALUE_REWARD_WINNER_PAPER,
                               'value_reward_loser': global_variables.VALUE_REWARD_LOSER_PAPER,
                               'seed_value': seed_value, 'enemies_actions': 'random', 'env_type': 'numpy',
                               'predefined_env': None}

            for n_episodes in N_EPISODES_CONSIDERED:
                cols_df = global_variables.define_columns_causal_table(n_agents, n_enemies, n_goals)
                df = pd.DataFrame(columns=cols_df)

                dict_other_params['N_EPISODES'] = n_episodes

                # Create an environment
                env = CustomEnv(dict_env_params, False)

                class_train = Training(dict_env_params, dict_learning_params, dict_other_params,
                                       f'{label_kind_of_alg}',
                                       f'{label_exploration_strategy}')
                # Train the agent
                class_train.start_train(env, df_track=True)

                df_track = class_train.get_df_track()

                dict_storing_data['grid_size'][simulation_n] = (rows, cols)
                dict_storing_data['n_enemies'][simulation_n] = n_enemies
                dict_storing_data['n_episodes'][simulation_n] = n_episodes
                dict_storing_data['df'][simulation_n] = df_track

                out_causal_table = CausalInference(df_track, n_agents, n_enemies, n_goals).return_causal_table()
                dict_storing_data['causal_table_generated'] = out_causal_table


def prepare_df_for_comparison(df1, df2):
    col_df1 = df1.columns.to_list()
    col_df2 = df2.columns.to_list()
    # Sort DataFrames by values
    sorted_df1 = df1.sort_index(axis=0).sort_index(axis=1).reset_index(drop=True)
    sorted_df2 = df2.sort_index(axis=0).sort_index(axis=1).reset_index(drop=True)

    new_col_df1_to_drop = [s for s in sorted_df1.columns.to_list() if s not in col_df1]
    new_col_df2_to_drop = [s for s in sorted_df2.columns.to_list() if s not in col_df2]

    check_df1 = sorted_df1.drop(columns=new_col_df1_to_drop)
    check_df2 = sorted_df2.drop(columns=new_col_df2_to_drop)

    return check_df1, check_df2
