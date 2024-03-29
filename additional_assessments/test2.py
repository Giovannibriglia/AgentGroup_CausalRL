import numpy as np
import global_variables
import os
import json
from scripts.algorithms.causal_discovery import CausalDiscovery
from scripts.utils.environment import CustomEnv
from scripts.utils.train_models import Training

"""
Comparison of Offline and Online Causal Q-Learning in various Test Environments.

We conducted experiments comparing offline and online causal Q-Learning in identical test settings. 
For offline causal Q-Learning, we used three distinct causal tables: 
    1) In the first case, the causal table was derived from an environment where 
    the goal was positioned in the top-right corner of the grid. Due to this placement, certain causal relationships, 
    such as "if the goal is on my right, then I go right" and "if the goal is beneath me, I go down"  were impossible to 
    establish. This resulted in an incomplete causal table. 
    2) The second causal table was extracted from an environment where the goal was placed in the left edge but not in
    any corner. In this case the relation "if the goal is on my right, then I go right" was impossible to establish.
    3) The third causal table was extracted from an environment where the goal was placed in one of the central cells
    of the grid. Here, the 'Goal_Nearby_Agent' feature was observable across all possible values.

Therefore, we have three kinds of algorithms to evaluate:
    1) Offline causal Q-Learning with incomplete causal knowledge
    2) Offline causal Q-Learning with complete causal knowledge
    3) Online causal Q-Learning

We evaluated the performance differences between these algorithms across three 4x4 grid environments, each containing
two enemies:
    1) The goal was situated in the bottom-left corner in the first environment.
    2) In the second environment, the goal was adjacent to the right edge of the grid but not in any corner.
    3) The third environment featured the goal positioned in one of the central cells.

"""
# TODO: get_batch_online_causal_table
# TODO: @Stefano help me, change key names in envs_causality and envs_test dicts
dir_save = f'Results_Test2'


def offline_cd(env_params, other_params):
    label_kind_of_alg = global_variables.LABEL_RANDOM_AGENT
    label_exploration_strategy = global_variables.LABEL_RANDOM_AGENT

    env = CustomEnv(dict_env_params)

    class_train = Training(env_params, dict_learning_params, other_params,
                           f'{label_kind_of_alg}',
                           f'{label_exploration_strategy}')

    class_train.start_train(env, batch_update_df_track=1000, episodes_to_visualize=[0])
    df_track = class_train.get_df_track()
    out_causal_table = CausalDiscovery(df_track, N_AGENTS, N_ENEMIES, N_GOALS).return_causal_table()
    return out_causal_table


ROWS_GRID = 4
COLUMNS_GRID = 4
N_AGENTS = 1
N_ENEMIES = 1
N_GOALS = 1
N_EPISODES_CD = 5000
N_TRAINING_EPISODES = global_variables.N_TRAINING_EPISODES

env_causality1 = {'agents_positions': [(0, 3)],
                  'enemies_positions': [(2, 1)],
                  'goals_positions': [(0, 0)],
                  'walls_positions': []}
env_causality2 = {'agents_positions': [(0, 3)],
                  'enemies_positions': [(2, 1)],
                  'goals_positions': [(2, 0)],
                  'walls_positions': []}
env_causality3 = {'agents_positions': [(0, 3)],
                  'enemies_positions': [(2, 1)],
                  'goals_positions': [(1, 1)],
                  'walls_positions': []}
envs_causality = {'incomplete_CD': env_causality1, 'less_incomplete_CD': env_causality2, 'complete_CD': env_causality3}

env_test1 = {'agents_positions': [(0, 3)],
             'enemies_positions': [(2, 1)],
             'goals_positions': [(3, 3)],
             'walls_positions': []}
env_test2 = {'agents_positions': [(0, 3)],
             'enemies_positions': [(2, 1)],
             'goals_positions': [(1, 3)],
             'walls_positions': []}
env_test3 = {'agents_positions': [(0, 3)],
             'enemies_positions': [(2, 1)],
             'goals_positions': [(1, 2)],
             'walls_positions': []}
envs_test = {'incomplete_knowledge_for_OfflineCD': env_test1, 'less_incomplete__knowledge_for_OfflineCD': env_test2,
             'complete_knowledge_for_OfflineCD': env_test3}

dict_learning_params = global_variables.DICT_LEARNING_PARAMETERS_PAPER

for label_env_causality in envs_causality.keys():
    env_causality = envs_causality[label_env_causality]
    seed_value = global_variables.seed_values[0]

    dict_env_params = {'rows': ROWS_GRID, 'cols': COLUMNS_GRID, 'n_agents': N_AGENTS, 'n_enemies': N_ENEMIES,
                       'n_goals': N_GOALS, 'n_actions': global_variables.N_ACTIONS_PAPER, 'if_maze': False,
                       'value_reward_alive': global_variables.VALUE_REWARD_ALIVE_PAPER,
                       'value_reward_winner': global_variables.VALUE_REWARD_WINNER_PAPER,
                       'value_reward_loser': global_variables.VALUE_REWARD_LOSER_PAPER,
                       'seed_value': seed_value, 'enemies_actions': 'random', 'env_type': 'numpy',
                       'predefined_env': env_causality}

    dict_other_params = global_variables.DICT_OTHER_PARAMETERS_PAPER
    dict_other_params['N_EPISODES'] = N_EPISODES_CD

    offline_causal_table = offline_cd(dict_env_params, dict_other_params)

    for label_env_test in envs_test:
        env_test = envs_test[label_env_test]

        dict_env_params = {'rows': ROWS_GRID, 'cols': COLUMNS_GRID, 'n_agents': N_AGENTS, 'n_enemies': N_ENEMIES,
                           'n_goals': N_GOALS, 'n_actions': global_variables.N_ACTIONS_PAPER, 'if_maze': False,
                           'value_reward_alive': global_variables.VALUE_REWARD_ALIVE_PAPER,
                           'value_reward_winner': global_variables.VALUE_REWARD_WINNER_PAPER,
                           'value_reward_loser': global_variables.VALUE_REWARD_LOSER_PAPER,
                           'seed_value': seed_value, 'enemies_actions': 'random', 'env_type': 'numpy',
                           'predefined_env': env_test}

        dict_other_params = global_variables.DICT_OTHER_PARAMETERS_PAPER

        # Create an environment
        environment = CustomEnv(dict_env_params)

        for label_kind_of_alg in [global_variables.LABEL_Q_LEARNING, global_variables.LABEL_DQN]:

            for label_kind_of_alg2 in [global_variables.LABEL_CAUSAL_OFFLINE, global_variables.LABEL_CAUSAL_ONLINE]:
                label_exploration_strategy = global_variables.LABEL_EPSILON_GREEDY

                class_train = Training(dict_env_params, dict_learning_params, dict_other_params,
                                       f'{label_kind_of_alg}_{label_kind_of_alg2}',
                                       f'{label_exploration_strategy}')

                add_name_dir_save = 'Grid' + f'{ROWS_GRID}x{COLUMNS_GRID}_{N_ENEMIES}' + 'enemies' if N_ENEMIES > 1 else 'enemy'

                dir_save_final = f'{dir_save}/{add_name_dir_save}'
                name_save = f'{label_kind_of_alg}_{label_kind_of_alg2}_{label_exploration_strategy}_{label_env_causality}_{label_env_test}'

                cond_online = label_kind_of_alg2 == global_variables.LABEL_CAUSAL_ONLINE

                class_train.start_train(environment,
                                        dir_save_metrics=dir_save_final,
                                        name_save_metrics=name_save,
                                        batch_update_df_track=get_batch_episodes() if cond_online else None,
                                        episodes_to_visualize=global_variables.EPISODES_TO_VISUALIZE_PAPER,
                                        dir_save_videos=dir_save_final,
                                        name_save_videos=name_save)
