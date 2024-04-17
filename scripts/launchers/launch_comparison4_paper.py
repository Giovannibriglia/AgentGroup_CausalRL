import pandas as pd
import global_variables
from scripts.utils.environment import CustomEnv
from scripts.utils.others import get_batch_episodes
from scripts.utils.train_models import Training
import json

"""The objective of this script is to perform comparative analyses aimed at understanding the effectiveness of 
transfer learning within a maze-like environment. This investigation utilizes the Q-Learning algorithm with an 
Epsilon-Greedy exploration strategy in both vanilla and offline-extracted causal knowledge settings.

Two distinct scenarios were examined: the first involves an agent beginning with an uninitialized Q-Table, 
thus lacking knowledge of the goal's location; the second scenario features an agent equipped with a pre-trained 
Q-table derived from identical grid-environment configurations (with matching numbers of enemies, goal positions, 
and starting positions for enemies and agents).

The simulation results include metrics such as rewards for each episode, computation time for each episode, 
final q-table, the number of steps taken to complete each episode and number of timeout occurred, 
along with accompanying videos."""

dir_start_results = 'Comparison123'
dir_save = 'Comparison4'

GRID_SIZES = [(5, 5)]#global_variables.GRID_SIZES_CONSIDERED_PAPER
ENEMIES = [2] #lobal_variables.N_ENEMIES_CONSIDERED_PAPER
N_SIMULATIONS = global_variables.N_SIMULATIONS_PAPER

if_maze = True
OFFLINE_CAUSAL_TABLE = pd.read_pickle(f'{global_variables.PATH_CAUSAL_TABLE_OFFLINE}')
TABLE_BATCH_EPISODES = pd.read_pickle(f'{global_variables.PATH_RESULTS_BATCH_EPISODES_ONLINE_CD}')


# TODO: FIX LABELS TF AND NO TF and check the type of the q-table
def get_q_table(game_infos: str, dir_results: str, algo: str):
    try:
        with open(f'{global_variables.GLOBAL_PATH_REPO}/Results/{dir_results}/{game_infos}/{algo}.json', 'r') as file:
            series = json.load(file)

            return series['q_table']
    except:
        raise AssertionError('q table not available, check your usage of this function.')

for simulation_n in range(N_SIMULATIONS):
    for rows, cols in GRID_SIZES:
        for n_enemies in ENEMIES:

            if not (rows == 5 and n_enemies == 10):
                seed_value = global_variables.seed_values[simulation_n]

                dict_env_params = {'rows': rows, 'cols': cols, 'n_agents': 1, 'n_enemies': n_enemies, 'n_goals': 1,
                                   'n_actions': global_variables.N_ACTIONS_PAPER,
                                   'if_maze': if_maze,
                                   'value_reward_alive': global_variables.VALUE_REWARD_ALIVE_PAPER,
                                   'value_reward_winner': global_variables.VALUE_REWARD_WINNER_PAPER,
                                   'value_reward_loser': global_variables.VALUE_REWARD_LOSER_PAPER,
                                   'seed_value': seed_value, 'enemies_actions': 'random', 'env_type': 'numpy',
                                   'predefined_env': None}
                dict_other_params = global_variables.DICT_OTHER_PARAMETERS_PAPER

                # Create an environment
                environment = CustomEnv(dict_env_params)

                label_kind_of_alg = global_variables.LABEL_Q_LEARNING

                for label_kind_of_alg2 in [global_variables.LABEL_VANILLA, global_variables.LABEL_CAUSAL_OFFLINE]:

                    if global_variables.LABEL_CAUSAL in label_kind_of_alg2:
                        for if_transfer_learning in [False, True]:
                            label_exploration_strategy = global_variables.LABEL_EPSILON_GREEDY

                            dict_learning_params = global_variables.DICT_LEARNING_PARAMETERS_PAPER

                            game_info1 = f'Grid{rows}x{cols}_{n_enemies}'
                            game_info1 += 'enemies' if n_enemies > 1 else 'enemy'

                            name_alg = f'{label_kind_of_alg}_{label_kind_of_alg2}_{label_exploration_strategy}_game{simulation_n}'

                            dict_learning_params['KNOWLEDGE_TRANSFERRED'] = get_q_table(game_info1,
                                                                                        dir_start_results,
                                                                                        name_alg) if if_transfer_learning else None

                            class_train = Training(dict_env_params, dict_learning_params, dict_other_params,
                                                   f'{label_kind_of_alg}_{label_kind_of_alg2}',
                                                   f'{label_exploration_strategy}',
                                                   OFFLINE_CAUSAL_TABLE)

                            add_name = 'Grid' if if_maze else 'Maze'
                            add_name += f'{rows}x{cols}_{n_enemies}' + 'enemies' if n_enemies > 1 else 'enemy'

                            dir_save_final = f'{dir_save}/{add_name}'
                            name_save = 'TF_' if if_transfer_learning else ''
                            name_save += name_alg

                            cond_online = label_kind_of_alg2 == global_variables.LABEL_CAUSAL_ONLINE

                            class_train.start_train(environment,
                                                    dir_save_metrics=dir_save_final,
                                                    name_save_metrics=name_save,
                                                    batch_update_df_track=get_batch_episodes(n_enemies, rows,
                                                                                             cols,
                                                                                             TABLE_BATCH_EPISODES) if cond_online else None,
                                                    episodes_to_visualize=global_variables.EPISODES_TO_VISUALIZE_PAPER,
                                                    dir_save_videos=dir_save_final,
                                                    name_save_videos=name_save)

                    else:
                        label_exploration_strategy = global_variables.LABEL_EPSILON_GREEDY

                        dict_learning_params = global_variables.DICT_LEARNING_PARAMETERS_PAPER
                        dict_learning_params['KNOWLEDGE_TRANSFERRED'] = None

                        class_train = Training(dict_env_params, dict_learning_params, dict_other_params,
                                               f'{label_kind_of_alg}_{label_kind_of_alg2}',
                                               f'{label_exploration_strategy}')

                        add_name_dir_save = 'Maze' if if_maze else 'Grid' + f'{rows}x{cols}_{n_enemies}' + 'enemies' if n_enemies > 1 else 'enemy'

                        dir_save_final = f'{dir_save}/{add_name_dir_save}'
                        name_save = f'{label_kind_of_alg}_{label_kind_of_alg2}_{label_exploration_strategy}_game{simulation_n}'

                        cond_online = label_kind_of_alg2 == global_variables.LABEL_CAUSAL_ONLINE

                        class_train.start_train(environment,
                                                dir_save_metrics=dir_save_final,
                                                name_save_metrics=name_save,
                                                batch_update_df_track=get_batch_episodes(n_enemies, rows,
                                                                                         cols,
                                                                                         TABLE_BATCH_EPISODES) if cond_online else None,
                                                episodes_to_visualize=global_variables.EPISODES_TO_VISUALIZE_PAPER,
                                                dir_save_videos=dir_save_final,
                                                name_save_videos=name_save)


"""if __name__ == '__main__':
    dict_env_params = {'rows': 4, 'cols': 4, 'n_agents': 1, 'n_enemies': 1, 'n_goals': 1,
                       'n_actions': global_variables.N_ACTIONS_PAPER,
                       'if_maze': if_maze,
                       'value_reward_alive': global_variables.VALUE_REWARD_ALIVE_PAPER,
                       'value_reward_winner': global_variables.VALUE_REWARD_WINNER_PAPER,
                       'value_reward_loser': global_variables.VALUE_REWARD_LOSER_PAPER,
                       'seed_value': 1, 'enemies_actions': 'random', 'env_type': 'numpy',
                       'predefined_env': None}
    dict_other_params = global_variables.DICT_OTHER_PARAMETERS_PAPER

    dict_learning_params = global_variables.DICT_LEARNING_PARAMETERS_PAPER

    class_train = Training(dict_env_params, dict_learning_params, dict_other_params,
                           f'QL_causal_offline',
                           f'EG',
                           OFFLINE_CAUSAL_TABLE)"""
