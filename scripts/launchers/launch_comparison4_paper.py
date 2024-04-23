import pandas as pd
import global_variables
from scripts.utils.environment import CustomEnv
from scripts.utils.others import get_batch_episodes
from scripts.utils.train_models import Training
import json
import os
import re

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

GRID_SIZES = global_variables.GRID_SIZES_CONSIDERED_PAPER
ENEMIES = global_variables.N_ENEMIES_CONSIDERED_PAPER
N_SIMULATIONS = global_variables.N_SIMULATIONS_PAPER

if_maze = True
OFFLINE_CAUSAL_TABLE = pd.read_pickle(f'{global_variables.PATH_CAUSAL_TABLE_OFFLINE}')
TABLE_BATCH_EPISODES = pd.read_pickle(f'{global_variables.PATH_RESULTS_BATCH_EPISODES_ONLINE_CD}')


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

                environment = CustomEnv(dict_env_params)

                label_kind_of_alg = global_variables.LABEL_Q_LEARNING

                for label_kind_of_alg2 in [global_variables.LABEL_VANILLA, global_variables.LABEL_CAUSAL_OFFLINE,
                                           global_variables.LABEL_CAUSAL_ONLINE]:

                    for if_transfer_learning in [True, False]:
                        label_exploration_strategy = global_variables.LABEL_EPSILON_GREEDY

                        dict_learning_params = global_variables.DICT_LEARNING_PARAMETERS_PAPER

                        game_info1 = f'Grid{rows}x{cols}_{n_enemies}'
                        game_info1 += 'enemies' if n_enemies > 1 else 'enemy'

                        name_alg = f'{label_kind_of_alg}_{label_kind_of_alg2}_{label_exploration_strategy}_game{simulation_n}'

                        dict_learning_params['KNOWLEDGE_TRANSFERRED'] = get_q_table(game_info1,
                                                                                    dir_start_results,
                                                                                    name_alg) if if_transfer_learning else None

                        print(f'** Simulation: {simulation_n + 1}/{N_SIMULATIONS}')
                        class_train = Training(dict_env_params, dict_learning_params, dict_other_params,
                                               f'{label_kind_of_alg}_{label_kind_of_alg2}',
                                               f'{label_exploration_strategy}',
                                               OFFLINE_CAUSAL_TABLE)

                        add_name = 'Grid' if if_maze else 'Maze'
                        add_name += f'{rows}x{cols}_{n_enemies}' + 'enemies' if n_enemies > 1 else 'enemy'

                        dir_save_final = f'{dir_save}/{add_name}'

                        name_save = f'{label_kind_of_alg}_{label_kind_of_alg2}_{label_exploration_strategy}_'
                        name_save += 'TransferLearning' if if_transfer_learning else 'NoTL'
                        name_save += f'_game{simulation_n}'

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


"""def rename_files(directory):
    for root, dirs, files in os.walk(directory):
        for filename in files:
            # Separate the base name and the extension
            base_name, extension = os.path.splitext(filename)

            # Detect parts like "game4"
            match = re.search(r'(game\d+)', base_name)
            if match:
                game_part = match.group(0)  # The 'game' followed by digits (e.g., 'game4')
                base_name = base_name.replace(game_part, '')  # Remove 'game4' from base
            else:
                game_part = ''  # No 'game' followed by digits part found

            # Remove 'TF' and append '_TransferLearning' or append '_NoTL'
            if 'TF' in base_name:
                new_base_name = base_name.replace('TF_', '') + 'TransferLearning'
            else:
                new_base_name = base_name + 'NoTL'

            # Construct the new filename, placing the 'game' part at the end
            new_filename = new_base_name
            if game_part:
                new_filename += '_' + game_part  # Append 'game4' at the end if it was found
            new_filename += extension  # Append the file extension

            # Full path with the new file name
            new_file_path = os.path.join(root, new_filename)

            # Rename the file
            os.rename(os.path.join(root, filename), new_file_path)
            print(f'Renamed {os.path.join(root, filename)} to {new_file_path}')


# Replace 'your_directory_path' with the path to the directory you want to process
rename_files(f'{global_variables.GLOBAL_PATH_REPO}/Results/Comparison4')"""

