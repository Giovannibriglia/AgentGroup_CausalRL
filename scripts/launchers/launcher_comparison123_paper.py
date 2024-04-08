from scripts.utils.environment import CustomEnv
from scripts.utils.train_models import Training
from scripts.utils.others import get_batch_episodes
import global_variables

"""The objective of this script is to conduct comparative analyses across various environments, algorithms (
Q-Learning and DQN), exploration strategies (Epsilon-Greedy, Thompson Sampling, Boltzmann Machine, and Softmax 
Annealing), and algorithm types (vanilla, with offline-extracted causal knowledge, with online-extracted causal 
knowledge).

The simulation results include metrics such as rewards for each episode, computation time for each episode, 
final q-table (if available), the number of steps taken to complete each episode and number of timeout occurred, 
along with accompanying videos."""

dir_save = 'Comparison123'

if_maze = False

for simulation_n in range(global_variables.N_SIMULATIONS_PAPER):
    for rows, cols in global_variables.GRID_SIZES_CONSIDERED_PAPER:
        for n_enemies in global_variables.N_ENEMIES_CONSIDERED_PAPER:
            
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
                dict_learning_params = global_variables.DICT_LEARNING_PARAMETERS_PAPER
                dict_other_params = global_variables.DICT_OTHER_PARAMETERS_PAPER

                # Create an environment
                environment = CustomEnv(dict_env_params)

                for label_kind_of_alg in [global_variables.LABEL_Q_LEARNING, global_variables.LABEL_DQN]:

                    for label_kind_of_alg2 in [global_variables.LABEL_VANILLA, global_variables.LABEL_CAUSAL_OFFLINE]:

                        for label_exploration_strategy in [global_variables.LABEL_SOFTMAX_ANNEALING,
                                                           global_variables.LABEL_THOMPSON_SAMPLING,
                                                           global_variables.LABEL_BOLTZMANN_MACHINE,
                                                           global_variables.LABEL_EPSILON_GREEDY]:
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
                                                                                             cols) if cond_online else None,
                                                    episodes_to_visualize=global_variables.EPISODES_TO_VISUALIZE_PAPER,
                                                    dir_save_videos=dir_save_final,
                                                    name_save_videos=name_save)
