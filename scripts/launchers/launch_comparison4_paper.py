import global_variables
from scripts.utils.environment import CustomEnv
from scripts.utils.train_models import Training

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

dir_save = 'Comparison4'

if_maze = True

# TODO: FIX LABELS TF AND NO TF
def get_q_table():
    # TODO: implementation
    pass


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
                dict_other_params = global_variables.DICT_OTHER_PARAMETERS_PAPER

                # Create an environment
                environment = CustomEnv(dict_env_params)

                label_kind_of_alg = global_variables.LABEL_Q_LEARNING

                for label_kind_of_alg2 in [global_variables.LABEL_VANILLA, global_variables.LABEL_CAUSAL_OFFLINE]:

                    for if_transfer_learning in [True, False]:
                        label_exploration_strategy = global_variables.LABEL_EPSILON_GREEDY

                        dict_learning_params = global_variables.DICT_LEARNING_PARAMETERS_PAPER
                        dict_learning_params['KNOWLEDGE_TRANSFERRED'] = get_q_table() if if_transfer_learning else None

                        class_train = Training(dict_env_params, dict_learning_params, dict_other_params,
                                               f'{label_kind_of_alg}_{label_kind_of_alg2}',
                                               f'{label_exploration_strategy}')

                        add_name_dir_save = 'Maze' if if_maze else 'Grid'
                        add_name_dir_save += f'{rows}x{cols}_{n_enemies}' + 'enemies' if n_enemies > 1 else 'enemy'

                        dir_save_final = f'{dir_save}/{add_name_dir_save}'
                        name_save = 'TransferLearning_' if if_transfer_learning else ''
                        name_save += f'{label_kind_of_alg}_{label_kind_of_alg2}_{label_exploration_strategy}_game{simulation_n}'

                        """class_train.start_train(environment,
                                                    dir_save_metrics=dir_save_final,
                                                    name_save_metrics=name_save,
                                                    batch_update_df_track=None,
                                                    episodes_to_visualize=global_variables.EPISODES_TO_VISUALIZE_PAPER,
                                                    dir_save_videos=dir_save_final,
                                                    name_save_videos=name_save)"""
