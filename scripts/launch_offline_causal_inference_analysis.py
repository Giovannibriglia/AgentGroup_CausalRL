import global_variables
from scripts.environment import CustomEnv
from scripts.train_models import Training

dict_env_params = {'rows': 5, 'cols': 5, 'n_agents': 1, 'n_enemies': 2, 'n_goals': 1,
                   'n_actions': global_variables.N_ACTIONS_PAPER,
                   'if_maze': False,
                   'value_reward_alive': global_variables.VALUE_REWARD_ALIVE_PAPER,
                   'value_reward_winner': global_variables.VALUE_REWARD_WINNER_PAPER,
                   'value_reward_loser': global_variables.VALUE_REWARD_LOSER_PAPER,
                   'seed_value': 1, 'enemies_actions': 'random', 'env_type': 'numpy',
                   'predefined_env': None}
dict_learning_params = global_variables.DICT_LEARNING_PARAMETERS_PAPER
dict_other_params = global_variables.DICT_OTHER_PARAMETERS_PAPER

# Create an environment
env = CustomEnv(dict_env_params, False)

for label_kind_of_alg in [global_variables.LABEL_RANDOM_AGENT, global_variables.LABEL_Q_LEARNING]:

    if label_kind_of_alg == global_variables.LABEL_RANDOM_AGENT:
        label_exploration_strategy = 'random'
        class_train = Training(dict_env_params, dict_learning_params, dict_other_params,
                               f'{label_kind_of_alg}_{global_variables.LABEL_VANILLA}',
                               f'{label_exploration_strategy}')
        # Train the agent
        class_train.start_train(env, dir_save_metrics=None, name_sav_metrics=None,
                                df_track=False, batch_update_df_track=1000,
                                episodes_to_visualize=[], dir_save_video='Comparison123',
                                name_save_video=f'{label_kind_of_alg}_{global_variables.LABEL_VANILLA}_{label_exploration_strategy}')

        class_train.get_df_track().to_excel(f'{global_variables.GLOBAL_PATH_REPO}/df_track.xlsx')
