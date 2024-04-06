import random
import numpy as np
import global_variables
import os
import json
from scripts.algorithms.causal_discovery import CausalDiscovery
from scripts.utils.environment import CustomEnv
from scripts.utils.train_models import Training

"""This assessment mirrors the one conducted in test1 but within a "toroidal" setting. We maintained identical grid 
sizes, training episodes, and number of simulations as before. Rather than creating a new environment, we adjusted 
the track dataframe to ensure the toroidal nature of the environment. For instance, if the agent selects an action 
that directs it towards a wall on the right, the original result would be "action = 1, deltaX = 0, deltaY = 0"; with 
this modification, the result will be "action = 1, deltaX = -n_cols, deltaY = 0", facilitating toroidal behavior. For
the correct operation, in the case of row modification, the winner and the defeats are ignored.
"""

NAME_DIR_RESULTS = f'{global_variables.GLOBAL_PATH_REPO}/Results/Test3_2'

N_SIMULATIONS = global_variables.N_SIMULATIONS_PAPER
N_TRAINING_EPISODE = global_variables.N_TRAINING_EPISODES
N_AGENTS = 1
N_ENEMIES = 1
N_GOALS = 1
GRID_SIZES = [(3, 3), (4, 4), (6, 6), (8, 8), (10, 10)]

label_kind_of_alg = global_variables.LABEL_RANDOM_AGENT
label_exploration_strategy = global_variables.LABEL_RANDOM_AGENT

# SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
os.makedirs(NAME_DIR_RESULTS, exist_ok=True)
DIR_SAVE_RESULTS = f'{NAME_DIR_RESULTS}'


def make_toroidal(df_input, n_agents, n_enemies, n_goals, n_cols, n_rows):
    new_df = df_input.copy()
    cols_new_df = new_df.columns.to_list()
    cols_actions, cols_DeltaX, cols_DeltaY, cols_GoalNearby, cols_EnemyNearby, cols_rewards = [], [], [], [], [], []
    for ag in range(n_agents):
        cols_actions.append([s for s in cols_new_df if
                             global_variables.LABEL_COL_ACTION in s and f'{global_variables.LABEL_AGENT_CAUSAL_TABLE}{ag}' in s][
                                0])
        cols_DeltaX.append([s for s in cols_new_df if
                            global_variables.LABEL_COL_DELTAX in s and f'{global_variables.LABEL_AGENT_CAUSAL_TABLE}{ag}' in s][
                               0])
        cols_DeltaY.append([s for s in cols_new_df if
                            global_variables.LABEL_COL_DELTAY in s and f'{global_variables.LABEL_AGENT_CAUSAL_TABLE}{ag}' in s][
                               0])

        cols_rewards.append([s for s in cols_new_df if
                             global_variables.LABEL_COL_REWARD in s and f'{global_variables.LABEL_AGENT_CAUSAL_TABLE}{ag}' in s][
                                0])

        for goal in range(n_goals):
            cols_GoalNearby.append([s for s in cols_new_df if f'Goal{goal}' in s and
                                    global_variables.LABEL_NEARBY_CAUSAL_TABLE in s and f'{global_variables.LABEL_AGENT_CAUSAL_TABLE}{ag}' in s][
                                       0])
        for enemy in range(n_enemies):
            cols_EnemyNearby.append([s for s in cols_new_df if f'Enemy{enemy}' in s and
                                     global_variables.LABEL_NEARBY_CAUSAL_TABLE in s and f'{global_variables.LABEL_AGENT_CAUSAL_TABLE}{ag}' in s][
                                        0])

    for index, row in new_df.iterrows():
        for ag in range(n_agents):
            if row[cols_actions[ag]] != 0 and row[cols_DeltaX[ag]] == 0 and row[cols_DeltaY[ag]] == 0:

                new_df.at[index, cols_rewards[ag]] = 0
                for en in range(n_enemies):
                    new_df.at[index, cols_EnemyNearby[en]] = global_variables.VALUE_ENTITY_FAR

                for goal in range(n_goals):
                    new_df.at[index, cols_GoalNearby[goal]] = global_variables.VALUE_ENTITY_FAR

                new_df.at[index, cols_DeltaY[ag]] = - (n_rows - 1)
                new_df.at[index, cols_DeltaY[ag]] = - (n_rows - 1)

                if row[cols_actions[ag]] == 1:  # up
                    new_df.at[index, cols_DeltaY[ag]] = - (n_rows - 1)
                elif row[cols_actions[ag]] == 2:  # down
                    new_df.at[index, cols_DeltaY[ag]] = (n_rows - 1)
                elif row[cols_actions[ag]] == 3:  # right
                    new_df.at[index, cols_DeltaX[ag]] = - (n_cols - 1)
                elif row[cols_actions[ag]] == 4:  # left
                    new_df.at[index, cols_DeltaX[ag]] = (n_cols - 1)
    return new_df


for rows, cols in GRID_SIZES:
    dir_save = f'{DIR_SAVE_RESULTS}/TorGrid{rows}x{cols}'
    os.makedirs(dir_save, exist_ok=True)

    for simulation_n in range(N_SIMULATIONS):
        name_save = f'{rows}x{cols}_game{simulation_n}'

        seed_value = global_variables.seed_values[simulation_n]
        np.random.seed(seed_value)
        random.seed(seed_value)

        dict_learning_params = global_variables.DICT_LEARNING_PARAMETERS_PAPER
        dict_other_params = global_variables.DICT_OTHER_PARAMETERS_PAPER

        dict_env_params = {'rows': rows, 'cols': cols, 'n_agents': N_AGENTS, 'n_enemies': N_ENEMIES, 'n_goals': N_GOALS,
                           'n_actions': global_variables.N_ACTIONS_PAPER, 'if_maze': False,
                           'value_reward_alive': global_variables.VALUE_REWARD_ALIVE_PAPER,
                           'value_reward_winner': global_variables.VALUE_REWARD_WINNER_PAPER,
                           'value_reward_loser': global_variables.VALUE_REWARD_LOSER_PAPER,
                           'seed_value': seed_value, 'enemies_actions': 'random', 'env_type': 'numpy',
                           'predefined_env': None}

        dict_other_params['N_EPISODES'] = N_TRAINING_EPISODE

        env = CustomEnv(dict_env_params)

        env_to_save = np.vectorize(lambda x: env.number_names_grid.get(x, str(x)))(env.grid_for_game)
        with open(f'{dir_save}/env_{name_save}.json', 'w') as json_file:
            json.dump(env_to_save.tolist(), json_file)

        class_train = Training(dict_env_params, dict_learning_params, dict_other_params,
                               f'{label_kind_of_alg}',
                               f'{label_exploration_strategy}')
        class_train.start_train(env, batch_update_df_track=1000)
        df_track = class_train.get_df_track()

        new_df_track = make_toroidal(df_track, n_agents=N_AGENTS, n_enemies=N_ENEMIES, n_goals=N_GOALS, n_rows=rows,
                                     n_cols=cols)

        cd = CausalDiscovery(new_df_track, N_AGENTS, N_ENEMIES, N_GOALS, dir_save, f'graph_{name_save}',
                             f'causal_table_{name_save}')


