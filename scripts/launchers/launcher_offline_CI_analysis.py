import numpy as np
import random
import global_variables
from scripts.algorithms.causal_inference import CausalInference
from scripts.environment import CustomEnv
from scripts.train_models import Training

"""
The aim of this function is to produce the 'ground truth' for the causal table. Given its significance, we opted for
 a substantial number of episodes to ensure its adequacy. Additionally, for the environment under consideration,
 we selected a 3x3 grid with one enemy and one goal. Lastly, the agent navigates using a random policy.

The result of this simulation is the causal table used in algorithms incorporating offline causal inference.
"""
# TODO: Extracting causal dependencies encounters a challenge: when the agent encounters a wall, it executes an action
#  (nonzero) despite resulting in deltaX and deltaY being 0. This discrepancy arises because the action isn't 0.
#  A potential solution involves incorporating the impact of walls as a distinct feature termed "Wall_Nearby_Agent"
#  in the modeling process.
N_AGENTS = 1
N_ENEMIES = 1
N_GOALS = 1
GRID_SIZE = (3, 3)
N_EPISODES = 10000

label_kind_of_alg = global_variables.LABEL_RANDOM_AGENT
label_exploration_strategy = 'random'

seed_value = global_variables.seed_values[0]
np.random.seed(seed_value)
random.seed(seed_value)

dict_learning_params = global_variables.DICT_LEARNING_PARAMETERS_PAPER
dict_other_params = global_variables.DICT_OTHER_PARAMETERS_PAPER

rows, cols = GRID_SIZE
dict_env_params = {'rows': rows, 'cols': cols, 'n_agents': N_AGENTS, 'n_enemies': N_ENEMIES, 'n_goals': N_GOALS,
                   'n_actions': global_variables.N_ACTIONS_PAPER, 'if_maze': False,
                   'value_reward_alive': global_variables.VALUE_REWARD_ALIVE_PAPER,
                   'value_reward_winner': global_variables.VALUE_REWARD_WINNER_PAPER,
                   'value_reward_loser': global_variables.VALUE_REWARD_LOSER_PAPER,
                   'seed_value': seed_value, 'enemies_actions': 'random', 'env_type': 'numpy',
                   'predefined_env': None}

dict_other_params['N_EPISODES'] = N_EPISODES

# Create an environment
env = CustomEnv(dict_env_params, False)

class_train = Training(dict_env_params, dict_learning_params, dict_other_params,
                       f'{label_kind_of_alg}',
                       f'{label_exploration_strategy}')

class_train.start_train(env, df_track=True)
df_track = class_train.get_df_track()
out_causal_table = CausalInference(df_track, N_AGENTS, N_ENEMIES, N_GOALS).return_causal_table()

out_causal_table.to_excel(f'{global_variables.GLOBAL_PATH_REPO}/mario.xlsx')
out_causal_table.to_pickle(f'{global_variables.PATH_CAUSAL_TABLE_OFFLINE}')
