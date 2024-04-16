import networkx as nx
from causalnex.structure import StructureModel
from matplotlib import pyplot as plt
import global_variables
from scripts.algorithms.causal_discovery import CausalDiscovery
from scripts.utils.environment import CustomEnv
from scripts.utils.train_models import Training
import json

"""
The aim of this script is to produce the 'ground truth' for the causal table. Given its significance, we opted for
 a substantial number of episodes to ensure its adequacy. Additionally, for the environment under consideration,
 we selected a 3x3 grid with one enemy and one goal. Lastly, the agent navigates using a random policy.

The result of this simulation is the causal table used in algorithms incorporating offline causal discovery.
"""
# TODO: Extracting causal dependencies encounters a challenge: when the agent encounters a wall, it executes an action
#  (nonzero) despite resulting in deltaX and deltaY being 0. This discrepancy arises because the action isn't 0.
#  A potential solution involves incorporating the impact of walls as a distinct feature termed "Wall_Nearby_Agent"
#  in the modeling process.
N_AGENTS = 1
N_ENEMIES = 1
N_GOALS = 1
GRID_SIZE = (8, 8)
N_EPISODES = global_variables.N_TRAINING_EPISODES

label_kind_of_alg = global_variables.LABEL_RANDOM_AGENT
label_exploration_strategy = global_variables.LABEL_RANDOM_AGENT

seed_value = global_variables.seed_values[0]

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

env = CustomEnv(dict_env_params)

class_train = Training(dict_env_params, dict_learning_params, dict_other_params,
                       f'{label_kind_of_alg}',
                       f'{label_exploration_strategy}')

class_train.start_train(env, batch_update_df_track=1000)
df_track = class_train.get_df_track()

cd = CausalDiscovery(df_track, N_AGENTS, N_ENEMIES, N_GOALS)
out_causal_table = cd.return_causal_table()
out_causal_graph = cd.return_causal_graph()

out_causal_table.to_pickle(f'{global_variables.GLOBAL_PATH_REPO}/out_causal_table_8x8.pkl')

with open(f'{global_variables.GLOBAL_PATH_REPO}/out_causal_graph_8x8.json', 'w') as json_file:
    json.dump(out_causal_graph, json_file)

fig = plt.figure(dpi=1000)
sm = StructureModel()
sm.add_edges_from(out_causal_graph)
plt.title(f'Causal graph ground truth', fontsize=16)
nx.draw(sm, with_labels=True, font_size=7, arrowsize=30, arrows=True,
        edge_color='orange', node_size=1000, font_weight='bold', pos=nx.circular_layout(sm))
plt.show()
