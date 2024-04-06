import os
import networkx as nx
from causalnex.structure import StructureModel
from matplotlib import pyplot as plt
import global_variables
from scripts.algorithms.causal_discovery import CausalDiscovery
from scripts.utils.environment import CustomEnv
from scripts.utils.train_models import Training
import json

"""The purpose of this script is to generate the 'ground truth' for the causal table and for the causal graph. 
Recognizing its critical importance, we conducted a substantial number of episodes to ensure its accuracy. 
Additionally, for the specific environment in question, we selected a 3x3 grid layout with one enemy and one goal. 
The agent navigates this grid using a random policy. This simulation has been conducted for the specified number of 
scenarios defined by global_variables.N_SIMULATIONS_PAPER, aiming to gather a diverse range of environmental 
configurations.

From each simulation, we extracted both the causal graph and the causal table. These pieces of information were then 
meticulously processed and merged (details of which are pending clarification from *** @gio***) to derive the final 
versions of the causal graph and causal table used in the offline causal settings. This process embodies the offline 
causal discovery.

This simulation was prompted by a challenge encountered during the extraction of causal dependencies. Namely, 
when the agent encounters a wall, it executes a non-zero action, despite resulting in deltaX and deltaY being zero. 
This inconsistency arises because the action taken is not zero. By experimenting with multiple 3x3 grids featuring 
the goal placed in different cells and subsequently merging the resultant causal tables and graphs, we hope to 
address and resolve this issue effectively."""

DIR_SAVING = 'OfflineCD_MultiEnv'


N_AGENTS = 1
N_ENEMIES = 1
N_GOALS = 1
GRID_SIZE = (3, 3)
N_EPISODES = 3000
N_SIMULATIONS = global_variables.N_SIMULATIONS_PAPER

label_kind_of_alg = global_variables.LABEL_RANDOM_AGENT
label_exploration_strategy = global_variables.LABEL_RANDOM_AGENT

DIR_SAVING = f'{global_variables.GLOBAL_PATH_REPO}/Results/{DIR_SAVING}'
os.makedirs(DIR_SAVING, exist_ok=True)

# generate data
rows, cols = GRID_SIZE
for sim_n in range(N_SIMULATIONS):
    seed_value = global_variables.seed_values[sim_n]

    dict_learning_params = global_variables.DICT_LEARNING_PARAMETERS_PAPER
    dict_other_params = global_variables.DICT_OTHER_PARAMETERS_PAPER

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

    out_causal_table.to_pickle(f'{DIR_SAVING}/causal_table_{sim_n}.pkl')
    out_causal_table.to_excel(f'{DIR_SAVING}/causal_table_{sim_n}.xlsx')

    with open(f'{DIR_SAVING}/causal_graph_{sim_n}.json', 'w') as json_file:
        json.dump(out_causal_graph, json_file)

    fig = plt.figure(dpi=1000)
    sm = StructureModel()
    sm.add_edges_from(out_causal_graph)
    plt.title(f'Causal graph ground truth', fontsize=16)
    nx.draw(sm, with_labels=True, font_size=7, arrowsize=30, arrows=True,
            edge_color='orange', node_size=1000, font_weight='bold', pos=nx.circular_layout(sm))
    plt.savefig(f'{DIR_SAVING}/causal_graph_{sim_n}.png')
    plt.show()
    plt.close(fig)

# evaluate data
# TODO: merge causal graphs and merge causal table in some way

