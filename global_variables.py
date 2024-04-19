import os
from itertools import product
import palettable
import numpy as np
import pandas as pd

# TODO: RIORDINARE COME DIO COMANDA


GLOBAL_PATH_REPO = os.path.dirname(os.path.abspath(__file__))
seed_values = np.load(f'{GLOBAL_PATH_REPO}/scripts/utils/seed_values.npy')
N_ACTIONS_PAPER = 5
DICT_IMPLEMENTED_ACTIONS = {0: np.array([0, 0]),  # stop
                            1: np.array([1, 0]),  # down
                            2: np.array([-1, 0]),  # up
                            3: np.array([0, 1]),  # right
                            4: np.array([0, -1])}  # left

N_TRAINING_EPISODES = 3000
N_SIMULATIONS_PAPER = 10
GRID_SIZES_CONSIDERED_PAPER = [(10, 10), (5, 5)]
N_ENEMIES_CONSIDERED_PAPER = [10, 5, 2]
EPISODES_TO_VISUALIZE_PAPER = [0,
                               int(N_TRAINING_EPISODES / 3),
                               int(N_TRAINING_EPISODES * 0.66),
                               N_TRAINING_EPISODES - 1]

N_EPISODES_CONSIDERED_FOR_SENSITIVE_ANALYSIS_PAPER = [100, 250, 500, 1000]

LABEL_RANDOM_AGENT = 'random'
LABEL_Q_LEARNING = 'QL'
LABEL_DQN = 'DQN'
LABEL_VANILLA = 'vanilla'
LABEL_CAUSAL = 'causal'
LABEL_CAUSAL_OFFLINE = f'{LABEL_CAUSAL}_offline'
LABEL_CAUSAL_ONLINE = f'{LABEL_CAUSAL}_online'

LIST_IMPLEMENTED_ALGORITHMS = [f'{LABEL_RANDOM_AGENT}',
                               f'{LABEL_Q_LEARNING}_{LABEL_VANILLA}', f'{LABEL_Q_LEARNING}_{LABEL_CAUSAL_OFFLINE}',
                               f'{LABEL_Q_LEARNING}_{LABEL_CAUSAL_ONLINE}',
                               f'{LABEL_DQN}_{LABEL_VANILLA}', f'{LABEL_DQN}_{LABEL_CAUSAL_OFFLINE}']

LABEL_EPSILON_GREEDY = 'EG'
LABEL_THOMPSON_SAMPLING = 'TS'
LABEL_BOLTZMANN_MACHINE = 'BM'
LABEL_SOFTMAX_ANNEALING = 'SA'

LIST_IMPLEMENTED_EXPLORATION_STRATEGIES = [f'{LABEL_EPSILON_GREEDY}', f'{LABEL_THOMPSON_SAMPLING}',
                                           f'{LABEL_SOFTMAX_ANNEALING}', f'{LABEL_BOLTZMANN_MACHINE}']

GROUPS_IMPLEMENTED_ALG_EXP_STR = [f'{algo}_{exp_str}'
                                  for algo in LIST_IMPLEMENTED_ALGORITHMS
                                  if algo != LABEL_RANDOM_AGENT
                                  for exp_str in LIST_IMPLEMENTED_EXPLORATION_STRATEGIES]

# TODO: CHANGE VALUES
palettes = {
    LABEL_EPSILON_GREEDY: palettable.colorbrewer.sequential.Blues_9,
    LABEL_THOMPSON_SAMPLING: palettable.colorbrewer.sequential.Greens_9,
    LABEL_SOFTMAX_ANNEALING: palettable.colorbrewer.sequential.Oranges_9,
    LABEL_BOLTZMANN_MACHINE: palettable.colorbrewer.sequential.Reds_9
}
COLORS_ALGORITHMS = {}
combinations = product(LIST_IMPLEMENTED_ALGORITHMS[1:], LIST_IMPLEMENTED_EXPLORATION_STRATEGIES)
for algo, exp_str in combinations:
    palette = palettes[exp_str]
    algo_exp_str = f'{algo}_{exp_str}'
    palette_index = LIST_IMPLEMENTED_ALGORITHMS.index(algo) - 1  # Adjust index since we start from index 1 in LIST_IMPLEMENTED_ALGORITHMS
    color_index = LIST_IMPLEMENTED_EXPLORATION_STRATEGIES.index(exp_str)
    color = palette.mpl_colors[(palette_index + color_index) % len(palette.mpl_colors)]  # Wrap around the palette
    COLORS_ALGORITHMS[algo_exp_str] = color


KEY_METRIC_REWARDS_EPISODE = 'rewards_for_episodes'
KEY_METRICS_STEPS_EPISODE = 'steps_for_episode'
KEY_METRIC_TIME_EPISODE = 'time_for_episode'
KEY_METRIC_TIMEOUT_CONDITION = 'if_timeout_occurred'
KEY_METRIC_Q_TABLE = 'q_table'

PATH_IMAGES_FOR_RENDER = f'{GLOBAL_PATH_REPO}/scripts/utils/images_for_render'

DICT_LEARNING_PARAMETERS_PAPER = {'GAMMA': 0.99, 'LEARNING_RATE': 0.0001,
                                  'START_EXPLORATION_PROBABILITY': 1, 'MIN_EXPLORATION_PROBABILITY': 0.01,
                                  'EXPLORATION_GAME_PERCENT': 0.6,
                                  'BATCH_SIZE': 64, 'TAU': 0.005, 'HIDDEN_LAYERS': 128, 'REPLAY_MEMORY_CAPACITY': 10000,
                                  'KNOWLEDGE_TRANSFERRED': None}

DICT_OTHER_PARAMETERS_PAPER = {'WHO_MOVES_FIRST': 'enemy',
                               'TIMEOUT_IN_HOURS': 4,
                               'KIND_TH_CHECKS_CAUSAL_INFERENCE': 'consecutive',
                               'TH_CHECKS_CAUSAL_INFERENCE': 3,
                               'N_EPISODES': N_TRAINING_EPISODES}

" ******************************************************************************************************************** "

# For environment script
KEY_SAME_ENEMY_ACTIONS = 'same_sequence'
KEY_RANDOM_ENEMY_ACTIONS = 'random'

VALUE_WALL_CELL = -2
VALUE_ENEMY_CELL = -1
VALUE_EMPTY_CELL = 0
VALUE_AGENT_CELL = 1
VALUE_GOAL_CELL = 2

N_WALLS_COEFFICIENT = 2

VALUE_ENTITY_FAR = 50

LEN_PREDEFINED_ENEMIES_ACTIONS = 20

DELAY_VISUALIZATION_VIDEO = 1
FPS_video = 3

" ******************************************************************************************************************** "

# For causal table
LABEL_ENEMY_CAUSAL_TABLE = 'Enemy'
LABEL_AGENT_CAUSAL_TABLE = 'Agent'
LABEL_GOAL_CAUSAL_TABLE = 'Goal'
LABEL_NEARBY_CAUSAL_TABLE = 'Nearby'
LABEL_COL_REWARD = 'Reward'
LABEL_COL_ACTION = 'Action'
LABEL_COL_DELTAX = 'DeltaX'
LABEL_COL_DELTAY = 'DeltaY'

COL_NEARBY_GOAL = f'{LABEL_ENEMY_CAUSAL_TABLE}0_Nearby_{LABEL_AGENT_CAUSAL_TABLE}0'
COL_NEARBY_ENEMY = f'{LABEL_GOAL_CAUSAL_TABLE}0_Nearby_{LABEL_AGENT_CAUSAL_TABLE}0'


def define_columns_causal_table(n_agents: int, n_enemies: int, n_goals: int) -> list:
    cols = []
    for agent in range(n_agents):
        cols.append(f'{LABEL_COL_DELTAX}_{LABEL_AGENT_CAUSAL_TABLE}{agent}')
        cols.append(f'{LABEL_COL_DELTAY}_{LABEL_AGENT_CAUSAL_TABLE}{agent}')
        cols.append(f'{LABEL_COL_REWARD}_{LABEL_AGENT_CAUSAL_TABLE}{agent}')
        cols.append(f'{LABEL_COL_ACTION}_{LABEL_AGENT_CAUSAL_TABLE}{agent}')
        for enemy in range(n_enemies):
            cols.append(
                f'{LABEL_ENEMY_CAUSAL_TABLE}{enemy}_{LABEL_NEARBY_CAUSAL_TABLE}_{LABEL_AGENT_CAUSAL_TABLE}{agent}')
        for goal in range(n_goals):
            cols.append(
                f'{LABEL_GOAL_CAUSAL_TABLE}{goal}_{LABEL_NEARBY_CAUSAL_TABLE}_{LABEL_AGENT_CAUSAL_TABLE}{agent}')
    return cols


VALUE_REWARD_LOSER_PAPER = -1
VALUE_REWARD_ALIVE_PAPER = 0
VALUE_REWARD_WINNER_PAPER = 1

PATH_CAUSAL_TABLE_OFFLINE = f'{GLOBAL_PATH_REPO}/scripts/utils/ground_truth_causal_table.pkl'
PATH_CAUSAL_GRAPH_OFFLINE = f'{GLOBAL_PATH_REPO}/scripts/utils/ground_truth_causal_graph.json'
PATH_IMG_CAUSAL_GRAPH_OFFLINE = f'{GLOBAL_PATH_REPO}/scripts/utils/ground_truth_causal_graph.png'
PATH_RESULTS_BATCH_EPISODES_ONLINE_CD = f'{GLOBAL_PATH_REPO}/scripts/utils/batch_episodes_for_online_cd.pkl'


def get_possible_actions(causal_table: pd.DataFrame,
                         enemies_nearby: np.ndarray = None, goals_nearby: np.ndarray = None) -> list:
    """possible_actions = np.arange(0, self.n_actions, 1)
    goals_nearby = [s for s in list(set(goals_nearby)) if s != 50]
    enemies_nearby = [s for s in list(set(enemies_nearby)) if s != 50]
    if len(goals_nearby) > 0:
        possible_actions = goals_nearby
    elif len(enemies_nearby) > 0:
        possible_actions = [s for s in possible_actions if s not in enemies_nearby]

    return possible_actions"""
    try:
        if enemies_nearby is not None:
            enemies_nearby = list(set(enemies_nearby))
        if goals_nearby is not None:
            goals_nearby = list(set(goals_nearby))

        col_action = next(s for s in causal_table.columns if LABEL_COL_ACTION in s)
        col_reward = next(s for s in causal_table.columns if LABEL_COL_REWARD in s)
        col_enemy_nearby = next(s for s in causal_table.columns if LABEL_ENEMY_CAUSAL_TABLE in s and
                                LABEL_NEARBY_CAUSAL_TABLE in s)
        col_goal_nearby = next(s for s in causal_table.columns if
                               LABEL_GOAL_CAUSAL_TABLE in s and LABEL_NEARBY_CAUSAL_TABLE in s)

        if enemies_nearby is not None and goals_nearby is not None:
            filtered_rows = causal_table[(causal_table[col_goal_nearby].isin(goals_nearby)) &
                                         (causal_table[col_enemy_nearby].isin(enemies_nearby))]
        elif enemies_nearby is not None:
            filtered_rows = causal_table[(causal_table[col_goal_nearby].isin([50])) &
                                         (causal_table[col_enemy_nearby].isin(enemies_nearby))]
        elif goals_nearby is not None:
            filtered_rows = causal_table[(causal_table[col_goal_nearby].isin(goals_nearby)) &
                                         (causal_table[col_enemy_nearby].isin([50]))]
        else:
            filtered_rows = causal_table

        max_achievable_reward = filtered_rows[col_reward].max()
        filtered_max_reward = filtered_rows[filtered_rows[col_reward] == max_achievable_reward]
        # Group by action and calculate average rewards
        grouped = filtered_max_reward.groupby([col_reward, col_enemy_nearby, col_goal_nearby])[col_action]
        # Initialize a variable to hold the common values
        possible_actions = None
        # Iterate over the groups
        for name, group in grouped:
            # If it's the first group, initialize common_values with the values of the first group
            if possible_actions is None:
                possible_actions = set(group)
            # Otherwise, take the intersection of common_values and the values of the current group
            else:
                possible_actions = possible_actions.intersection(group)
        if possible_actions is not None:
            possible_actions = list(possible_actions)
        else:
            possible_actions = []

        return possible_actions
    except:
        print('error in "get_possible_actions" function')
        return []