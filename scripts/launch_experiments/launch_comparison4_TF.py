import numpy as np
import pandas as pd
from scripts.env import env_game
import os
from scripts.algorithms import models
import time
import random

seed_values = np.load('../utils/seed_values.npy')


def get_batch_episodes(n_enemies, rows):
    table = pd.read_pickle(
        '../../results/TradeOff_causality_batch_episodes_enemies/results_tradeoff_online_causality.pkl')

    condition = (table['Grid Size'] == rows) & (table['Enemies'] == n_enemies) & (table['Suitable'] == 'yes')
    result_column = table.loc[condition, 'Episodes'].to_list()
    try:
        batch = min(result_column)
        if batch is not None:
            return batch
        else:
            return 500
    except:
        return 500


def extract_paths_with_pattern(directory, pattern):
    paths_with_pattern = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if pattern in file_path:
                paths_with_pattern.append(file_path)
    return paths_with_pattern


def extract_variables_from_path(path, dir):
    # Split the path into parts
    path_parts = path.split(os.path.sep)

    # Check if dir is in the path
    if dir in path_parts:
        # Find the index of dir in the path
        index_results4 = path_parts.index(dir)

        # Extract variables starting from the part after dir
        variables = path_parts[index_results4 + 1:]

        # Filter out the pattern variable if present
        # variables = [variable for variable in variables if variable != pattern]

        # Create a dictionary with extracted variables
        """extracted_variables.append({
            f'Var{i + 1}': variables[i] if i < len(variables) else None
            for i in range(len(variables))
        })"""
        dict_out = {
            'Grid/Maze': variables[0] if len(variables) > 0 else None,
            'SameEnAct/RandEnAct': variables[1] if len(variables) > 1 else None,
            'n_enemies': variables[2] if len(variables) > 2 else None,
            'rows_x_cols': variables[3] if len(variables) > 3 else None,
            'game_n': int(variables[4].replace("env_game", "").replace(".npy", "")) if len(variables) > 4 else None
        }
        return dict_out
    else:
        print('Problem in reading ')
        return {}


def drop_last_folder_from_path(path):
    # Split the path into components
    path_components = os.path.split(path)

    # Remove the last component (last folder) and reconstruct the path
    new_path = os.path.join(*path_components[:-1])

    return new_path


def change_first_and_second_path_remove_last(input_string, new_first_component, new_second_component):
    input_string = input_string.replace("\\", "/")
    path_components = input_string.split('/')  # Adjust the separator based on your specific case

    # Change the first and second path components (if there are at least two components)
    if len(path_components) >= 2:
        path_components[0] = new_first_component
        path_components[1] = new_second_component

    # Remove the last path component
    path_components.pop()

    # Join the components back into a single path using backslashes
    result_path = '/'.join(path_components)  # Adjust the separator based on your specific case

    return result_path


# 'QL_EG', 'QL_SA', 'QL_BM', 'QL_TS' + all 'causal' + 'offline'/'online'
# 'DQN' + 'causal'
algorithms = ['QL_TS_basic', 'QL_TS_causal_offline', 'QL_TS_causal_online',
              'QL_EG_basic', 'QL_EG_causal_offline', 'QL_EG_causal_online',
              'QL_SA_basic', 'QL_SA_causal_offline', 'QL_SA_causal_online',
              'QL_BM_basic', 'QL_BM_causal_offline', 'QL_BM_causal_online'
              ]

n_episodes = 3000
vect_if_maze = [False]
dir_start = f'Env_Comparison123'
pattern_env_game = 'env_game'
dir_results = f'Results_Comparison4'
who_moves_first = 'Enemy'  # 'Enemy' or 'Agent'
n_games = 5

episodes_to_visualize = [0, int(n_episodes * 0.33), int(n_episodes * 0.66), n_episodes - 1]

paths_with_pattern = extract_paths_with_pattern(dir_start, pattern_env_game)

for path_game in paths_with_pattern:
    dict_vars = extract_variables_from_path(path_game, dir_start)

    n_enemies = dict_vars['n_enemies']
    if_maze = dict_vars['Grid/Maze']
    rows = dict_vars['rows_x_cols'][dict_vars['rows_x_cols'].find('x')+1:]
    cols = dict_vars['rows_x_cols'][:dict_vars['rows_x_cols'].find('x')]

    game_n = dict_vars['game_n']
    if_same_enemy_actions = dict_vars['SameEnAct/RandEnAct']

    predefined_env = np.load(path_game)

    path_game = change_first_and_second_path_remove_last(path_game, dir_results, 'Maze' if if_maze == 'Grid' else 'Grid')
    os.makedirs(path_game, exist_ok=True)

    components = path_game.split("/")
    numerical_part = components[0].split("_")[-1]
    components[0] = "Env_" + numerical_part
    directory_env = "/".join(components)
    os.makedirs(directory_env, exist_ok=True)

    seed_value = seed_values[game_n]
    np.random.seed(seed_value)
    random.seed(seed_value)

    n_agents = 1
    n_act_agents = 5
    n_act_enemies = 5
    n_goals = 1

    env = env_game.CustomEnv(rows=rows, cols=cols, n_agents=n_agents, n_act_agents=n_act_agents,
                             n_enemies=n_enemies, n_act_enemies=n_act_enemies, n_goals=n_goals,
                             if_maze=True if if_maze == 'Grid' else False,
                             if_same_enemies_actions=True if if_same_enemy_actions == 'SameEnAct' else False,
                             dir_saving=path_game, game_n=game_n,
                             seed_value=seed_value, predefined_env=predefined_env)

    np.save(f"{path_game}/env_game{game_n}.npy", env.grid_for_game)
    np.save(f"{directory_env}/env_game{game_n}.npy", env.grid_for_game)

    BATCH_EPISODES_UPDATE_BN = get_batch_episodes(n_enemies, rows)

    for alg in algorithms:
        print(f'\n*** {alg} - Game {game_n}/{n_games} ****')
        time.sleep(1)

        start_time = time.time()

        env_for_alg = env
        rewards = []
        steps = []

        dir_q_table = f'{dir_start}/{if_maze}/{if_same_enemy_actions}/{n_enemies}/{rows}x{cols}'
        if 'TS' in alg:
            alpha = np.load(f'{dir_q_table}/{alg}_alpha_game{game_n}.npy')
            beta = np.load(f'{dir_q_table}/{alg}_beta_game{game_n}.npy')
            predefined_q_table = [alpha, beta]
        else:
            predefined_q_table = np.load(f'{dir_q_table}/{alg}_q_table_game{game_n}.npy')

        # returned: reward for episode, actions for episode and the final Q-table
        if 'QL' in alg:
            if 'offline' in alg or 'basic' in alg:
                rewards, steps, q_table = models.QL_causality_offline(env_for_alg, n_act_agents,
                                                                      n_episodes,
                                                                      alg, who_moves_first,
                                                                      episodes_to_visualize,
                                                                      seed_value,
                                                                      predefined_q_table)
            else:
                rewards, steps, q_table = models.QL_causality_online(env_for_alg, n_act_agents,
                                                                     n_episodes,
                                                                     alg, who_moves_first,
                                                                     episodes_to_visualize,
                                                                     seed_value,
                                                                     BATCH_EPISODES_UPDATE_BN,
                                                                     predefined_q_table)

        else:
            rewards, steps, q_table = models.DQNs(env_for_alg, n_act_agents, n_episodes,
                                                  alg, who_moves_first, episodes_to_visualize,
                                                  seed_value, predefined_q_table)

        if len(rewards) == n_episodes:
            computation_time = (time.time() - start_time) / 60  # minutes
        else:
            computation_time = 'timeout'

        np.save(f"{path_game}/{alg}_rewards_game{game_n}.npy", rewards)
        np.save(f"{path_game}/{alg}_steps_game{game_n}.npy", steps)
        np.save(f'{path_game}/{alg}_computation_time_game{game_n}.npy', computation_time)

        if 'TS' in alg:
            alpha, beta = q_table[0], q_table[1]
            np.save(f'{path_game}/{alg}_alpha_game{game_n}.npy', alpha)
            np.save(f'{path_game}/{alg}_beta_game{game_n}.npy', beta)
            np.save(f'{directory_env}/{alg}_alpha_game{game_n}.npy', alpha)
            np.save(f'{directory_env}/{alg}_beta_game{game_n}.npy', beta)
        else:
            np.save(f'{path_game}/{alg}_q_table_game{game_n}.npy', q_table)
            np.save(f'{directory_env}/{alg}_q_table_game{game_n}.npy', q_table)

    # plots.plot_av_rew_steps(directory, algorithms, n_games, n_episodes, rows, cols, n_enemies)
    # plots.plot_av_computation_time(directory, algorithms, n_games, rows, cols, n_enemies)
