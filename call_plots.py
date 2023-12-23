import plots
import os

algorithms = ['QL_EG', 'QL_SA', 'QL_BM', 'QL_TS',
              'QL_EG_causal', 'QL_SA_causal', 'QL_BM_causal', 'QL_TS_causal']
n_games = 10
vect_rows = [5, 10]
vect_n_enemies = [2, 5, 10]
n_episodes = 2500
vect_if_maze = [False]
vect_if_same_enemies_actions = [False]
dir_start = f'Baseline_and_Comp1'
who_moves_first = 'Enemy'  # 'Enemy' or 'Agent'

os.makedirs(dir_start, exist_ok=True)
for if_maze in vect_if_maze:
    if if_maze:
        env_name = 'Maze'
    else:
        env_name = 'Grid'
    directory = dir_start + f'/{env_name}'
    os.makedirs(directory, exist_ok=True)

    for if_same_enemies_actions in vect_if_same_enemies_actions:
        if if_same_enemies_actions:
            en_act = 'SameEnAct'
        else:
            en_act = 'RandEnAct'
        directory = dir_start + f'/{env_name}'+ f'/{en_act}'
        os.makedirs(directory, exist_ok=True)
        for n_enemies in vect_n_enemies:
            directory = dir_start + f'/{env_name}' + f'/{en_act}' + f'/{n_enemies}Enem'
            os.makedirs(directory, exist_ok=True)
            for rows in vect_rows:
                cols = rows
                directory = dir_start + f'/{env_name}' + f'/{en_act}' + f'/{n_enemies}Enem' + f'/{rows}x{cols}'
                os.makedirs(directory, exist_ok=True)

                plots.plot_av_rew_steps(directory, algorithms, n_games, n_episodes, rows, cols, n_enemies)
                plots.plot_av_computation_time(directory, algorithms, n_games, rows, cols, n_enemies)