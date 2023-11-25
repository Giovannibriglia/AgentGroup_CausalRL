import numpy as np
import env_game
import os
import models

algorithms = ['DeepQNetwork']
n_games = 1
vect_rows = [5]
vect_n_enemies = [1]
n_episodes = 1000
vect_if_maze = [False]
vect_if_same_enemies_actions = [False]
dir_start = 'Results'

os.makedirs(dir_start, exist_ok=True)
for if_maze in vect_if_maze:
    directory = dir_start
    if if_maze:
        directory += '/Maze'
    else:
        directory += '/Grid'
    os.makedirs(directory, exist_ok=True)
    for if_same_enemies_actions in vect_if_same_enemies_actions:
        if if_same_enemies_actions:
            directory += '/SameEnAct'
        else:
            directory += '/RandEnAct'
        os.makedirs(directory, exist_ok=True)
        for n_enemies in vect_n_enemies:
            directory += f'/{n_enemies}En'
            os.makedirs(directory, exist_ok=True)
            for rows in vect_rows:
                cols = rows
                directory += f'/{rows}x{cols}'
                os.makedirs(directory, exist_ok=True)

                for game_n in range(1, n_games+1, 1):
                    n_agents = 1
                    n_act_agents = 5
                    n_act_enemies = 5
                    n_goals = 1
                    env = env_game.CustomEnv(rows, cols, n_agents, n_act_agents, n_enemies, n_act_enemies, n_goals,
                                             if_maze, if_same_enemies_actions)

                    for alg in algorithms:
                        print(f'\n*** {alg} ****')
                        env_for_alg = env
                        rewards = []
                        steps = []
                        # returned: reward for episode and steps for episode
                        if alg == 'QL':
                            rewards, steps = models.QL(env_for_alg, n_act_agents, n_episodes)
                        elif alg == 'CQL3':
                            rewards, steps = models.CQL3(env_for_alg, n_act_agents, n_episodes)
                        elif alg == 'CQL4':
                            rewards, steps = models.CQL4(env_for_alg, n_act_agents, n_episodes)
                        elif alg == 'DeepQNetwork':
                            rewards, steps = models.DeepQNetwork(env_for_alg, n_act_agents, n_episodes)

                        np.save(f"{directory}/{alg}_rewards_game{game_n}.npy", rewards)
                        np.save(f"{directory}/{alg}_steps_game{game_n}.npy", steps)
