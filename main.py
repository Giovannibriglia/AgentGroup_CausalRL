import numpy as np
from pandas.io import pickle
import env_game
import os
import glob
import models

algorithms = ['QL', 'CQL4', 'DQN']
n_games = 3
vect_rows = [10]
vect_n_enemies = [1]
n_episodes = 250
vect_if_maze = [False]
vect_if_same_enemies_actions = [False]
dir_start = 'Results'

os.makedirs(dir_start, exist_ok=True)
for if_maze in vect_if_maze:
    dir = dir_start
    if if_maze:
        dir += '/Maze'
    else:
        dir += '/Grid'
    os.makedirs(dir, exist_ok=True)
    for if_same_enemies_actions in vect_if_same_enemies_actions:
        if if_same_enemies_actions:
            dir += '/SameEnAct'
        else:
            dir += '/RandEnAct'
        os.makedirs(dir, exist_ok=True)
        for n_enemies in vect_n_enemies:
            dir += f'/{n_enemies}En'
            os.makedirs(dir, exist_ok=True)
            for rows in vect_rows:
                cols = rows
                dir += f'/{rows}x{cols}'
                os.makedirs(dir, exist_ok=True)

                for game_n in range(1, n_games+1, 1):
                    n_agents = 1
                    n_act_agents = 5
                    n_act_enemies = 5
                    n_goals = 1
                    env = env_game.CustomEnv(rows, cols, n_agents, n_act_agents, n_enemies, n_act_enemies, n_goals,
                                             if_maze, if_same_enemies_actions)

                    for alg in algorithms:
                        env_for_alg = env
                        rewards = []
                        steps = []
                        # returned: reward for episode and steps for episode
                        if alg == 'QL':
                            rewards, steps = models.QL(env_for_alg, n_act_agents, n_episodes)
                        elif alg == 'CQL4':
                            rewards, steps = models.CQL4()
                        elif alg == 'DQN':
                            rewards, steps = models.DQN()

                        np.save(f"{dir}/{alg}_rewards_game{game_n}.npy", rewards)
                        np.save(f"{dir}/{alg}_steps_game{game_n}.npy", steps)
