import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings("ignore")

# discounted factor
GAMMA = 0.99
# learning rate
LEARNING_RATE = 0.01
# how many tries the system can do in the exploration
exploration_actions_threshold = 10

causal_table = pd.read_pickle('final_causal_table.pkl')


def causal_model_movement(causal_table, action, oldX, oldY, rows, cols):
    action = 'Action_Agent1_Act' + str(action)

    row_causal_table = causal_table.loc[action, :]

    newX = oldX
    newY = oldY

    for col in range(len(row_causal_table)):
        if 'StateX_Agent' in row_causal_table.index[col]:
            if 0 <= oldX + row_causal_table[col] <= cols - 1:
                newX += row_causal_table[col]
        if 'StateY_Agent' in row_causal_table.index[col]:
            if 0 <= oldY + row_causal_table[col] <= rows - 1:
                newY += row_causal_table[col]

    return newX, newY


def causal_model_gameover(causal_table, possible_actions_in, nearbies, attached):
    possible_actions = possible_actions_in

    rows_colGameOverIsTrue = causal_table[causal_table['GameOver_Agent1'] == 1]

    for nearby in nearbies:
        if nearby != 50 and nearby != 0:
            input = ['EnemiesNearby_Agent1_Dir' + str(nearby)]
            row_nearby = rows_colGameOverIsTrue[input] == 1
            index_colGameOver_int = np.where((row_nearby[input] == True).values == True)[0][0]
            row_GameOver = causal_table.loc[row_nearby.index[index_colGameOver_int], :]

            for action in possible_actions:
                if row_GameOver['Action_Agent1_Act' + str(action)] == 1:
                    possible_actions.remove(action)

    if attached:
        input = ['EnemiesAttached_Agent1']

        row_nearby = rows_colGameOverIsTrue[input] == 1
        index_colGameOver_int = np.where((row_nearby[input] == True).values == True)[0][0]
        row_GameOver = causal_table.loc[row_nearby.index[index_colGameOver_int], :]

        for action in possible_actions:
            if row_GameOver['Action_Agent1_Act' + str(action)] == 1:
                possible_actions.remove(action)

    return possible_actions


def QL(env, n_act_agents, n_episodes):
    rows = env.rows
    cols = env.cols
    Q_table = np.zeros((rows, cols, n_act_agents))

    exploration_proba = 1
    min_exploration_proba = 0.01
    exploration_decreasing_decay = -np.log(min_exploration_proba) / (0.6 * n_episodes)

    average_episodes_rewards = []
    steps_for_episode = []

    for e in tqdm(range(n_episodes)):
        agent = 0
        if e == 0:
            res_loser = True
        else:
            res_loser = False
        current_state, rewards, dones, enemies_nearby_all_agents, enemies_attached_all_agents = env.reset(res_loser)
        current_stateX = current_state[agent][0]
        current_stateY = current_state[agent][1]

        total_episode_reward = 0
        step_for_episode = 0
        done = False

        while not done:
            current_stateX = env.pos_agents[-1][agent][0]
            current_stateY = env.pos_agents[-1][agent][1]

            # print('current acquired', [current_stateX, current_stateY])
            step_for_episode += 1
            " epsiilon-greedy "
            if np.random.uniform(0, 1) < exploration_proba:  # exploration
                action = env.action_space.sample()
            else:  # exploitation
                action = np.argmax(Q_table[current_stateX, current_stateY, :])

            result = env.step_agent(action)
            # print('result:', result)
            next_stateX = int(result[0][agent][0])
            next_stateY = int(result[0][agent][0])
            reward = int(result[1][agent])
            done = result[2][agent]  # If agent wins, end loop and restart
            if_lose = result[3]

            Q_table[current_stateX, current_stateY, action] = (1 - LEARNING_RATE) * Q_table[
                current_stateX, current_stateY, action] + LEARNING_RATE * (reward + GAMMA * max(
                Q_table[next_stateX, next_stateY, :]))
            total_episode_reward = total_episode_reward + reward

            if if_lose:
                current_state, _, _, _, _ = env.reset(reset_n_times_loser=False)
                current_stateX = current_state[agent][0]
                current_stateY = current_state[agent][1]

            if (abs(current_stateX - next_stateX) + abs(current_stateY - next_stateY)) > 1 and n_act_agents < 5:
                print('movement control problem:', [current_stateX, current_stateY], [next_stateX, next_stateY])

        average_episodes_rewards.append(total_episode_reward)
        steps_for_episode.append(step_for_episode)
        # updating the exploration proba using exponential decay formula
        exploration_proba = max(min_exploration_proba, np.exp(-exploration_decreasing_decay * e))

    print(f'Average reward: {np.mean(average_episodes_rewards)}')
    return average_episodes_rewards, steps_for_episode


def CQL3(env, n_act_agents, n_episodes):

    rows = env.rows
    cols = env.cols
    # initialize the Q-Table
    Q_table = np.zeros((rows, cols, n_act_agents))  # x, y, actions
    # initialize table to keep track of the explored states
    Q_table_track = np.zeros((rows, cols))  # x, y, actions

    exploration_proba = 1
    min_exploration_proba = 0.01
    exploration_decreasing_decay = -np.log(min_exploration_proba) / (0.6 * n_episodes)

    average_episodes_rewards = []
    steps_for_episode = []

    for e in tqdm(range(n_episodes)):
        agent = 0
        if e == 0:
            res_loser = True
        else:
            res_loser = False
        current_state, rewards, dones, enemies_nearby_all_agents, enemies_attached_all_agents = env.reset(res_loser)
        current_stateX = current_state[agent][0]
        current_stateY = current_state[agent][1]

        total_episode_reward = 0
        step_for_episode = 0
        done = False

        while not done:
            current_stateX = env.pos_agents[-1][agent][0]
            current_stateY = env.pos_agents[-1][agent][1]

            # print('current acquired', [current_stateX, current_stateY])
            step_for_episode += 1
            " epsiilon-greedy "
            if np.random.uniform(0, 1) < exploration_proba:  # exploration
                enemies_nearby_agent = enemies_nearby_all_agents[agent]
                enemies_attached_agent = enemies_attached_all_agents[agent]
                possible_actions = [s for s in range(n_act_agents)]

                possible_actions = causal_model_gameover(causal_table, possible_actions, enemies_nearby_agent,
                                                         enemies_attached_agent)

                new = False
                check_tries = 0
                if len(possible_actions) > 0:
                    while not new:
                        action = random.sample(possible_actions, 1)[0]
                        next_stateX, next_stateY = causal_model_movement(causal_table, action, current_stateY,
                                                                         current_stateX, rows, cols)

                        if Q_table_track[next_stateX, next_stateY] == 0:
                            new = True
                            reward = 0  # reward to have found a new state
                            Q_table_track[next_stateX, next_stateY] = 1
                        else:
                            check_tries += 1
                            if check_tries == exploration_actions_threshold:
                                new = True
                                reward = 0  # reward to have not found a new state
                                action = random.sample(possible_actions, 1)[0]
                else:
                    action = env.action_space.sample()

                next_stateX, next_stateY = causal_model_movement(causal_table, action, current_stateY, current_stateX, rows, cols)

                if enemies_nearby_agent[0] == action and len(possible_actions) > 0:  # for debugging
                    print('Enemies nearby: ', enemies_nearby_agent)
                    print('Enemies attached: ', enemies_attached_agent)
                    print('Possible actions: ', possible_actions)
                    print('Exploration: problem in action selection with nearby model -> action taken', action)

                # additional Q-Table udpate
                Q_table[current_stateX, current_stateY, action] = (1 - LEARNING_RATE) * Q_table[
                    current_stateX, current_stateY, action] + LEARNING_RATE * (reward + GAMMA * max(
                    Q_table[next_stateX, next_stateY, :]))

            else:  # exploitation
                enemies_nearby_agent = enemies_nearby_all_agents[agent]
                enemies_attached_agent = enemies_attached_all_agents[agent]
                possible_actions = [s for s in range(n_act_agents)]

                possible_actions = causal_model_gameover(causal_table, possible_actions, enemies_nearby_agent,
                                                         enemies_attached_agent)

                if len(possible_actions) > 0:
                    max_value = max(Q_table[current_stateX, current_stateY, possible_actions])

                    actions_wrong = []
                    for act_test in [s for s in range(n_act_agents)]:
                        if act_test not in possible_actions:
                            actions_wrong.append(act_test)
                    actions_wrong = list(set(actions_wrong))
                    q_table_values = list(
                        np.array(np.where(Q_table[current_stateX, current_stateY, :] == max_value)[0]))

                    possibilities = []
                    for act_test in q_table_values:
                        if act_test not in actions_wrong:
                            possibilities.append(act_test)
                    possibilities = list(set(possibilities))
                    action = random.sample(possibilities, k=1)[0]

                    if action in enemies_nearby_agent:
                        print('Exploitation mod problem')
                        print('Enemies nearby: ', enemies_nearby_agent)
                        print('Enemies attached: ', enemies_attached_agent)
                        print(f'Possible actions: {possible_actions}, possibilities: {possibilities}')
                        print('Wrong actions ', actions_wrong)
                        print('Q-table situation: ', np.mean(Q_table[current_stateX, current_stateY, possible_actions]))
                        print('Action taken', action)

            result = env.step_agent(action)
            # print('result:', result)
            next_stateX = int(result[0][agent][0])
            next_stateY = int(result[0][agent][0])
            reward = int(result[1][agent])
            done = result[2][agent]  # If agent wins, end loop and restart
            if_lose = result[3]

            Q_table[current_stateX, current_stateY, action] = (1 - LEARNING_RATE) * Q_table[
                current_stateX, current_stateY, action] + LEARNING_RATE * (reward + GAMMA * max(
                Q_table[next_stateX, next_stateY, :]))
            total_episode_reward = total_episode_reward + reward

            if if_lose:
                current_state, _, _, _, _ = env.reset(reset_n_times_loser=False)
                current_stateX = current_state[agent][0]
                current_stateY = current_state[agent][1]

            if (abs(current_stateX - next_stateX) + abs(current_stateY - next_stateY)) > 1 and n_act_agents < 5:
                print('movement control problem:', [current_stateX, current_stateY], [next_stateX, next_stateY])

        average_episodes_rewards.append(total_episode_reward)
        steps_for_episode.append(step_for_episode)
        # updating the exploration proba using exponential decay formula
        exploration_proba = max(min_exploration_proba, np.exp(-exploration_decreasing_decay * e))

    print(f'Average reward: {np.mean(average_episodes_rewards)}')
    return average_episodes_rewards, steps_for_episode


def CQL4(env, n_act_agents, n_episodes):

    rows = env.rows
    cols = env.cols
    Q_table = np.zeros((rows, cols, n_act_agents))

    exploration_proba = 1
    min_exploration_proba = 0.01
    exploration_decreasing_decay = -np.log(min_exploration_proba) / (0.6 * n_episodes)

    average_episodes_rewards = []
    steps_for_episode = []

    for e in tqdm(range(n_episodes)):
        agent = 0
        if e == 0:
            res_loser = True
        else:
            res_loser = False
        current_state, rewards, dones, enemies_nearby_all_agents, enemies_attached_all_agents = env.reset(res_loser)
        current_stateX = current_state[agent][0]
        current_stateY = current_state[agent][1]

        total_episode_reward = 0
        step_for_episode = 0
        done = False

        while not done:
            current_stateX = env.pos_agents[-1][agent][0]
            current_stateY = env.pos_agents[-1][agent][1]

            # print('current acquired', [current_stateX, current_stateY])
            step_for_episode += 1
            " epsiilon-greedy "
            if np.random.uniform(0, 1) < exploration_proba:  # exploration
                enemies_nearby_agent = enemies_nearby_all_agents[agent]
                enemies_attached_agent = enemies_attached_all_agents[agent]
                possible_actions = [s for s in range(n_act_agents)]

                possible_actions = causal_model_gameover(causal_table,
                                                         possible_actions,
                                                         enemies_nearby_agent,
                                                         enemies_attached_agent)

                if len(possible_actions) > 0:
                    action = random.sample(possible_actions, 1)[0]
                else:
                    action = env.action_space.sample()

                next_stateX, next_stateY = causal_model_movement(causal_table, action, current_stateY, current_stateX, rows, cols)

                if enemies_nearby_agent[0] == action and len(possible_actions) > 0:
                    print('Enemies nearby: ', enemies_nearby_agent)
                    print('Enemies attached: ', enemies_attached_agent)
                    print('Possible actions: ', possible_actions)
                    print('Exploration: problem in action selection with nearby model -> action taken', action)

                reward = 0
                # additional Q-Table udpate
                Q_table[current_stateX, current_stateY, action] = (1 - LEARNING_RATE) * Q_table[
                    current_stateX, current_stateY, action] + LEARNING_RATE * (reward + GAMMA * max(
                    Q_table[next_stateX, next_stateY, :]))
            else:  # exploitation
                enemies_nearby_agent = enemies_nearby_all_agents[agent]
                enemies_attached_agent = enemies_attached_all_agents[agent]
                possible_actions = [s for s in range(n_act_agents)]

                possible_actions = causal_model_gameover(causal_table, possible_actions, enemies_nearby_agent,
                                                         enemies_attached_agent)

                if len(possible_actions) > 0:
                    max_value = max(Q_table[current_stateX, current_stateY, possible_actions])

                    actions_wrong = []
                    for act_test in [s for s in range(n_act_agents)]:
                        if act_test not in possible_actions:
                            actions_wrong.append(act_test)
                    actions_wrong = list(set(actions_wrong))
                    q_table_values = list(
                        np.array(np.where(Q_table[current_stateX, current_stateY, :] == max_value)[0]))

                    possibilities = []
                    for act_test in q_table_values:
                        if act_test not in actions_wrong:
                            possibilities.append(act_test)
                    possibilities = list(set(possibilities))
                    action = random.sample(possibilities, k=1)[0]

                    if action in enemies_nearby_agent:
                        print('Exploitation mod problem')
                        print('Enemies nearby: ', enemies_nearby_agent)
                        print('Enemies attached: ', enemies_attached_agent)
                        print(f'Possible actions: {possible_actions}, possibilities: {possibilities}')
                        print('Wrong actions ', actions_wrong)
                        print('Q-table situation: ', np.mean(Q_table[current_stateX, current_stateY, possible_actions]))
                        print('Action taken', action)

            result = env.step_agent(action)
            # print('result:', result)
            next_stateX = int(result[0][agent][0])
            next_stateY = int(result[0][agent][0])
            reward = int(result[1][agent])
            done = result[2][agent]  # If agent wins, end loop and restart
            if_lose = result[3]

            Q_table[current_stateX, current_stateY, action] = (1 - LEARNING_RATE) * Q_table[
                current_stateX, current_stateY, action] + LEARNING_RATE * (reward + GAMMA * max(
                Q_table[next_stateX, next_stateY, :]))
            total_episode_reward = total_episode_reward + reward

            if if_lose:
                current_state, _, _, _, _ = env.reset(reset_n_times_loser=False)
                current_stateX = current_state[agent][0]
                current_stateY = current_state[agent][1]

            if (abs(current_stateX - next_stateX) + abs(current_stateY - next_stateY)) > 1 and n_act_agents < 5:
                print('movement control problem:', [current_stateX, current_stateY], [next_stateX, next_stateY])

        average_episodes_rewards.append(total_episode_reward)
        steps_for_episode.append(step_for_episode)
        # updating the exploration proba using exponential decay formula
        exploration_proba = max(min_exploration_proba, np.exp(-exploration_decreasing_decay * e))

    print(f'Average reward: {np.mean(average_episodes_rewards)}')
    return average_episodes_rewards, steps_for_episode


def DQN():
    return 0
