import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# discounted factor
gamma = 0.99
# learning rate
lr = 0.01
# how many tries the system can do in the exploration
exploration_actions_threshold = 10


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
            else:
                action = np.argmax(Q_table[current_stateX, current_stateY, :])

            result = env.step_agent(action)
            # print('result:', result)
            next_stateX = int(result[0][agent][0])
            next_stateY = int(result[0][agent][0])
            reward = int(result[1][agent])
            done = result[2][agent]  # If agent wins, end loop and restart
            if_lose = result[3]

            Q_table[current_stateX, current_stateY, action] = (1 - lr) * Q_table[
                current_stateX, current_stateY, action] + lr * (reward + gamma * max(
                Q_table[next_stateX, next_stateY, :]))
            total_episode_reward = total_episode_reward + reward
            if (abs(current_stateX - next_stateX) + abs(current_stateY - next_stateY)) > 1 and n_act_agents < 5:
                print('movement control problem:', [current_stateX, current_stateY], [next_stateX, next_stateY])

        average_episodes_rewards.append(total_episode_reward)
        steps_for_episode.append(step_for_episode)
        # updating the exploration proba using exponential decay formula
        exploration_proba = max(min_exploration_proba, np.exp(-exploration_decreasing_decay * e))

    fig = plt.figure()
    plt.plot(average_episodes_rewards)
    plt.plot(step_for_episode)
    plt.show()
    return average_episodes_rewards, steps_for_episode

def CQL4():

    return 0

def DQN():

    return 0