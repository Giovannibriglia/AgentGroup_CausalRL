import os
import random
import re
import time
import warnings
from itertools import product
import networkx as nx
import numpy as np
import pandas as pd
from causalnex.inference import InferenceEngine
from causalnex.network import BayesianNetwork
from causalnex.structure.notears import from_pandas
from tqdm.auto import tqdm
from scripts.algorithms import exploration_strategies

warnings.filterwarnings("ignore")

col_action = 'Action_Agent0'
col_deltaX = 'DeltaX_Agent0'
col_deltaY = 'DeltaY_Agent0'
col_reward = 'Reward_Agent0'
col_nearby_enemy = 'Enemy0_Nearby_Agent0'
col_nearby_goal = 'Goal0_Nearby_Agent0'

causal_table_offline = pd.read_pickle('../launch_experiments/offline_heuristic_table.pkl')


def get_possible_actions(n_act_agents, enemies_nearby_all_agents, goals_nearby_all_agents, if_online):
    if if_online:
        try:
            causal_table = pd.read_pickle('online_heuristic_table.pkl')
        except:
            return list(np.arange(0, n_act_agents, 1))
    else:
        causal_table = causal_table_offline

    nearbies_goals = goals_nearby_all_agents[0]
    nearbies_enemies = enemies_nearby_all_agents[0]
    possible_actions = list(np.arange(0, n_act_agents, 1))

    check_goal = False
    possible_actions_for_goal = []
    for nearby_goal in nearbies_goals:
        action_to_do = causal_table[
            (causal_table[col_reward] == 1) & (causal_table[col_nearby_goal] == nearby_goal)].reset_index(drop=True)
        if not action_to_do.empty:
            action_to_do = action_to_do.loc[0, col_action]
            if action_to_do in possible_actions:
                possible_actions_for_goal.append(action_to_do)
                check_goal = True

    if not check_goal:
        for nearby_enemy in nearbies_enemies:
            action_to_remove = causal_table[
                (causal_table[col_reward] == -1) & (causal_table[col_nearby_enemy] == nearby_enemy)].reset_index(
                drop=True)
            if not action_to_remove.empty:
                action_to_remove = action_to_remove.loc[0, col_action]
                if action_to_remove in possible_actions:
                    possible_actions.remove(action_to_remove)
        # print(f'Enemies nearby: {nearbies_enemies} -- Possible actions: {possible_actions}')
    else:
        possible_actions = possible_actions_for_goal
        # print(f'Goals nearby: {nearbies_goals} -- Possible actions: {possible_actions}')

    return possible_actions


def create_df(env):
    n_agents = env.n_agents
    n_enemies = env.n_enemies
    n_goals = env.n_goals
    cols_df = []
    for agent in range(n_agents):
        cols_df.append(f'Action_Agent{agent}')
        cols_df.append(f'DeltaX_Agent{agent}')
        cols_df.append(f'DeltaY_Agent{agent}')
        if n_goals > 0 or n_enemies > 0:
            cols_df.append(f'Reward_Agent{agent}')
        for enemy in range(n_enemies):
            cols_df.append(f'Enemy{enemy}_Nearby_Agent{agent}')
        for goal in range(n_goals):
            cols_df.append(f'Goal{goal}_Nearby_Agent{agent}')

    df = pd.DataFrame(columns=cols_df)
    return df, cols_df


class Causality:

    def __init__(self):
        self.df = None
        self.features_names = None
        self.structureModel = None
        self.past_structureModel = None
        self.check_structureModel = 0
        self.bn = None
        self.ie = None
        self.independents_var = None
        self.dependents_var = None
        self.th_sm = 5

    def process_df(self, df_start):
        start_columns = df_start.columns.to_list()
        n_enemies_columns = [s for s in start_columns if 'Enemy' in s]
        if n_enemies_columns == 1:
            return df_start
        else:
            df_only_nearbies = df_start[n_enemies_columns]

            new_column = []
            for episode in range(len(df_start)):
                single_row = df_only_nearbies.loc[episode].tolist()

                if df_start.loc[episode, 'Reward_Agent0'] == -1:
                    enemy_nearbies_true = [s for s in single_row if s != 50]
                    action_agent = df_start.loc[episode, 'Action_Agent0']

                    if action_agent in enemy_nearbies_true:
                        new_column.append(action_agent)
                    else:
                        new_column.append(50)
                else:
                    new_column.append(random.choice(single_row))

            df_out = df_start.drop(columns=n_enemies_columns)

            df_out['Enemy0_Nearby_Agent0'] = new_column

            return df_out

    def training(self, e, df):
        new_df = self.process_df(df)

        if self.check_structureModel <= self.th_sm:
            time1 = time.time()
            self.df = pd.concat([self.df, new_df], axis=0, ignore_index=True)
            # print(f'\nstructuring model...')
            self.structureModel = from_pandas(self.df)
            self.structureModel.remove_edges_below_threshold(0.2)
            # print('\nConcatenation time: ', time.time() - time1, len(self.df), self.check_structureModel)

        self.features_names = self.df.columns.to_list()

        if nx.number_weakly_connected_components(self.structureModel) == 1 and nx.is_directed_acyclic_graph(
                self.structureModel) and len(self.df) > 100:
            # print(f'training bayesian network...')
            if self.check_structureModel <= self.th_sm:
                time2 = time.time()
                self.bn = BayesianNetwork(self.structureModel)
                self.bn.fit_node_states_and_cpds(self.df)
                # print('BN implementation time: ', time.time() - time2)
            else:
                self.bn.fit_node_states_and_cpds(new_df)

            bad_nodes = [node for node in self.bn.nodes if not re.match("^[0-9a-zA-Z_]+$", node)]
            if bad_nodes:
                print('Bad nodes: ', bad_nodes)
            time3 = time.time()
            self.ie = InferenceEngine(self.bn)

            self.dependents_var = []
            self.independents_var = []

            #  print('do-calculus-1...')
            # understand who influences whom
            before = self.ie.query()
            for var in self.features_names:
                count_var = 0
                for value in self.df[var].unique():
                    try:
                        self.ie.do_intervention(var, int(value))
                        after = self.ie.query()
                        features = [s for s in self.features_names if s not in var]
                        count_var_value = 0
                        for feat in features:
                            best_key_before, max_value_before = max(before[feat].items(), key=lambda x: x[1])
                            best_key_after, max_value_after = max(after[feat].items(), key=lambda x: x[1])

                            if best_key_after != best_key_before and round(max_value_after, 4) != round(
                                    1 / len(after[feat]), 4):
                                count_var_value += 1
                        self.ie.reset_do(var)

                        count_var += count_var_value
                    except:
                        pass

                if count_var > 0:
                    # print(f'{var} --> {count_var} changes, internally caused ')
                    self.dependents_var.append(var)
                else:
                    # print(f'{var} --> externally caused')
                    self.independents_var.append(var)

            # print(f'**Externally caused: {self.independents_var}')
            # print(f'**Externally influenced: {self.dependents_var}')
            # print('do-calculus-2...')
            causal_table = pd.DataFrame(columns=self.features_names)

            arrays = []
            for feat_ind in self.dependents_var:
                arrays.append(self.df[feat_ind].unique())
            var_combinations = list(product(*arrays))

            for n, comb_n in enumerate(var_combinations):
                # print('\n')
                for var_ind in range(len(self.dependents_var)):
                    try:
                        self.ie.do_intervention(self.dependents_var[var_ind], int(comb_n[var_ind]))
                        # print(f'{self.dependents_var[var_ind]} = {int(comb_n[var_ind])}')
                        causal_table.at[n, f'{self.dependents_var[var_ind]}'] = int(comb_n[var_ind])
                    except:
                        # print(f'no {self.dependents_var[var_ind]} = {int(comb_n[var_ind])}')
                        causal_table.at[n, f'{self.dependents_var[var_ind]}'] = pd.NA

                after = self.ie.query()
                for var_dep in self.independents_var:
                    # print(f'{var_dep}) {after[var_dep]}')
                    max_key, max_value = max(after[var_dep].items(), key=lambda x: x[1])
                    if round(max_value, 4) != round(1 / len(after[var_dep]), 4):
                        causal_table.at[n, f'{var_dep}'] = int(max_key)
                        # print(f'{var_dep}) -> {max_key}: {max_value}')
                    else:
                        causal_table.at[n, f'{var_dep}'] = pd.NA
                        # print(f'{var_dep}) -> unknown')

                for var_ind in range(len(self.dependents_var)):
                    self.ie.reset_do(self.dependents_var[var_ind])

            causal_table.dropna(axis=0, how='any', inplace=True)
            causal_table.to_pickle('online_heuristic_table.pkl')

            # print('do-calculus time: ', time.time() - time3, '\n')

            nodes1 = set(self.structureModel.nodes)
            edges1 = set(self.structureModel.edges)
            if self.past_structureModel is not None:
                edges2 = set(self.past_structureModel.edges)
                nodes2 = set(self.past_structureModel.nodes)
            else:
                edges2 = None
                nodes2 = None

            if (nodes1 == nodes2) and (edges1 == edges2):
                self.check_structureModel += 1
            else:
                self.past_structureModel = self.structureModel
                # self.check_structureModel = 0


def DQNs(env, dict_env_parameters, dict_learning_parameters):

    rows = env.rows
    cols = env.cols

    n_act_agents = dict_env_parameters['n_act_agents']
    n_episodes = dict_env_parameters['n_episodes']
    alg = dict_env_parameters['alg']
    who_moves_first = dict_env_parameters['who_moves_first']
    episodes_to_visualize = dict_env_parameters['episodes_to_visualize']
    seed_value = dict_env_parameters['seed_value']

    TIMEOUT_IN_HOURS = dict_learning_parameters['TIMEOUT_IN_HOURS']

    np.random.seed(seed_value)
    random.seed(seed_value)
    action_space_size = n_act_agents

    dict_env_for_expl_strategy = {'rows': rows, 'cols': cols, 'action_space_size': action_space_size, 'n_episodes': n_episodes, 'alg': alg}

    if 'SA' in alg:
        agent = exploration_strategies.SoftmaxAnnealingQAgent(dict_env_for_expl_strategy, dict_learning_parameters)
    elif 'TS' in alg:
        agent = exploration_strategies.ThompsonSamplingQAgent(dict_env_for_expl_strategy, dict_learning_parameters)
    elif 'BM' in alg:
        agent = exploration_strategies.BoltzmannQAgent(dict_env_for_expl_strategy, dict_learning_parameters)
    else:  # 'EG'
        agent = exploration_strategies.EpsilonGreedyQAgent(dict_env_for_expl_strategy, dict_learning_parameters)

    average_episodes_rewards = []
    steps_for_episode = []

    first_visit = True

    pbar = tqdm(range(n_episodes))
    for e in pbar:
        agent_n = 0
        if e == 0:
            if first_visit:
                current_state, _, _, _, _ = env.reset(reset_n_times_loser=True)
                initial_time = time.time()
                first_visit = False
            else:
                current_state, _, _, _, _ = env.reset(reset_n_times_loser=False)
        else:
            current_state, _, _, _, _ = env.reset(reset_n_times_loser=False)

        current_state = current_state[agent_n]

        total_episode_reward = 0
        step_for_episode = 0
        done = False

        if e in episodes_to_visualize:
            if_visualization = True
            env.init_gui(alg, e)
        else:
            if_visualization = False

        while not done:

            if (time.time() - initial_time) > TIMEOUT_IN_HOURS * 3600:
                q_table = agent.policy_net

                return average_episodes_rewards, steps_for_episode, q_table

            possible_actions = None
            if who_moves_first == 'Enemy':
                env.step_enemies()
                if if_visualization:
                    env.movement_gui(n_episodes, step_for_episode)
                """_, _, if_lose = env.check_winner_gameover_agent(current_state[0], current_state[1])
                if not if_lose:"""
                if 'causal' in alg:
                    enemies_nearby_all_agents, goals_nearby_all_agents = env.get_nearbies_agent()
                    possible_actions = get_possible_actions(n_act_agents, enemies_nearby_all_agents,
                                                            goals_nearby_all_agents, if_online=False)

                general_state = np.zeros(rows * cols)
                current_stateX = current_state[0]
                current_stateY = current_state[1]
                general_state[current_stateX * rows + current_stateY] = 1

                action = agent.choose_action(general_state, possible_actions)
                next_state = env.step_agent(action)[agent_n]

                if if_visualization:
                    env.movement_gui(n_episodes, step_for_episode)
                """else:
                    next_state = current_state"""
                new_stateX_ag = next_state[0]
                new_stateY_ag = next_state[1]
            else:  # who_moves_first == 'Agent':
                if 'causal' in alg:
                    enemies_nearby_all_agents, goals_nearby_all_agents = env.get_nearbies_agent()
                    possible_actions = get_possible_actions(n_act_agents, enemies_nearby_all_agents,
                                                            goals_nearby_all_agents, if_online=False)

                general_state = np.zeros(rows * cols)
                current_stateX = current_state[0]
                current_stateY = current_state[1]
                general_state[current_stateX * rows + current_stateY] = 1

                action = agent.choose_action(general_state, possible_actions)
                next_state = env.step_agent(action)[0]

                if if_visualization:
                    env.movement_gui(n_episodes, step_for_episode)
                new_stateX_ag = next_state[0]
                new_stateY_ag = next_state[1]
                _, dones, _ = env.check_winner_gameover_agent(new_stateX_ag, new_stateY_ag)
                if not dones[agent_n]:
                    env.step_enemies()
                    if if_visualization:
                        env.movement_gui(n_episodes, step_for_episode)

            rewards, dones, if_lose = env.check_winner_gameover_agent(new_stateX_ag, new_stateY_ag)
            reward = int(rewards[agent_n])
            done = dones[agent_n]  # If agent wins, end loop and restart

            if 'causal' in alg and if_lose and not ([current_state] == [next_state] and action != 0) and len(
                    possible_actions) > 0:
                print(f'\nLose: wrong causal gameover model in {alg}')
                print(f'New agents pos: {env.pos_agents[-1]}')
                print(f'Enemies pos: {env.pos_enemies[-1]} - enemies nearby: {enemies_nearby_all_agents}')
                print(f'Possible actions: {possible_actions} - chosen action: {action}')

            next_general_state = np.zeros(env.rows * env.cols)
            next_general_state[new_stateX_ag * rows + new_stateY_ag] = 1

            agent.update_Q_or_memory(general_state, action, reward, next_general_state)
            agent.optimize_model()
            agent.update_target_net()

            total_episode_reward += reward

            if if_lose:
                current_state, _, _, _, _ = env.reset(reset_n_times_loser=False)
                current_state = current_state[agent_n]
            else:
                current_state = next_state
            step_for_episode += 1

        if if_visualization:
            env.save_video()

        if 'SA' in alg or 'EG' in alg:
            agent.update_exp_fact(e)

        if total_episode_reward > 1:
            print('**** marioooo ***')
        average_episodes_rewards.append(total_episode_reward)
        steps_for_episode.append(step_for_episode)
        pbar.set_postfix_str(
            f"Average reward: {round(np.mean(average_episodes_rewards), 3)}, Number of defeats: {env.n_times_loser}")

    print(f'Average reward: {round(np.mean(average_episodes_rewards), 3)}, Number of defeats: {env.n_times_loser}')

    q_table = agent.policy_net

    return average_episodes_rewards, steps_for_episode, q_table


def QL_causality_offline(env, dict_env_parameters, dict_learning_parameters,
                         predefined_q_table=None):
    rows = env.rows
    cols = env.cols

    n_act_agents = dict_env_parameters['n_act_agents']
    n_episodes = dict_env_parameters['n_episodes']
    alg = dict_env_parameters['alg']
    who_moves_first = dict_env_parameters['who_moves_first']
    episodes_to_visualize = dict_env_parameters['episodes_to_visualize']
    seed_value = dict_env_parameters['seed_value']

    TIMEOUT_IN_HOURS = dict_learning_parameters['TIMEOUT_IN_HOURS']

    np.random.seed(seed_value)
    random.seed(seed_value)
    action_space_size = n_act_agents

    dict_env_for_expl_strategy = {'rows': rows, 'cols': cols, 'action_space_size': action_space_size,
                                  'n_episodes': n_episodes, 'alg': alg}

    if 'SA' in alg:
        agent = exploration_strategies.SoftmaxAnnealingQAgent(dict_env_for_expl_strategy, dict_learning_parameters, predefined_q_table)
    elif 'TS' in alg:
        agent = exploration_strategies.ThompsonSamplingQAgent(dict_env_for_expl_strategy, dict_learning_parameters, predefined_q_table)
    elif 'BM' in alg:
        agent = exploration_strategies.BoltzmannQAgent(dict_env_for_expl_strategy, dict_learning_parameters, predefined_q_table)
    else:  # 'EG'
        agent = exploration_strategies.EpsilonGreedyQAgent(dict_env_for_expl_strategy, dict_learning_parameters, predefined_q_table)

    average_episodes_rewards = []
    steps_for_episode = []

    first_visit = True

    pbar = tqdm(range(n_episodes))
    for e in pbar:
        agent_n = 0
        if e == 0:
            if first_visit:
                current_state, _, _, _, _ = env.reset(reset_n_times_loser=True)
                initial_time = time.time()
                first_visit = False
            else:
                current_state, _, _, _, _ = env.reset(reset_n_times_loser=False)
        else:
            current_state, _, _, _, _ = env.reset(reset_n_times_loser=False)
        current_state = current_state[agent_n]

        total_episode_reward = 0
        step_for_episode = 0
        done = False

        if e in episodes_to_visualize:
            if_visualization = True
            env.init_gui(alg, e)
        else:
            if_visualization = False

        while not done:

            if (time.time() - initial_time) > TIMEOUT_IN_HOURS * 3600:
                if 'TS' in alg:
                    q_table = [agent.alpha, agent.beta]
                else:
                    q_table = agent.q_table

                return average_episodes_rewards, steps_for_episode, q_table

            possible_actions = None
            if who_moves_first == 'Enemy':
                env.step_enemies()
                if if_visualization:
                    env.movement_gui(n_episodes, step_for_episode)
                """_, _, if_lose = env.check_winner_gameover_agent(current_state[0], current_state[1])
                if not if_lose:"""
                if 'causal' in alg:
                    enemies_nearby_all_agents, goals_nearby_all_agents = env.get_nearbies_agent()
                    possible_actions = get_possible_actions(n_act_agents, enemies_nearby_all_agents,
                                                            goals_nearby_all_agents, if_online=False)
                action = agent.choose_action(current_state, possible_actions)
                next_state = env.step_agent(action)[0]
                if if_visualization:
                    env.movement_gui(n_episodes, step_for_episode)
                """else:
                    next_state = current_state"""
                new_stateX_ag = next_state[0]
                new_stateY_ag = next_state[1]
            else:  # who_moves_first == 'Agent':
                if 'causal' in alg:
                    enemies_nearby_all_agents, goals_nearby_all_agents = env.get_nearbies_agent()
                    possible_actions = get_possible_actions(n_act_agents, enemies_nearby_all_agents,
                                                            goals_nearby_all_agents, if_online=False)

                action = agent.choose_action(current_state, possible_actions)
                next_state = env.step_agent(action)[0]
                if if_visualization:
                    env.movement_gui(n_episodes, step_for_episode)
                new_stateX_ag = next_state[0]
                new_stateY_ag = next_state[1]
                _, dones, _ = env.check_winner_gameover_agent(new_stateX_ag, new_stateY_ag)
                if not dones[agent_n]:
                    env.step_enemies()
                    if if_visualization:
                        env.movement_gui(n_episodes, step_for_episode)

            rewards, dones, if_lose = env.check_winner_gameover_agent(new_stateX_ag, new_stateY_ag)
            reward = int(rewards[agent_n])
            done = dones[agent_n]  # If agent wins, end loop and restart

            if 'causal' in alg and if_lose and not ([current_state] == [next_state] and action != 0) and len(
                    possible_actions) > 0:
                print(f'\nLose: wrong causal gameover model in {alg}')
                print(f'New agents pos: {env.pos_agents[-1]}')
                print(f'Enemies pos: {env.pos_enemies[-1]} - enemies nearby: {enemies_nearby_all_agents}')
                print(f'Possible actions: {possible_actions} - chosen action: {action}')

            # Update the Q-table/values
            agent.update_Q_or_memory(current_state, action, reward, next_state)

            total_episode_reward += reward

            if if_lose:
                current_state, _, _, _, _ = env.reset(reset_n_times_loser=False)
                current_state = current_state[agent_n]
            else:
                current_state = next_state
            step_for_episode += 1

        if if_visualization:
            env.save_video()

        if 'SA' in alg or 'EG' in alg:
            agent.update_exp_fact(e)

        average_episodes_rewards.append(total_episode_reward)
        steps_for_episode.append(step_for_episode)
        pbar.set_postfix_str(
            f"Average reward: {round(np.mean(average_episodes_rewards), 3)}, Number of defeats: {env.n_times_loser}")

    print(f'Average reward: {round(np.mean(average_episodes_rewards), 3)}, Number of defeats: {env.n_times_loser}')

    if 'TS' in alg:
        q_table = [agent.alpha, agent.beta]
    else:
        q_table = agent.q_table

    return average_episodes_rewards, steps_for_episode, q_table


def QL_causality_online(env, dict_env_parameters, dict_learning_parameters, BATCH_EPISODES_UPDATE_BN=500,
                         predefined_q_table=None):

    try:
        os.remove('online_heuristic_table.pkl')
    except:
        pass

    rows = env.rows
    cols = env.cols

    n_act_agents = dict_env_parameters['n_act_agents']
    n_episodes = dict_env_parameters['n_episodes']
    alg = dict_env_parameters['alg']
    who_moves_first = dict_env_parameters['who_moves_first']
    episodes_to_visualize = dict_env_parameters['episodes_to_visualize']
    seed_value = dict_env_parameters['seed_value']

    TIMEOUT_IN_HOURS = dict_learning_parameters['TIMEOUT_IN_HOURS']
    EXPLORATION_GAME_PERCENT = dict_learning_parameters['EXPLORATION_GAME_PERCENT']
    TH_CONSECUTIVE_CHECKS_CAUSAL_TABLE = dict_learning_parameters['TH_CONSECUTIVE_CHECKS_CAUSAL_TABLE']

    np.random.seed(seed_value)
    random.seed(seed_value)
    action_space_size = n_act_agents

    dict_env_for_expl_strategy = {'rows': rows, 'cols': cols, 'action_space_size': action_space_size,
                                  'n_episodes': n_episodes, 'alg': alg}

    if 'SA' in alg:
        agent = exploration_strategies.SoftmaxAnnealingQAgent(dict_env_for_expl_strategy, dict_learning_parameters,
                                                              predefined_q_table)
    elif 'TS' in alg:
        agent = exploration_strategies.ThompsonSamplingQAgent(dict_env_for_expl_strategy, dict_learning_parameters,
                                                              predefined_q_table)
    elif 'BM' in alg:
        agent = exploration_strategies.BoltzmannQAgent(dict_env_for_expl_strategy, dict_learning_parameters,
                                                       predefined_q_table)
    else:  # 'EG'
        agent = exploration_strategies.EpsilonGreedyQAgent(dict_env_for_expl_strategy, dict_learning_parameters,
                                                           predefined_q_table)

    average_episodes_rewards = []
    steps_for_episode = []

    causality = Causality()
    first_visit = True
    check_causal_table = 0

    pbar = tqdm(range(n_episodes))
    for e in pbar:
        agent_n = 0
        if e == 0:
            if first_visit:
                current_state, _, _, _, _ = env.reset(reset_n_times_loser=True)
                df_for_causality, columns_df_causality = create_df(env)
                counter_e = 0
                initial_time = time.time()
                first_visit = False
            else:
                current_state, _, _, _, _ = env.reset(reset_n_times_loser=False)
        else:
            current_state, _, _, _, _ = env.reset(reset_n_times_loser=False)
        current_state = current_state[agent_n]

        if e in episodes_to_visualize:
            if_visualization = True
            env.init_gui(alg, e)
        else:
            if_visualization = False

        total_episode_reward = 0
        step_for_episode = 0
        done = False

        while not done:
            if (time.time() - initial_time) > TIMEOUT_IN_HOURS * 3600:
                if 'TS' in alg:
                    q_table = [agent.alpha, agent.beta]
                else:
                    q_table = agent.q_table

                try:
                    os.remove('online_heuristic_table.pkl')
                except:
                    pass

                return average_episodes_rewards, steps_for_episode, q_table

            if who_moves_first == 'Enemy':
                env.step_enemies()
                if if_visualization:
                    env.movement_gui(n_episodes, step_for_episode)
                enemies_nearby_all_agents, goals_nearby_all_agents = env.get_nearbies_agent()
                possible_actions = get_possible_actions(n_act_agents, enemies_nearby_all_agents,
                                                        goals_nearby_all_agents, if_online=True)

                action = agent.choose_action(current_state, possible_actions)
                next_state = env.step_agent(action)[0]
                if if_visualization:
                    env.movement_gui(n_episodes, step_for_episode)

                new_stateX_ag = next_state[0]
                new_stateY_ag = next_state[1]

                for enemy in range(env.n_enemies):
                    df_for_causality.at[counter_e, f'Enemy{enemy}_Nearby_Agent{agent_n}'] = \
                        enemies_nearby_all_agents[agent_n][enemy]
                for goal in range(env.n_goals):
                    df_for_causality.at[counter_e, f'Goal{goal}_Nearby_Agent{agent_n}'] = \
                        goals_nearby_all_agents[agent_n][goal]

                df_for_causality.at[counter_e, f'Action_Agent{agent_n}'] = action
                df_for_causality.at[counter_e, f'DeltaX_Agent{agent_n}'] = int(new_stateX_ag - current_state[0])
                df_for_causality.at[counter_e, f'DeltaY_Agent{agent_n}'] = int(new_stateY_ag - current_state[1])

            else:  # who_moves_first == 'Agent':
                enemies_nearby_all_agents, goals_nearby_all_agents = env.get_nearbies_agent()
                possible_actions = get_possible_actions(n_act_agents, enemies_nearby_all_agents,
                                                        goals_nearby_all_agents, if_online=True)

                action = agent.choose_action(current_state, possible_actions)
                next_state = env.step_agent(action)[0]
                df_for_causality.at[counter_e, f'Action_Agent{agent_n}'] = action
                df_for_causality.at[counter_e, f'DeltaX_Agent{agent_n}'] = int(new_stateX_ag - current_state[0])
                df_for_causality.at[counter_e, f'DeltaY_Agent{agent_n}'] = int(new_stateY_ag - current_state[1])
                if if_visualization:
                    env.movement_gui(n_episodes, step_for_episode)
                new_stateX_ag = next_state[0]
                new_stateY_ag = next_state[1]
                for goal in range(env.n_goals):
                    df_for_causality.at[counter_e, f'Goal{goal}_Nearby_Agent{agent_n}'] = \
                        goals_nearby_all_agents[agent_n][goal]
                _, dones, _ = env.check_winner_gameover_agent(new_stateX_ag, new_stateY_ag)
                if not dones[agent_n]:
                    env.step_enemies()
                    for enemy in range(env.n_enemies):
                        df_for_causality.at[counter_e, f'Enemy{enemy}_Nearby_Agent{agent_n}'] = \
                            enemies_nearby_all_agents[agent_n][enemy]
                    if if_visualization:
                        env.movement_gui(n_episodes, step_for_episode)

            rewards, dones, if_lose = env.check_winner_gameover_agent(new_stateX_ag, new_stateY_ag)
            reward = int(rewards[agent_n])
            done = dones[agent_n]  # If agent wins, end loop and restart

            df_for_causality.at[counter_e, f'Reward_Agent{agent_n}'] = reward
            counter_e += 1

            if possible_actions is not None and if_lose and [current_state] != [next_state]:
                # print(f'\nLose: causal model not ready yet')
                pass

            # Update the Q-table/values
            agent.update_Q_or_memory(current_state, action, reward, next_state)

            total_episode_reward += reward

            if if_lose:
                current_state, _, _, _, _ = env.reset(reset_n_times_loser=False)
                current_state = current_state[agent_n]
            else:
                current_state = next_state
            step_for_episode += 1

        average_episodes_rewards.append(total_episode_reward)
        steps_for_episode.append(step_for_episode)

        if if_visualization:
            env.save_video()

        if 'SA' in alg or 'EG' in alg:
            agent.update_exp_fact(e)

        if e % BATCH_EPISODES_UPDATE_BN == 0 and e < int(
                 EXPLORATION_GAME_PERCENT * n_episodes) and check_causal_table < TH_CONSECUTIVE_CHECKS_CAUSAL_TABLE:
            pbar.set_postfix_str(
                f"Average reward: {round(np.mean(average_episodes_rewards), 3)}, Number of defeats: {env.n_times_loser}, do-calculus...")

            try:
                os.rename('online_heuristic_table.pkl', 'past_online_heuristic_table.pkl')
            except:
                pass

            for col in df_for_causality.columns:
                if col not in columns_df_causality:
                    df_for_causality.drop([col], axis=1)
                    print('This column was not in the initial columns: ', col)
                else:
                    df_for_causality[str(col)] = df_for_causality[str(col)].astype(str).str.replace(',', '').astype(
                        float)

            causality.training(e, df_for_causality)
            df_for_causality, columns_df_causality = create_df(env)
            counter_e = 0

            try:
                if len(pd.read_pickle('past_online_heuristic_table.pkl')) == len(
                        pd.read_pickle('online_heuristic_table.pkl')):
                    check_causal_table += 1
                    os.remove('past_online_heuristic_table.pkl')
                else:
                    check_causal_table = 0
            except:
                pass

        pbar.set_postfix_str(
            f"Average reward: {round(np.mean(average_episodes_rewards), 3)}, Number of defeats: {env.n_times_loser}, algorithm...")

    print(f'Average reward: {round(np.mean(average_episodes_rewards), 3)}, Number of defeats: {env.n_times_loser}')

    try:
        os.remove('online_heuristic_table.pkl')
    except:
        pass

    if 'TS' in alg:
        q_table = [agent.alpha, agent.beta]
    else:
        q_table = agent.q_table

    return average_episodes_rewards, steps_for_episode, q_table
