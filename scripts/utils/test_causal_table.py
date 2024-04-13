import itertools
import pandas as pd
import global_variables
import numpy as np


def _get_possible_actions(causal_table: pd.DataFrame,
                          enemies_nearby: np.ndarray = None, goals_nearby: np.ndarray = None) -> list:
    if enemies_nearby is not None:
        enemies_nearby = list(set(enemies_nearby))
    if goals_nearby is not None:
        goals_nearby = list(set(goals_nearby))

    col_action = next(s for s in causal_table.columns if global_variables.LABEL_COL_ACTION in s)
    col_reward = next(s for s in causal_table.columns if global_variables.LABEL_COL_REWARD in s)
    col_enemy_nearby = next(s for s in causal_table.columns if global_variables.LABEL_ENEMY_CAUSAL_TABLE in s and
                            global_variables.LABEL_NEARBY_CAUSAL_TABLE in s)
    col_goal_nearby = next(s for s in causal_table.columns if
                           global_variables.LABEL_GOAL_CAUSAL_TABLE in s and global_variables.LABEL_NEARBY_CAUSAL_TABLE in s)

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
    # Group by action
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


class TestCausalTable:
    def __init__(self, table_to_test: pd.DataFrame, function_to_test):
        self.n_actions = 5
        self.entity_far = 50

        self.actions = np.arange(0, self.n_actions, 1).tolist()
        self.possible_values = self.actions + [self.entity_far]

        self.table_to_test = table_to_test
        self.function_to_test = function_to_test

    def _result_without_causal_logic(self, enemies_nearby: list = None, goals_nearby: list = None) -> list:

        possible_actions = self.actions.copy()

        if goals_nearby is not None and goals_nearby[0] != 50:
            possible_actions = goals_nearby
        elif enemies_nearby is not None:
            enemies = np.unique(enemies_nearby)
            possible_actions = [s for s in possible_actions if s not in enemies]

        # possible_actions.sort(reverse=True)

        return possible_actions

    def check(self):
        for_goals = [s for s in self.possible_values if s != 0]
        combinations = list(itertools.product(for_goals, itertools.product(self.possible_values, repeat=2))) # (goal, (enemies))

        for comb in combinations:
            goal_nearby = comb[0]
            goal_nearby = [goal_nearby]
            enemies_nearby = list(comb[1])

            to_evaluate = self.function_to_test(self.table_to_test, enemies_nearby, goal_nearby)
            true = self._result_without_causal_logic(enemies_nearby, goal_nearby)

            if not to_evaluate == true:
                print(f'\n Enemies nearby: {enemies_nearby} - Goal nearby: {goal_nearby}')
                print('My: ', to_evaluate)
                print('True: ', true)


if __name__ == '__main__':
    ct = pd.read_pickle(f'{global_variables.PATH_CAUSAL_TABLE_OFFLINE}')

    test = TestCausalTable(ct, _get_possible_actions)

    test.check()
