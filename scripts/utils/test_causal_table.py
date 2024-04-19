import itertools
import pandas as pd
import global_variables
import numpy as np


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

    def do_check(self) -> bool:
        for_goals = [s for s in self.possible_values if s != 0]
        combinations = list(itertools.product(for_goals, itertools.product(self.possible_values, repeat=2))) # (goal, (enemies))

        count_wrongs = 0
        for comb in combinations:
            goal_nearby = comb[0]
            goal_nearby = [goal_nearby]
            enemies_nearby = list(comb[1])

            to_evaluate = self.function_to_test(self.table_to_test, enemies_nearby, goal_nearby)
            true = self._result_without_causal_logic(enemies_nearby, goal_nearby)

            if not to_evaluate == true:
                """print(f'\n Enemies nearby: {enemies_nearby} - Goal nearby: {goal_nearby}')
                print('My: ', to_evaluate)
                print('True: ', true)"""
                count_wrongs += 1

        if count_wrongs == 0:
            print('Causal table and function for its usage are corrects')
            return True
        else:
            print(f'Causal table and function for its usage are incorrect - {count_wrongs} errors')
            return False


if __name__ == '__main__':
    ct = pd.read_pickle('C:\\Users\giova\Documents\Research\CausalRL\out_causal_table_8x8.pkl')
    # ct = pd.read_pickle(f'{global_variables.PATH_CAUSAL_TABLE_OFFLINE}')

    test = TestCausalTable(ct, global_variables.get_possible_actions)

    test.do_check()
