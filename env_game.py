import random
import warnings
import pygame
from gym.spaces import Discrete

warnings.filterwarnings("ignore")

class CustomEnv:

    def __init__(self, rows, cols, n_agents, n_act_agents, n_enemies, n_act_enemies, n_goals, if_maze,
                 if_same_enemies_actions):
        self.rows = rows
        self.cols = cols
        self.n_agents = n_agents
        self.n_act_agents = n_act_agents
        self.n_enemies = n_enemies
        self.n_act_enemies = n_act_enemies
        self.n_goals = n_goals
        self.n_walls = rows * 2

        # reward definitionm
        self.reward_alive = 0
        self.reward_winner = 1
        self.reward_loser = -1

        self.n_times_loser = 0

        #  game episode
        self.n_steps = 0
        self.len_actions_enemies = 1000

        self.status_col_name = 'Status'

        # grid for visualize agents and enemies positions
        self.grid_for_game = []
        # list for saving enemy' positions
        self.pos_enemies = []
        self.pos_enemies_for_reset = []
        # list for saving agents' positions
        self.pos_agents = []
        self.pos_agents_for_reset = []
        # goal's position
        self.pos_goals = []

        # action space of agents
        self.action_space = Discrete(self.n_act_agents, start=0)

        # defining empyt matrices for game
        for ind_row in range(self.rows):
            row = []
            for ind_col in range(self.cols):
                row.append('-')
            self.grid_for_game.append(row)

        self.observation_space = rows * cols

        # positioning enemies
        row_pos_enemies = []
        for enemy in range(1, self.n_enemies + 1, 1):
            # check if same position
            do = True
            while (do):
                x_nem = random.randint(0, self.rows - 1)
                y_nem = random.randint(0, self.cols - 1)
                if ([x_nem, y_nem] not in row_pos_enemies):
                    do = False
            self.grid_for_game[x_nem][y_nem] = 'En' + str(enemy)
            row_pos_enemies.append([x_nem, y_nem])
            self.pos_enemies_for_reset.append([x_nem, y_nem])
            # self.pos_enemies.append([x_nem, y_nem])
        self.pos_enemies.append(row_pos_enemies)

        # positioning agents
        row_pos_agents = []
        for agent in range(1, self.n_agents + 1, 1):
            # check if same position than enemies
            do = True
            while (do):
                x_agent = random.randint(0, self.rows - 1)
                y_agent = random.randint(0, self.cols - 1)
                if ([x_agent, y_agent] not in self.pos_enemies[0]):
                    do = False
            self.grid_for_game[x_agent][y_agent] = 'Agent' + str(agent)
            row_pos_agents.append([x_agent, y_agent])
            self.pos_agents_for_reset.append([x_agent, y_agent])
        self.pos_agents.append(row_pos_agents)

        self.reset_enemies_attached = [[False] * self.n_enemies] * self.n_agents
        self.reset_enemies_nearby = []
        for agent in range(0, self.n_agents, 1):
            single_agent = []
            for enemy in range(0, self.n_enemies, 1):
                x_ag = self.pos_agents_for_reset[agent][0]
                y_ag = self.pos_agents_for_reset[agent][1]
                x_en = self.pos_enemies_for_reset[enemy][0]
                y_en = self.pos_enemies_for_reset[enemy][1]
                single_agent.append(self.get_direction(x_ag, y_ag, x_en, y_en))
            self.reset_enemies_nearby.append(single_agent)

        self.if_same_enemies_actions = if_same_enemies_actions
        if self.if_same_enemies_actions:
            self.list_enemies_actions = []
            for enemy in range(0, self.n_enemies, 1):
                enemy_actions = []
                for act in range(self.len_actions_enemies):
                    enemy_actions.append(random.randint(0, self.n_act_enemies - 1))
                self.list_enemies_actions.append(enemy_actions)

        if if_maze:
            self.walls = []
            for wall in range(0, self.n_walls, 1):
                # check if same position than enemies and agents
                do = True
                while (do):
                    x_wall = random.randint(0, self.rows - 1)
                    y_wall = random.randint(0, self.cols - 1)
                    if ([x_wall, y_wall] not in self.pos_enemies[0] and [x_wall, y_wall] not in self.pos_agents[0] and [
                        x_wall, y_wall]):
                        do = False
                self.grid_for_game[x_wall][y_wall] = 'W'
                self.walls.append([x_wall, y_wall])

        # positioning goal
        for goal in range(1, self.n_goals + 1, 1):
            # check if same position than enemies and agents
            do = True
            while (do):
                x_goal = random.randint(0, self.rows - 1)
                y_goal = random.randint(0, self.cols - 1)
                if ([x_goal, y_goal] not in self.pos_enemies[0] and [x_goal, y_goal] not in self.pos_agents[0]):
                    do = False
            self.grid_for_game[x_goal][y_goal] = 'Goal' + str(goal)
            self.pos_goals.append([x_goal, y_goal])

        for goal_x, goal_y in self.pos_goals:
            check_goals = 0
            vetX = [-1, 0, 1]
            vetY = [-1, 0, 1]

            if goal_x == 0:
                check_goals += 1
                vetX.remove(-1)
            if goal_y == 0:
                check_goals += 1
                vetY.remove(-1)

            if goal_x == self.cols:
                check_goals += 1
                vetX.remove(1)
            if goal_y == self.rows:
                check_goals += 1
                vetY.remove(1)

            for addX in vetX:
                for addY in vetY:
                    if 0 < goal_x + addX < self.cols and 0 < goal_y + addY < self.rows:
                        if self.grid_for_game[goal_x + addX][goal_y + addY] == 'W':
                            check_goals += 1

            if check_goals >= self.n_act_agents - 1:
                for addX in vetX:
                    for addY in vetY:
                        if 0 < goal_x + addX < self.cols and 0 < goal_y + addY < self.rows:
                            if self.grid_for_game[goal_x + addX][goal_y + addY] == 'W':
                                self.grid_for_game[goal_x + addX][goal_y + addY] = '-'
                                break

        for ag_x, ag_y in self.pos_agents_for_reset:
            check_agents = 0
            vetX = [-1, 0, 1]
            vetY = [-1, 0, 1]

            if ag_x == 0:
                check_agents += 1
                vetX.remove(-1)
            if ag_y == 0:
                check_agents += 1
                vetY.remove(-1)

            if ag_x == self.cols:
                check_agents += 1
                vetX.remove(1)
            if ag_y == self.rows:
                check_agents += 1
                vetY.remove(1)

            for addX in vetX:
                for addY in vetY:
                    if 0 < ag_x + addX < self.cols and 0 < ag_y + addY < self.rows:
                        if self.grid_for_game[ag_x + addX][ag_y + addY] == 'W':
                            check_agents += 1

            if check_agents >= self.n_act_agents - 1:
                for addX in vetX:
                    for addY in vetY:
                        if 0 < ag_x + addX < self.cols and 0 < ag_y + addY < self.rows:
                            if self.grid_for_game[ag_x + addX][ag_y + addY] == 'W':
                                self.grid_for_game[ag_x + addX][ag_y + addY] = '-'
                                break

        for ind in range(len(self.grid_for_game)):
            print(self.grid_for_game[ind])

    def step_enemies(self):

        self.n_steps = self.n_steps + 1
        rewards, dones, enemies_nearby, goals_nearby = [], [], [], []
        " Enemies "
        " Setting enemies positions"
        new_enemies_pos = []
        for enemy in range(1, self.n_enemies + 1, 1):
            last_stateX_en = self.pos_enemies[-1][enemy - 1][0]
            last_stateY_en = self.pos_enemies[-1][enemy - 1][1]

            if self.if_same_enemies_actions:
                if self.n_steps < self.len_actions_enemies:
                    n = int(self.n_steps)
                else:
                    n = int(self.n_steps - self.len_actions_enemies * (int(self.n_steps / self.len_actions_enemies)))
                # n = (self.len_actions_enemies / self.n_steps)
                action = self.list_enemies_actions[enemy - 1][n]
            else:
                action = random.randint(0, self.n_act_enemies - 1)

            new_stateX_en, new_stateY_en, _, _, _ = self.get_action(action, last_stateX_en, last_stateY_en,
                                                               self.grid_for_game)
            # print('enemy pos: ', [new_stateX_en, new_stateY_en])

            if (abs(new_stateX_en - last_stateX_en) + abs(
                    new_stateY_en - last_stateY_en)) > 1 and self.n_act_enemies < 5:
                print('Enemy wrong movement', [last_stateX_en, last_stateY_en], '-', [new_stateX_en, new_stateY_en])

            new_enemies_pos.append([new_stateX_en, new_stateY_en])

        self.pos_enemies.append(new_enemies_pos)

        for agent in range(1, self.n_agents + 1, 1):
            x_ag = self.pos_agents[-1][agent - 1][0]
            y_ag = self.pos_agents[-1][agent - 1][1]
            single_agent_enemies = []
            for enemy in range(1, self.n_enemies + 1, 1):
                x_en = self.pos_enemies[-1][enemy - 1][0]
                y_en = self.pos_enemies[-1][enemy - 1][1]
                direction_nearby_enemy = self.get_direction(x_ag, y_ag, x_en, y_en)
                single_agent_enemies.append(direction_nearby_enemy)

            enemies_nearby.append(single_agent_enemies)

            single_agent_goals = []
            for goal in range(1, self.n_goals+1, 1):
                x_goal = self.pos_goals[goal-1][0]
                y_goal = self.pos_goals[goal-1][1]
                direction_nearby_goal = self.get_direction(x_ag, y_ag, x_goal, y_goal)
                single_agent_goals.append(direction_nearby_goal)

            goals_nearby.append(single_agent_goals)

        return enemies_nearby, goals_nearby, new_enemies_pos

    def step_agent(self, agents_actions):
        " Agents "
        rewards = []
        dones = []
        new_agents_pos = []
        for agent in range(1, self.n_agents + 1, 1):
            # print(self.pos_agents[self.n_steps-1])
            last_stateX_ag = self.pos_agents[-1][agent - 1][0]
            last_stateY_ag = self.pos_agents[-1][agent - 1][1]

            action = agents_actions

            new_stateX_ag, new_stateY_ag, res_action, _, _ = self.get_action(action, last_stateX_ag, last_stateY_ag,
                                                                             self.grid_for_game)
            # print('ag inside', [last_stateX_ag, last_stateY_ag], [new_stateX_ag, new_stateY_ag])

            if (abs(new_stateX_ag - last_stateX_ag) + abs(
                    new_stateY_ag - last_stateY_ag)) > 1 and self.n_act_agents < 5:
                print('Agent wrong movement', [last_stateX_ag, last_stateY_ag], '-', [new_stateX_ag, new_stateY_ag],
                      action)

            # check if agent wins
            if_win = False
            if_lose = False
            for goal in self.pos_goals:
                goal_x = goal[0]
                goal_y = goal[1]
                if new_stateX_ag == goal_x and new_stateY_ag == goal_y:
                    reward = self.reward_winner
                    done = True
                    if_win = True
                    # print('winner', goal, [new_stateX_ag, new_stateY_ag])
            # check if agent loses
            if not if_win:

                for enemy in range(0, self.n_enemies, 1):
                    X_en = self.pos_enemies[self.n_steps][enemy][0]
                    Y_en = self.pos_enemies[self.n_steps][enemy][1]
                    if new_stateX_ag == X_en and new_stateY_ag == Y_en:
                        reward = self.reward_loser
                        if_lose = True
                        self.n_times_loser += 1
                        done = False
                        self.n_steps_enemies_actions = 0
                        # print(f'Loser) En: {[X_en, Y_en]}, Ag before: {[last_stateX_ag, last_stateY_ag]}, Action: {action} -> ResAction: {res_action}, Ag after: {[new_stateX_ag, new_stateY_ag]}')
                # otherwise agent is alive
                if not if_lose:
                    # print('alive')
                    reward = self.reward_alive
                    done = False

            new_agents_pos.append([new_stateX_ag, new_stateY_ag])
            self.pos_agents.append(new_agents_pos)

            # print('new inside', [new_stateX_ag, new_stateY_ag])
            rewards.append(reward)
            dones.append(done)

        return new_agents_pos, rewards, dones, if_lose

    def reset(self, reset_n_times_loser):
        # print('reset')
        if reset_n_times_loser:
            self.n_times_loser = 0

        # reset agents' states
        self.n_steps = 0
        reset_rewards = [0] * self.n_agents
        reset_dones = [False] * self.n_agents

        self.pos_agents = []
        self.pos_agents.append(self.pos_agents_for_reset)

        self.pos_enemies = []
        self.pos_enemies.append(self.pos_enemies_for_reset)

        return self.pos_agents[-1], reset_rewards, reset_dones, self.reset_enemies_nearby, self.reset_enemies_attached

    def get_direction(self, x_ag, y_ag, x_en, y_en):
        deltaX = x_en - x_ag
        deltaY = y_en - y_ag

        direction_ag_en = -1

        if deltaX == 0 and deltaY == 0:  # stop
            direction_ag_en = 0
        elif deltaX == 1 and deltaY == 0:  # right
            direction_ag_en = 1
        elif deltaX == -1 and deltaY == 0:  # left
            direction_ag_en = 2
        elif deltaX == 0 and deltaY == 1:  # up
            direction_ag_en = 3
        elif deltaX == 0 and deltaY == -1:  # down
            direction_ag_en = 4
        elif deltaX == 1 and deltaY == 1 and self.n_act_agents > 5:  # diag up right
            direction_ag_en = 5
        elif deltaX == 1 and deltaY == -1 and self.n_act_agents > 5:  # diag down right
            direction_ag_en = 6
        elif deltaX == -1 and deltaY == 1 and self.n_act_agents > 5:  # diag up left
            direction_ag_en = 7
        elif deltaX == -1 and deltaY == -1 and self.n_act_agents > 5:  # diag down left
            direction_ag_en = 8
        else:  # otherwise
            direction_ag_en = 50

        # print([x_ag, y_ag], [x_en, y_en], direction_ag_en)

        return direction_ag_en

    def get_action(self, action, last_stateX, last_stateY, grid):

        if action == 0:  # stop
            new_stateX = last_stateX
            new_stateY = last_stateY
            actionX = 0
            actionY = 0
        elif action == 1:  # right
            sub_action = 1
            if 0 <= sub_action + last_stateX < self.cols:
                new_stateX = sub_action + last_stateX
                actionX = sub_action
            else:
                new_stateX = last_stateX
                actionX = 0
                action = 0
            new_stateY = last_stateY
            actionY = 0
        elif action == 2:  # left
            sub_action = -1
            if 0 <= sub_action + last_stateX < self.cols:
                new_stateX = sub_action + last_stateX
                actionX = sub_action
            else:
                new_stateX = last_stateX
                actionX = 0
                action = 0
            new_stateY = last_stateY
            actionY = 0
        elif action == 3:  # up
            new_stateX = last_stateX
            actionX = 0
            sub_action = 1
            if 0 <= sub_action + last_stateY < self.rows:
                new_stateY = sub_action + last_stateY
                actionY = sub_action
            else:
                new_stateY = last_stateY
                actionY = 0
                action = 0
        elif action == 4:  # down
            new_stateX = last_stateX
            actionX = 0
            sub_action = -1
            if 0 <= sub_action + last_stateY < self.rows:
                new_stateY = sub_action + last_stateY
                actionY = sub_action
            else:
                new_stateY = last_stateY
                actionY = 0
                action = 0

        if grid[new_stateX][new_stateY] == 'W':
            # print('in) wall', [last_stateX, last_stateY], [new_stateX, new_stateY])
            new_stateX = last_stateX
            new_stateY = last_stateY
            action = 0
            actionX = 0
            actionY = 0
            # print('out) wall',[new_stateX, new_stateY])

        return new_stateX, new_stateY, action, actionX, actionY


class GUI:
        #  notice that pygame has x and y coordinates reversed than the implementation done in th "CustomEnv" class
        def __init__(self, rows, cols, n_agents, n_enemies, ag_coord, en_coord, walls_coord, goals_coord, time):
            pygame.font.init()
            self.rows = rows
            self.cols = cols
            self.n_agents = n_agents
            self.n_enemies = n_enemies
            self.pre_ag_coord = ag_coord
            self.pre_en_coord = en_coord
            self.walls_coord = walls_coord
            self.goals_coord = goals_coord
            self.time = time

            self.delay_visualization = 50

            self.font_size = 20
            self.FONT = pygame.font.SysFont('comicsans', self.font_size)

            fix_size_width = 900
            fix_size_height = 750
            WIDTH, HEIGHT = fix_size_width - self.font_size * 2, fix_size_height - self.font_size * 2
            self.WINDOW = pygame.display.set_mode((fix_size_width, fix_size_height))
            pygame.display.set_caption('Game')

            self.width_im = int(WIDTH / self.cols)
            self.height_im = int(HEIGHT / self.rows)
            self.new_sizes = (self.width_im, self.height_im)

            self.pics_agents = []
            for _ in range(self.n_agents):
                self.pics_agents.append(pygame.transform.scale(pygame.image.load('supermario.png'), self.new_sizes))
            self.pics_enemies = []
            for _ in range(self.n_enemies):
                self.pics_enemies.append(pygame.transform.scale(pygame.image.load('bowser.png'), self.new_sizes))
            self.pics_walls = []
            for _ in range(len(self.walls_coord)):
                self.pics_walls.append(pygame.transform.scale(pygame.image.load('wall.png'), self.new_sizes))
            self.pics_goals = []
            for _ in range(len(self.goals_coord)):
                self.pics_goals.append(pygame.transform.scale(pygame.image.load('goal.png'), self.new_sizes))

            self.WINDOW.fill('black')

            for agent in range(self.n_agents):
                self.WINDOW.blit(self.pics_agents[agent], (self.pre_ag_coord[agent][1] * self.width_im,
                                                           self.pre_ag_coord[agent][
                                                               0] * self.height_im + self.font_size * 2))
            for enemy in range(self.n_enemies):
                self.WINDOW.blit(self.pics_enemies[enemy], (self.pre_en_coord[enemy][1] * self.width_im,
                                                            self.pre_en_coord[enemy][
                                                                0] * self.height_im + self.font_size * 2))
            for wall in range(len(self.walls_coord)):
                self.WINDOW.blit(self.pics_walls[wall], (
                    self.walls_coord[wall][1] * self.width_im,
                    self.walls_coord[wall][0] * self.height_im + self.font_size * 2))
            for goal in range(len(self.goals_coord)):
                self.WINDOW.blit(self.pics_goals[goal], (
                    self.goals_coord[goal][1] * self.width_im,
                    self.goals_coord[goal][0] * self.height_im + self.font_size * 2))
            time_text = self.FONT.render(f'', True, 'white')
            self.WINDOW.blit(time_text, (10, 10))
            pygame.display.update()
            pygame.time.delay(self.delay_visualization)

        def __del__(self):
            pygame.QUIT

        def movement(self, new_ag_coord, new_en_coord, time, timeout, algorithm, n_game, n_games, ep, n_ep, n_defeats):
            self.WINDOW.fill('black')

            if new_ag_coord != []:
                for agent in range(self.n_agents):
                    self.WINDOW.blit(self.pics_agents[agent], (
                        new_ag_coord[agent][1] * self.width_im,
                        new_ag_coord[agent][0] * self.height_im + self.font_size * 2))
                    self.pre_ag_coord[agent] = new_ag_coord[agent]
            else:
                for agent in range(self.n_agents):
                    self.WINDOW.blit(self.pics_agents[agent], (self.pre_ag_coord[agent][1] * self.width_im,
                                                               self.pre_ag_coord[agent][
                                                                   0] * self.height_im + self.font_size * 2))

            if new_en_coord != []:
                for enemy in range(self.n_enemies):
                    self.WINDOW.blit(self.pics_enemies[enemy], (
                        new_en_coord[enemy][1] * self.width_im,
                        new_en_coord[enemy][0] * self.height_im + self.font_size * 2))
                    self.pre_en_coord[enemy] = new_en_coord[enemy]
            else:
                for enemy in range(self.n_enemies):
                    self.WINDOW.blit(self.pics_enemies[enemy], (self.pre_en_coord[enemy][1] * self.width_im,
                                                                self.pre_en_coord[enemy][
                                                                    0] * self.height_im + self.font_size * 2))

            for wall in range(len(self.walls_coord)):
                self.WINDOW.blit(self.pics_walls[wall], (
                    self.walls_coord[wall][1] * self.width_im,
                    self.walls_coord[wall][0] * self.height_im + self.font_size * 2))
            for goal in range(len(self.goals_coord)):
                self.WINDOW.blit(self.pics_goals[goal], (
                    self.goals_coord[goal][1] * self.width_im,
                    self.goals_coord[goal][0] * self.height_im + self.font_size * 2))

            time_text = self.FONT.render(
                f'Time: {round((time - self.time) / 60, 2)}/{timeout} min - Game: {n_game}/{n_games} - Episode: {ep}/{n_ep} - Algorithm: {algorithm} - Defeats: {n_defeats}',
                True, 'white')
            self.WINDOW.blit(time_text, (10, 10))
            pygame.display.update()
            pygame.time.delay(self.delay_visualization)
