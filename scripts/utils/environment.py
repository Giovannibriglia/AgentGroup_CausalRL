import os
import random
import re
import shutil
import time
import warnings
from typing import Tuple

import cv2
import numpy as np
import pygame
import pygame.camera
from gymnasium.spaces import Discrete

import global_variables
from global_variables import VALUE_AGENT_CELL, VALUE_GOAL_CELL, VALUE_EMPTY_CELL, VALUE_WALL_CELL, VALUE_ENEMY_CELL, \
    VALUE_ENTITY_FAR, KEY_SAME_ENEMY_ACTIONS, KEY_RANDOM_ENEMY_ACTIONS, LEN_PREDEFINED_ENEMIES_ACTIONS, \
    N_WALLS_COEFFICIENT, DELAY_VISUALIZATION_VIDEO, FPS_video
from scripts.utils.others import create_next_alg_folder

warnings.filterwarnings("ignore")


class CustomEnv:
    def __init__(self, env_info: dict):

        self.enemies_positions = None
        self.walls_positions = None
        self.goals_nearby = None
        self.enemies_nearby = None
        self.agents_positions = None
        self.value_agent_cell = VALUE_AGENT_CELL
        self.value_enemy_cell = VALUE_ENEMY_CELL
        self.value_goal_cell = VALUE_GOAL_CELL
        self.value_empty_cell = VALUE_EMPTY_CELL
        self.value_wall_cell = VALUE_WALL_CELL
        self.number_names_grid = {
            self.value_empty_cell: '-',
            self.value_agent_cell: 'A',
            self.value_goal_cell: 'G',
            self.value_enemy_cell: 'E',
            self.value_wall_cell: 'W'
        }

        self.value_entity_far = VALUE_ENTITY_FAR
        self.key_same_enemy_actions = KEY_SAME_ENEMY_ACTIONS
        self.key_random_enemy_actions = KEY_RANDOM_ENEMY_ACTIONS

        self.n_times_loser = 0
        self.rows = env_info['rows']
        self.cols = env_info['cols']
        self.n_agents = int(env_info['n_agents'])
        self.n_enemies = int(env_info['n_enemies'])
        self.n_goals = int(env_info['n_goals'])
        self.n_actions = int(env_info['n_actions'])
        self.action_space = Discrete(self.n_actions, start=0)
        self.if_maze = env_info['if_maze']
        self.reward_alive = env_info['value_reward_alive']
        self.reward_winner = env_info['value_reward_winner']
        self.reward_loser = env_info['value_reward_loser']
        self.seed_value = env_info['seed_value']
        np.random.seed(self.seed_value)
        random.seed(self.seed_value)

        self._init_actions()

        if env_info['enemies_actions'] not in [self.key_same_enemy_actions, self.key_random_enemy_actions]:
            x = env_info['enemies_actions']
            raise AssertionError(f'enemies_actions = {x} invalid')
        else:
            self.enemies_actions = env_info['enemies_actions']
            if self.enemies_actions == self.key_same_enemy_actions:
                self.len_predefined_enemy_actions = LEN_PREDEFINED_ENEMIES_ACTIONS
                self.n_steps_same_enemies_actions = 0
                self.predefined_enemy_actions = np.zeros((self.n_enemies, self.len_predefined_enemy_actions))
                for enemy in range(self.n_enemies):
                    self.predefined_enemy_actions[enemy] = np.random.randint(0, self.n_actions,
                                                                             size=self.len_predefined_enemy_actions)

        if env_info['env_type'] not in ['numpy', 'torch']:
            x = env_info['env_type']
            raise AssertionError(f'env_type = {x} invalid')
        else:
            self.kind = env_info['env_type']

        self.grid_for_game = np.full((self.rows, self.cols), self.value_empty_cell)
        if env_info['predefined_env'] is not None:
            predefined_env = env_info['predefined_env']

            self.agents_positions = np.array(predefined_env['agents_positions'])
            self.enemies_positions = np.array(predefined_env['enemies_positions'])
            self.goals_positions = np.array(predefined_env['goals_positions'])
            self.walls_positions = np.array(predefined_env['walls_positions'])

            if len(self.walls_positions) > 0:
                self.n_walls = len(self.walls_positions)
                for wall in range(self.n_walls):
                    x, y = self.walls_positions[x, y] = self.value_wall_cell
            else:
                self.n_walls = 0

            for agent in range(len(self.agents_positions)):
                x, y = self.agents_positions[agent]
                self.grid_for_game[x, y] = self.value_agent_cell

            for enemy in range(len(self.enemies_positions)):
                x, y = self.enemies_positions[enemy]
                self.grid_for_game[x, y] = self.value_enemy_cell

            for goal in range(len(self.goals_positions)):
                x, y = self.goals_positions[goal]
                self.grid_for_game[x, y] = self.value_goal_cell
        else:
            # insert agents, enemies, goals
            self._insert_entities()

            if self.if_maze:
                self.n_walls = int(max(self.rows, self.cols) * N_WALLS_COEFFICIENT)
                if int(self.rows * self.cols) < self.n_walls + self.n_agents + self.n_goals + self.n_enemies:
                    raise AssertionError(f'too many entities for defining a maze: {self.n_walls} walls')
                else:
                    self._define_maze()
            else:
                self.n_walls = 0

        self.agent_positions_for_reset = self.agents_positions.copy()
        self.enemy_positions_for_reset = self.enemies_positions.copy()
        self.goal_positions_for_reset = self.goals_positions.copy()

        self.reset_enemies_nearby, self.reset_goal_nearby = self.get_nearby_agent()

        self._vis_grid_numpy()

    def _insert_entities(self):
        # Get matrix dimensions
        total_cells = self.rows * self.cols

        # Initialize a list to track used positions
        used_positions = []

        # Initialize arrays to store x and y coordinates of entities
        self.agents_positions = np.zeros((self.n_agents, 2), dtype=int)
        self.enemies_positions = np.zeros((self.n_enemies, 2), dtype=int)
        self.goals_positions = np.zeros((self.n_goals, 2), dtype=int)

        # Insert agents
        for i in range(self.n_agents):
            while True:
                position = np.random.randint(0, total_cells)
                if position not in used_positions:
                    used_positions.append(position)
                    x, y = position // self.cols, position % self.cols
                    self.agents_positions[i] = [x, y]
                    self.grid_for_game[x, y] = self.value_agent_cell
                    break

        # Insert enemies
        for i in range(self.n_enemies):
            while True:
                position = np.random.randint(0, total_cells)
                if position not in used_positions:
                    used_positions.append(position)
                    x, y = position // self.cols, position % self.cols
                    self.enemies_positions[i] = [x, y]
                    self.grid_for_game[x, y] = self.value_enemy_cell
                    break

        # Insert goals
        for i in range(self.n_goals):
            while True:
                position = np.random.randint(0, total_cells)
                if position not in used_positions:
                    used_positions.append(position)
                    x, y = position // self.cols, position % self.cols
                    self.goals_positions[i] = [x, y]
                    self.grid_for_game[x, y] = self.value_goal_cell
                    break

    def _define_maze(self):

        def __generate_random_path():
            # Find agent and goal positions
            agent_position = np.argwhere(self.grid_for_game == self.value_agent_cell)[0]
            goal_position = np.argwhere(self.grid_for_game == self.value_goal_cell)[0]

            # Define possible moves (up, down, left, right)
            moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

            # Initialize path with agent position
            path = [agent_position]

            # Iterate until agent reaches the goal
            while tuple(path[-1]) != tuple(goal_position):
                # Choose a random move
                move = moves[np.random.randint(0, len(moves))]

                # Calculate new position
                new_position = (path[-1][0] + move[0], path[-1][1] + move[1])

                # Check if the new position is within the grid and not an enemy (-1)
                if (0 <= new_position[0] < self.grid_for_game.shape[0] and
                        0 <= new_position[1] < self.grid_for_game.shape[1] and
                        self.grid_for_game[new_position[0], new_position[1]] != self.value_enemy_cell):
                    # Add new position to the path
                    path.append(new_position)

            return path

        def __place_random_walls(path):

            cells_with_entity = np.transpose(np.where(self.grid_for_game == global_variables.VALUE_EMPTY_CELL))
            empty_positions = np.concatenate((cells_with_entity, path))

            if len(empty_positions) < self.n_walls:
                print("Error: Insufficient empty cells.")

            self.walls_positions = np.empty((self.n_walls, 2))

            selected_positions = np.random.choice(len(empty_positions), self.n_walls, replace=False)

            for n, idx in enumerate(selected_positions):
                i, j = empty_positions[idx]
                self.walls_positions[n] = [i, j]
                self.grid_for_game[i, j] = global_variables.VALUE_WALL_CELL

        random_path = __generate_random_path()
        __place_random_walls(random_path)

    def step_enemies(self):
        # Move enemies based on the enemies movement policy
        if self.enemies_actions == self.key_random_enemy_actions:
            for n, pos_xy_enemy in enumerate(self.enemies_positions):
                action = np.random.randint(0, self.n_actions)
                self.enemies_positions[n] = self._apply_action(pos_xy_enemy, action)
        elif self.enemies_actions == self.key_same_enemy_actions:
            for n, pos_xy_enemy in enumerate(self.enemies_positions):
                action = self.predefined_enemy_actions[self.n_steps_same_enemies_actions][n]
                self.enemies_positions[n] = self._apply_action(pos_xy_enemy, action)
            self.n_steps_same_enemies_actions += 1

    def step_agents(self, actions: list) -> np.ndarray:
        pos_agents = self.agents_positions.copy()
        for agent in range(self.n_agents):
            pos_xy = pos_agents[agent]
            action = actions[agent]

            self.agents_positions[agent] = self._apply_action(pos_xy, action)

        return self.agents_positions

    def check_winner_gameover_agents(self) -> Tuple[list, list, list]:
        """
        Check if the game is over and who won.
        Returns: tuple: A tuple containing reward, done flag, and if_lose flag.
        """

        rewards = []
        dones = []
        if_loses = []

        for agent in range(self.n_agents):
            agent_position = self.agents_positions[agent]
            goal_position = self.goals_positions[0]

            # Check if agent has reached the goal
            if np.array_equal(agent_position, goal_position):
                rewards.append(self.reward_winner)
                dones.append(True)
                if_loses.append(False)
            # Check if agent collided with any enemy
            elif np.any(np.all(agent_position == self.enemies_positions, axis=1)):
                self.n_times_loser += 1
                self.n_steps_same_enemies_actions = 0
                rewards.append(self.reward_loser)
                dones.append(False)
                if_loses.append(True)
            else:
                rewards.append(self.reward_alive)
                dones.append(False)
                if_loses.append(False)

        return rewards, dones, if_loses

    def reset(self, if_reset_n_time_loser) -> np.ndarray:
        if if_reset_n_time_loser:
            self.n_times_loser = 0

        self.enemies_nearby = self.reset_enemies_nearby.copy()
        self.goals_nearby = self.reset_goal_nearby.copy()
        self.agents_positions = self.agent_positions_for_reset.copy()
        self.enemies_positions = self.enemy_positions_for_reset.copy()

        return self.agents_positions

    def get_nearby_agent(self) -> Tuple[np.ndarray, np.ndarray]:

        def _remove_value(arr, value):
            arr = np.array(arr)
            return np.where(arr == value, np.nan, arr)

        self.enemies_nearby = np.full((self.n_agents, self.n_enemies), self.value_entity_far)
        self.goals_nearby = np.full((self.n_agents, self.n_goals), self.value_entity_far)

        for agent in range(self.n_agents):
            pos_xy_agent = self.agents_positions[agent]
            for m, pos_xy_goal in enumerate(self.goals_positions):
                self.goals_nearby[agent][m] = self._get_direction(pos_xy_agent, pos_xy_goal)

            for n, pos_xy_enemy in enumerate(self.enemies_positions):
                self.enemies_nearby[agent][n] = self._get_direction(pos_xy_agent, pos_xy_enemy)

        self.enemies_nearby = _remove_value(self.enemies_nearby, self.value_entity_far)
        self.goals_nearby = _remove_value(self.goals_nearby, self.value_entity_far)
        return self.enemies_nearby, self.goals_nearby

    def init_gui(self, algorithm: str, exploration_strategy: str, n_episodes: int, path_images: str, save_video: bool):

        self.if_save_video = save_video

        # GUI must be instantiated once for each simulation
        agent_png = f'{path_images}/supermario.png'
        enemy_png = f'{path_images}/bowser.png'
        wall_png = f'{path_images}/wall.png'
        goal_png = f'{path_images}/goal.png'

        pygame.font.init()

        self.algorithm = algorithm
        self.exploration_strategy = exploration_strategy
        self.n_episodes = n_episodes
        self.delay_visualization = DELAY_VISUALIZATION_VIDEO
        self.gui_n_defeats_start = self.n_times_loser
        self.font_size = 20
        self.FONT = pygame.font.SysFont('comicsans', self.font_size)

        fix_size_width = 1080
        fix_size_height = 720
        WIDTH, HEIGHT = fix_size_width - self.font_size * 2, fix_size_height - self.font_size * 2
        self.WINDOW = pygame.display.set_mode((fix_size_width, fix_size_height))
        pygame.display.set_caption('Game')

        self.width_im = int(WIDTH / self.cols)
        self.height_im = int(HEIGHT / self.rows)
        self.new_sizes = (self.width_im, self.height_im)

        self.pics_agents = []
        for _ in range(self.n_agents):
            self.pics_agents.append(pygame.transform.scale(pygame.image.load(agent_png), self.new_sizes))
        self.pics_enemies = []
        for _ in range(self.n_enemies):
            self.pics_enemies.append(pygame.transform.scale(pygame.image.load(enemy_png), self.new_sizes))
        self.pics_walls = []
        for _ in range(self.n_walls):
            self.pics_walls.append(pygame.transform.scale(pygame.image.load(wall_png), self.new_sizes))
        self.pics_goals = []
        for _ in range(self.n_goals):
            self.pics_goals.append(pygame.transform.scale(pygame.image.load(goal_png), self.new_sizes))

        if self.if_save_video:
            self.dir_temp_saving_images = create_next_alg_folder('../temp', f'images_{self.algorithm}')
            self.count_img = 0

    def movement_gui(self, current_episode: int, step_for_episode: int):

        ag_coord = self.agents_positions.copy()
        en_coord = self.enemies_positions.copy()
        if self.n_walls > 0:
            wall_coord = self.walls_positions.copy()
        else:
            wall_coord = []
        goal_coord = self.goals_positions.copy()

        self.WINDOW.fill('black')

        for agent in range(self.n_agents):
            self.WINDOW.blit(self.pics_agents[agent], (ag_coord[agent][1] * self.width_im,
                                                       ag_coord[agent][
                                                           0] * self.height_im + self.font_size * 2))

        for enemy in range(self.n_enemies):
            self.WINDOW.blit(self.pics_enemies[enemy], (en_coord[enemy][1] * self.width_im,
                                                        en_coord[enemy][
                                                            0] * self.height_im + self.font_size * 2))

        for wall in range(len(wall_coord)):
            self.WINDOW.blit(self.pics_walls[wall], (
                wall_coord[wall][1] * self.width_im,
                wall_coord[wall][0] * self.height_im + self.font_size * 2))

        for goal in range(self.n_goals):
            self.WINDOW.blit(self.pics_goals[goal], (
                goal_coord[goal][1] * self.width_im,
                goal_coord[goal][0] * self.height_im + self.font_size * 2))

        time_text = self.FONT.render(
            f'Episode: {current_episode}/{self.n_episodes} - Algorithm: {self.algorithm} - '
            # f'Exploration: {self.exploration_strategy} - '
            f'#Defeats: {self.n_times_loser - self.gui_n_defeats_start} - '
            f'#Actions: {step_for_episode}', True, 'white')
        self.WINDOW.blit(time_text, (10, 10))
        pygame.display.update()
        pygame.time.delay(self.delay_visualization)

        # Capture the current Pygame screen and convert it to a NumPy array
        pygame_image = cv2.cvtColor(pygame.surfarray.array3d(pygame.display.get_surface()), cv2.COLOR_RGB2BGR)
        pygame_image = cv2.rotate(pygame_image, cv2.ROTATE_90_CLOCKWISE)
        pygame_image = cv2.flip(pygame_image, 1)
        if self.if_save_video:
            cv2.imwrite(
                f'{self.dir_temp_saving_images}/im{self.count_img}_{self.algorithm}_{current_episode}episode.jpeg',
                pygame_image)

            self.count_img += 1

    def video_saving(self, link_saving: str):

        def sort_key(filename):
            # Extract the numeric part from the filename using regular expression
            numeric_part = re.search(r'\d+', filename).group()
            # Convert the extracted numeric part to an integer for sorting
            return int(numeric_part)

        # Set the path to the directory containing your images
        image_folder = self.dir_temp_saving_images

        # Set the frame rate (frames per second) for the video
        fps = FPS_video

        # Get the list of image filenames in the specified folder
        image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])

        # Check if there are any images in the folder
        if not image_files:
            print("No image files found in the specified folder.")
            exit()

        # Get the dimensions of the first image to determine the video resolution
        first_image_path = os.path.join(image_folder, image_files[0])
        img = cv2.imread(first_image_path)
        height, width, _ = img.shape

        # Create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs like 'XVID' or 'H264'
        out = cv2.VideoWriter(link_saving, fourcc, fps, (width, height))

        image_files = sorted(image_files, key=sort_key)

        # Write each image to the video file
        for image_file in image_files:
            image_path = os.path.join(image_folder, image_file)
            img = cv2.imread(image_path)

            # Ensure the image was read successfully
            if img is not None:
                out.write(img)
                os.remove(image_path)
            else:
                print(f"Failed to read image: {image_path}")

        # Release the VideoWriter object
        out.release()

        # Delete folder with temp images for video
        shutil.rmtree(self.dir_temp_saving_images)

    def _init_actions(self):
        self.dict_possible_actions = global_variables.DICT_IMPLEMENTED_ACTIONS
        if len(self.dict_possible_actions.keys()) < self.n_actions:
            raise AssertionError(f'the number of implemented actions is less than expected')

    def _apply_action(self, pos: np.ndarray, action: int) -> np.ndarray:
        new_pos = pos + self.dict_possible_actions[action]

        if abs((new_pos[1] - pos[1]) - (new_pos[0] - pos[0])) > 1:
            print('Start position: ', pos)
            print('Action: ', action)
            print('Final position: ', new_pos)
            raise AssertionError('Actions has moved entity wrong')

        if self.__is_valid_position(new_pos) and not self.__is_wall(new_pos):
            return new_pos
        return pos.copy()

    def __is_valid_position(self, pos: np.ndarray) -> bool:
        row, col = pos
        return 0 <= row < self.rows and 0 <= col < self.cols

    def __is_wall(self, pos: np.ndarray) -> bool:
        return self.grid_for_game[pos[0], pos[1]] == self.value_wall_cell

    def _get_direction(self, pos_xy_agent, pos_xy_other) -> int:
        def __find_key(dict_actions: dict, value: tuple) -> int:
            for key, val in dict_actions.items():
                if np.array_equal(val, value):
                    return key
            return self.value_entity_far  # If value is not found in the dictionary

        return __find_key(self.dict_possible_actions, pos_xy_other - pos_xy_agent)

    def _vis_grid_numpy(self):
        # Replace the numbers in the matrix with their names
        named_matrix = np.vectorize(lambda x: self.number_names_grid.get(x, str(x)))(self.grid_for_game)
        # Print the named matrix
        print(named_matrix, '\n')
        time.sleep(2)
