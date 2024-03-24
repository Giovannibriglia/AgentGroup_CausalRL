import random
import sys
import re
import time
import warnings
import cv2
import numpy as np
import pygame
from queue import PriorityQueue
from collections import deque
from gymnasium.spaces import Discrete
import pygame.camera
import os
from typing import Tuple

warnings.filterwarnings("ignore")

KEY_SAME_ENEMY_ACTIONS = 'same_sequence'
KEY_RANDOM_ENEMY_ACTIONS = 'random'

VALUE_WALL_CELL = -2
VALUE_ENEMY_CELL = -1
VALUE_EMPTY_CELL = 0
VALUE_AGENT_CELL = 1
VALUE_GOAL_CELL = 2

N_WALLS_COEFFICIENT = 2

VAlUE_ENTITY_FAR = 50

len_predefined_enemy_actions = 20

DELAY_VISUALIZATION_VIDEO = 5
FPS_video = 3


class CustomEnv:
    def __init__(self, env_info: dict, if_save_video: bool):

        self.enemy_positions = None
        self.walls_positions = None
        self.goal_nearby = None
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

        self.value_entity_far = VAlUE_ENTITY_FAR
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
        self.if_save_video = if_save_video

        self._init_actions()

        if env_info['enemies_actions'] not in [self.key_same_enemy_actions, self.key_random_enemy_actions]:
            x = env_info['enemies_actions']
            raise AssertionError(f'enemies_actions = {x} invalid')
        else:
            self.enemies_actions = env_info['enemies_actions']
            if self.enemies_actions == self.key_same_enemy_actions:
                self.len_predefined_enemy_actions = len_predefined_enemy_actions
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

        if env_info['predefined_env'] is not None:
            print('to implement already predefined')
            # TODO: to implement already predefined environment

        self.grid_for_game = np.full((self.rows, self.cols), self.value_empty_cell)
        # insert agents, enemies, goals
        self._insert_entities()

        self.reset_enemies_nearby, self.reset_goal_nearby = self.get_nearby_agent()

        if self.if_maze:
            # TODO: develop maze suitable for multi-agent systems, the main problem regards the path for each agent
            self.n_walls = int(max(self.rows, self.cols) * N_WALLS_COEFFICIENT)
            if int(self.rows * self.cols) < self.n_walls + self.n_agents + self.n_goals + self.n_enemies:
                raise AssertionError(f'too many entities for defining a maze: {self.n_walls} walls')
            else:
                # TODO: finish maze definition
                self._define_maze()
        else:
            self.n_walls = 0

        self._vis_grid_numpy()

    def _insert_entities(self):
        # Get matrix dimensions
        total_cells = self.rows * self.cols

        # Initialize list to track used positions
        used_positions = []

        # Initialize arrays to store x and y coordinates of entities
        self.agents_positions = np.zeros((self.n_agents, 2), dtype=int)
        self.enemy_positions = np.zeros((self.n_enemies, 2), dtype=int)
        self.goal_positions = np.zeros((self.n_goals, 2), dtype=int)

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

        self.agent_positions_for_reset = self.agents_positions.copy()

        # Insert enemies
        for i in range(self.n_enemies):
            while True:
                position = np.random.randint(0, total_cells)
                if position not in used_positions:
                    used_positions.append(position)
                    x, y = position // self.cols, position % self.cols
                    self.enemy_positions[i] = [x, y]
                    self.grid_for_game[x, y] = self.value_enemy_cell
                    break

        self.enemy_positions_for_reset = self.enemy_positions.copy()

        # Insert goals
        for i in range(self.n_goals):
            while True:
                position = np.random.randint(0, total_cells)
                if position not in used_positions:
                    used_positions.append(position)
                    x, y = position // self.cols, position % self.cols
                    self.goal_positions[i] = [x, y]
                    self.grid_for_game[x, y] = self.value_goal_cell
                    break

        self.goal_positions_for_reset = self.goal_positions.copy()

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
            # Find empty positions where walls can be placed
            empty_positions = np.argwhere(self.grid_for_game == 0)

            # Remove positions occupied by enemies, goals, agents, and the path
            for position in path:
                empty_positions = empty_positions[np.logical_not(np.all(empty_positions == position, axis=1))]

            # Shuffle the empty positions
            np.random.shuffle(empty_positions)

            self.walls_positions = np.zeros((min(self.n_walls, len(empty_positions)), 2), dtype=int)
            # Place walls randomly in the remaining empty positions
            for i in range(min(self.n_walls, len(empty_positions))):
                x, y = empty_positions[i]
                self.walls_positions[i] = [x, y]
                self.grid_for_game[x, y] = self.value_wall_cell

        random_path = __generate_random_path()
        __place_random_walls(random_path)

    def step_enemies(self):
        # Move enemies based on the enemies movement policy
        if self.enemies_actions == self.key_random_enemy_actions:
            for n, pos_xy_enemy in enumerate(self.enemy_positions):
                action = np.random.randint(0, self.n_actions)
                self.enemy_positions[n] = self._apply_action(pos_xy_enemy, action)
        elif self.enemies_actions == self.key_same_enemy_actions:
            for n, pos_xy_enemy in enumerate(self.enemy_positions):
                action = self.predefined_enemy_actions[self.n_steps_same_enemies_actions][n]
                self.enemy_positions[n] = self._apply_action(pos_xy_enemy, action)
            self.n_steps_same_enemies_actions += 1

    def step_agents(self, actions: int) -> np.ndarray:
        # TODO: multi-agent settings
        for agent in range(self.n_agents):
            pos_xy = self.agents_positions[agent]
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
            goal_position = self.goal_positions[0]

            # Check if agent has reached the goal
            if np.array_equal(agent_position, goal_position):
                rewards.append(self.reward_winner)
                dones.append(True)
                if_loses.append(False)
            # Check if agent collided with any enemy
            elif np.any(np.all(agent_position == self.enemy_positions, axis=1)):
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
        self.goal_nearby = self.reset_goal_nearby
        self.agents_positions = self.agent_positions_for_reset.copy()
        self.enemy_positions = self.enemy_positions_for_reset.copy()

        return self.agents_positions

    def get_nearby_agent(self) -> Tuple[np.ndarray, int]:

        self.enemies_nearby = np.full(self.n_enemies, self.value_entity_far)
        self.goal_nearby = np.full(self.n_goals, self.value_entity_far)

        for agent in range(self.n_agents):
            pos_xy_agent = self.agents_positions[agent]
            for m, pos_xy_goal in enumerate(self.goal_positions):
                self.goal_nearby[m] = self._get_direction(pos_xy_agent, pos_xy_goal)

            for n, pos_xy_enemy in enumerate(self.enemy_positions):
                self.enemies_nearby[n] = self._get_direction(pos_xy_agent, pos_xy_enemy)

        return self.enemies_nearby, self.goal_nearby

    def init_gui(self, algorithm: str, exploration_strategy: str, n_episodes: int, path_images: str):

        def __create_next_alg_folder(base_dir: str, core_word_path: str) -> str:
            # Ensure base directory exists
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)

            # Find existing folders with the given core word path
            alg_folders = [folder for folder in os.listdir(base_dir) if folder.startswith(core_word_path)]

            # Extract numbers from existing folders
            numbers = [int(folder.replace(core_word_path, "")) for folder in alg_folders if
                       folder[len(core_word_path):].isdigit()]

            if numbers:
                next_number = max(numbers) + 1
            else:
                next_number = 1

            while True:
                new_folder_name = f"{core_word_path}{next_number}"
                new_folder_path = os.path.join(base_dir, new_folder_name)
                try:
                    os.makedirs(new_folder_path)
                    print(f"Created folder: {new_folder_path}")
                    return new_folder_path
                except FileExistsError:
                    next_number += 1

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

        self.dir_temp_saving_images = __create_next_alg_folder('temp', f'{self.algorithm}')
        self.count_img = 0

    def movement_gui(self, current_episode: int, step_for_episode: int):

        ag_coord = self.agents_positions.copy()
        en_coord = self.enemy_positions.copy()
        if self.n_walls > 0:
            wall_coord = self.walls_positions.copy()
        else:
            wall_coord = []
        goal_coord = self.goal_positions.copy()

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
            f'Exploration: {self.exploration_strategy} - '
            f'#Defeats: {self.n_times_loser - self.gui_n_defeats_start} - '
            f'#Actions: {step_for_episode}', True, 'white')
        self.WINDOW.blit(time_text, (10, 10))
        pygame.display.update()
        pygame.time.delay(self.delay_visualization)

        # Capture the current Pygame screen and convert it to a NumPy array
        pygame_image = cv2.cvtColor(pygame.surfarray.array3d(pygame.display.get_surface()), cv2.COLOR_RGB2BGR)
        pygame_image = cv2.rotate(pygame_image, cv2.ROTATE_90_CLOCKWISE)
        pygame_image = cv2.flip(pygame_image, 1)

        cv2.imwrite(
            f'{self.dir_temp_saving_images}/im{self.count_img}_{self.algorithm}_{current_episode}episode.jpeg',
            pygame_image)

        self.count_img += 1

    def save_video(self, link_saving: str):

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

        # print(f"Video created and saved to: {os.path.abspath(output_path)}")

    def _init_actions(self):
        self.dict_possible_actions = {0: np.array([0, 0]),  # stop
                                      1: np.array([1, 0]),  # up
                                      2: np.array([-1, 0]),  # down
                                      3: np.array([0, 1]),  # right
                                      4: np.array([0, -1])}  # left
        if len(self.dict_possible_actions.keys()) < self.n_actions:
            raise AssertionError(f'the number of implemented actions is less than expected')

    def _apply_action(self, pos: np.ndarray, action: int) -> np.ndarray:
        new_pos = pos + self.dict_possible_actions[action]
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
        print(named_matrix)
        print('\n')
        time.sleep(2)


if __name__ == '__main__':
    info = {'rows': 5, 'cols': 5, 'n_agents': 1, 'n_enemies': 5, 'n_goals': 1, 'n_actions': 5, 'if_maze': True,
            'value_reward_alive': 0, 'value_reward_winner': 1, 'value_reward_loser': -1, 'seed_value': 4,
            'enemies_actions': 'same_sequence', 'env_type': 'numpy', 'predefined_env': None}

    env = CustomEnv(info, False)

    env.step_enemies()

    env.init_gui('QL', 3000, 'C:\\Users\giova\Documents\Research\CausalRL\images_for_render')

    env.movement_gui(10, 4)
