import random
from rl_base import Env
import numpy as np
from typing import List, NamedTuple
from enum import Enum
from pygame_config import *
from copy import deepcopy


class WumpusSetting(NamedTuple):
    action_costs: List
    event_rewards: List
    random_grid: bool
    taking_gold_ends: bool
    n_golds: int
    n_pits: int
    n_wumpuses: int


class Action(Enum):
    FORWARD=0; TURN_LEFT=1; TURN_RIGHT=2; TAKE_GOLD=3; SHOOT=4; CLIMB=5


class Event(Enum):
    LEFT_WITH_GOLD=0; TOOK_GOLD=1; DEATH_BY_WUMPUS=2; DEATH_BY_PIT=3
    WUMPUS_KILLED=4; BUMP=5; NOT_VISITED_FIELD=6


actions_desc = ["forward", "turn left", "turn right", "take gold", "shoot", "climb out"]
event_desc = ['Left with gold!', 'Gold taken!', 'Killed by wumpus!', 'Killed by pit!', 'Wumpus killed!', 'Bump!',
              'New field']

wumpus_settings = {
    'static_wumpus': WumpusSetting(
        action_costs=[-0.1, -5., -5., -20, -20, -20],
        event_rewards=[1000, 500, -1000, -1000, 300, -5, 5],
        random_grid=False, taking_gold_ends=False, n_golds=1, n_pits=3, n_wumpuses=1),
    'only_gold': WumpusSetting(
        action_costs=[-0.1, -5., -5., -20, -20, -20],
        event_rewards=[1000, 500, -1000, -1000, 300, -5, 5],
        random_grid=True, taking_gold_ends=False, n_golds=1, n_pits=0, n_wumpuses=0),
    'one_pit_only': WumpusSetting(
        action_costs=[-1, -20., -20., -40, -40, -40],
        event_rewards=[1000, 500, -1000, -1000, 300, -5, 5],
        random_grid=True, taking_gold_ends=False, n_golds=1, n_pits=1, n_wumpuses=0),
    'wumpus_gold_no_pits': WumpusSetting(
        action_costs=[-0.1, -5., -5., -20, -20, -20],
        event_rewards=[1000, 500, -1000, -1000, 300, -5, 50],
        random_grid=True, taking_gold_ends=False, n_golds=1, n_pits=0, n_wumpuses=1),
    'wumpus_one_pit': WumpusSetting(
        action_costs=[-1, -20., -20., -40, -40, -40],
        event_rewards=[1000, 500, -1000, -1000, 300, -5, 5],
        random_grid=True, taking_gold_ends=False, n_golds=1, n_pits=1, n_wumpuses=1),
    'wumpus_two_pits': WumpusSetting(
        action_costs=[-0.1, -5., -5., -20, -20, -20],
        event_rewards=[1000, 500, -1000, -1000, 300, -5, 5],
        random_grid=True, taking_gold_ends=False, n_golds=1, n_pits=2, n_wumpuses=1),
    'full_wumpus': WumpusSetting(
        action_costs=[-0.1, -5., -5., -20, -20, -20],
        event_rewards=[1000, 500, -1000, -1000, 300, -5, 5],
        random_grid=True, taking_gold_ends=False, n_golds=1, n_pits=3, n_wumpuses=1),
}


class FieldType(Enum):
    WUMPUS="W"; PIT="P"; GOLD="G"; REGULAR=" "; VISITED="."; AGENT="A"


field_to_asset_key = {
    FieldType.WUMPUS: 'wumpus',
    FieldType.GOLD: 'gold',
    FieldType.REGULAR: 'regular',
}


CAVE_ENTRY_X = 3; CAVE_ENTRY_Y = 0


class GridWorld:
    def __init__(self, random_grid=True, n_wumpuses=1, n_golds=1, n_pits=3):
        self.objects: List[List[FieldType]] = self.get_random_obj_grid(n_wumpuses, n_golds, n_pits)\
            if random_grid else self. get_stable_obj_grid()
        self.shape = (len(self.objects), len(self.objects[0]))
        self.breezes = np.zeros(self.shape)
        self.stenches = np.zeros(self.shape)
        self.visited = np.zeros(self.shape)
        self.fill_senses_grids()

    @staticmethod
    def get_random_obj_grid(n_wumpuses, n_golds, n_pits):
        size_x = 4
        size_y = 4
        env = []
        for _ in range(size_x):
            env.append([FieldType.REGULAR] * size_y)
        # noinspection PyTypeChecker
        env[CAVE_ENTRY_X][CAVE_ENTRY_Y] = ""
        # noinspection PyTypeChecker
        env[CAVE_ENTRY_X-1][CAVE_ENTRY_Y] = ""
        # noinspection PyTypeChecker
        env[CAVE_ENTRY_X][CAVE_ENTRY_Y+1] = ""
        # noinspection PyTypeChecker
        env[CAVE_ENTRY_X-1][CAVE_ENTRY_Y+1] = ""

        wumpus_nr = n_wumpuses
        gold_nr = n_golds
        pit_nr = n_pits
        while (wumpus_nr > 0) | (gold_nr > 0) | (pit_nr > 0):
            random_x = random.randint(0, size_x - 1)
            random_y = random.randint(0, size_y - 1)
            if wumpus_nr > 0:
                if env[random_x][random_y] == FieldType.REGULAR:
                    wumpus_nr -= 1
                    env[random_x][random_y] = FieldType.WUMPUS
            elif gold_nr > 0:
                if env[random_x][random_y] == FieldType.REGULAR:
                    gold_nr -= 1
                    env[random_x][random_y] = FieldType.GOLD
            elif pit_nr > 0:
                if env[random_x][random_y] == FieldType.REGULAR:
                    pit_nr -= 1
                    env[random_x][random_y] = FieldType.PIT

        env[CAVE_ENTRY_X][CAVE_ENTRY_Y] = FieldType.REGULAR
        env[CAVE_ENTRY_X-1][CAVE_ENTRY_Y] = FieldType.REGULAR
        env[CAVE_ENTRY_X][CAVE_ENTRY_Y+1] = FieldType.REGULAR
        env[CAVE_ENTRY_X-1][CAVE_ENTRY_Y+1] = FieldType.REGULAR

        return env

    @staticmethod
    def get_stable_obj_grid():
        env = [[FieldType.REGULAR, FieldType.PIT, FieldType.REGULAR, FieldType.GOLD],
               [FieldType.WUMPUS, FieldType.REGULAR, FieldType.REGULAR, FieldType.REGULAR],
               [FieldType.REGULAR, FieldType.REGULAR, FieldType.REGULAR, FieldType.REGULAR],
               [FieldType.REGULAR, FieldType.REGULAR, FieldType.PIT, FieldType.REGULAR]]

        return env

    def neighbouring_cells(self, x, y):
        cells = []
        if x > 0:
            cells.append((x-1, y))
        if y > 0:
            cells.append((x, y-1))
        if x+1 < self.shape[0]:
            cells.append((x+1, y))
        if y+1 < self.shape[1]:
            cells.append((x, y+1))
        return cells

    def fill_senses_grids(self):
        self.breezes *= 0
        self.stenches *= 0
        for x, row in enumerate(self.objects):
            for y, obj in enumerate(row):
                if obj == FieldType.PIT:
                    for (nx, ny) in self.neighbouring_cells(x, y):
                        self.breezes[nx, ny] = 1
                if obj == FieldType.WUMPUS:
                    for (nx, ny) in self.neighbouring_cells(x, y):
                        self.stenches[nx, ny] = 1

    def at_xy(self, x, y):
        return self.objects[x][y], self.breezes[x, y], self.stenches[x, y]


class Direction(Enum):
    UP=0; LEFT=1; DOWN=2; RIGHT=3


class Sense(Enum):
    STENCH=0; BREEZE=1; GLITTER=2; BUMP=3; SCREAM=4


class AgentState:
    def __init__(self, start_posx, start_posy, n_arrows, grids: GridWorld):
        self.pos_x = start_posx
        self.pos_y = start_posy
        self.agent_direction = Direction.UP
        self.gold_taken = False
        self.arrows_left = n_arrows
        self.senses = [False] * len(Sense)
        self.update_senses(grids)

    def update_senses(self, grids: GridWorld, bump=False, scream=False):
        obj, breeze, stench = grids.at_xy(self.pos_x, self.pos_y)
        self.senses[Sense.STENCH.value] = stench == 1
        self.senses[Sense.BREEZE.value] = breeze == 1
        self.senses[Sense.GLITTER.value] = obj == FieldType.GOLD
        self.senses[Sense.BUMP.value] = bump
        self.senses[Sense.SCREAM.value] = scream

    def __str__(self):
        s = f"Position=({self.pos_x}, {self.pos_y}); Direction={self.agent_direction.name};"
        s += f"Arrows left={self.arrows_left}; Gold={self.gold_taken};"
        s += f"Sensing: "
        for n, v in enumerate(self.senses):
            if v:
                s += Sense(n).name + ' '
        return s


class WumpusWorld(Env):

    def __init__(self, settings: WumpusSetting, name="Wumpus"):
        super().__init__(name)
        self.settings = settings
        self.grids = GridWorld(settings.random_grid, settings.n_wumpuses,
                               settings.n_golds, settings.n_pits)
        self.agent_state = AgentState(CAVE_ENTRY_X, CAVE_ENTRY_Y, settings.n_wumpuses, self.grids)
        # for rendering
        self.assets = None

    def reset_env(self):
        self.grids = GridWorld(self.settings.random_grid, self.settings.n_wumpuses, self.settings.n_golds,
                               self.settings.n_pits)
        self.agent_state = AgentState(CAVE_ENTRY_X, CAVE_ENTRY_Y, self.settings.n_wumpuses, self.grids)
        return self.get_state()

    def get_state(self):
        return deepcopy(self.agent_state)

    def try_to_kill_wumpus_at_xy(self, x, y):
        if self.grids.objects[x][y] == FieldType.WUMPUS:
            self.grids.objects[x][y] = FieldType.REGULAR
            self.grids.fill_senses_grids()
            return True
        else:
            return False

    def step(self, action_):
        action = Action(action_)

        reward = self.settings.action_costs[action_]
        info = [f"Agent action: {actions_desc[action_]}."]

        self.grids.visited[self.agent_state.pos_x, self.agent_state.pos_y] = 1

        bump = False
        scream = False
        done = False
        game_won = False

        if action == Action.FORWARD:
            if self.agent_state.agent_direction == Direction.UP:
                self.agent_state.pos_x -= 1
                if self.agent_state.pos_x < 0:
                    self.agent_state.pos_x = 0
                    bump = True
            elif self.agent_state.agent_direction == Direction.RIGHT:
                self.agent_state.pos_y += 1
                if self.agent_state.pos_y >= self.grids.shape[1]:
                    bump = True
                    self.agent_state.pos_y -= 1
            elif self.agent_state.agent_direction == Direction.DOWN:
                self.agent_state.pos_x += 1
                if self.agent_state.pos_x >= self.grids.shape[0]:
                    bump = True
                    self.agent_state.pos_x -= 1
            elif self.agent_state.agent_direction == Direction.LEFT:
                self.agent_state.pos_y -= 1
                if self.agent_state.pos_y < 0:
                    bump = True
                    self.agent_state.pos_y = 0
            else:
                raise Exception("Invalid agent direction")
            # check if met Wumpus or fell into pit
            event = None
            if self.grids.objects[self.agent_state.pos_x][self.agent_state.pos_y] == FieldType.WUMPUS:
                done = True
                event = Event.DEATH_BY_WUMPUS
            elif self.grids.objects[self.agent_state.pos_x][self.agent_state.pos_y] == FieldType.PIT:
                done = True
                event = Event.DEATH_BY_PIT
            else:
                if bump:
                    event = Event.BUMP
                if self.grids.visited[self.agent_state.pos_x][self.agent_state.pos_y] == 0:
                    event = Event.NOT_VISITED_FIELD
            if event is not None:
                reward += self.settings.event_rewards[event.value]
                info += [event_desc[event.value]]

        elif action == Action.TURN_LEFT:
            self.agent_state.agent_direction = Direction((self.agent_state.agent_direction.value + 1) % 4)
        elif action == Action.TURN_RIGHT:
            self.agent_state.agent_direction = Direction((self.agent_state.agent_direction.value - 1) % 4)
        elif action == Action.TAKE_GOLD:
            if self.grids.objects[self.agent_state.pos_x][self.agent_state.pos_y] == FieldType.GOLD:
                self.grids.objects[self.agent_state.pos_x][self.agent_state.pos_y] = FieldType.REGULAR
                self.agent_state.gold_taken = True
                # print('TOOK GOLD!')
                reward += self.settings.event_rewards[Event.TOOK_GOLD.value]
                info += [event_desc[Event.TOOK_GOLD.value]]
                if self.settings.taking_gold_ends:
                    done = True
                    game_won = True
            else:
                info += ["No gold here."]
        elif action == Action.SHOOT:
            wumpus_killed = False
            if self.agent_state.arrows_left > 0:
                self.agent_state.arrows_left -= 1
                if self.agent_state.agent_direction == Direction.UP:
                    for x in range(self.agent_state.pos_x-1, -1, -1):
                        if self.try_to_kill_wumpus_at_xy(x, self.agent_state.pos_y):
                            wumpus_killed = True
                            break
                elif self.agent_state.agent_direction == Direction.DOWN:
                    for x in range(self.agent_state.pos_x + 1, self.grids.shape[0]):
                        if self.try_to_kill_wumpus_at_xy(x, self.agent_state.pos_y):
                            wumpus_killed = True
                            break
                elif self.agent_state.agent_direction == Direction.LEFT:
                    for y in range(self.agent_state.pos_y - 1, -1, -1):
                        if self.try_to_kill_wumpus_at_xy(self.agent_state.pos_x, y):
                            wumpus_killed = True
                            break
                elif self.agent_state.agent_direction == Direction.RIGHT:
                    for y in range(self.agent_state.pos_y + 1, self.grids.shape[1]):
                        if self.try_to_kill_wumpus_at_xy(self.agent_state.pos_x, y):
                            wumpus_killed = True
                            break
                if wumpus_killed:
                    scream = True
                    reward += self.settings.event_rewards[Event.WUMPUS_KILLED.value]
                    info += [event_desc[Event.WUMPUS_KILLED.value]]
            else:
                info += ["Arrow no available, nothing happened."]
        elif action == Action.CLIMB:
            if (self.agent_state.pos_x == CAVE_ENTRY_X) & (self.agent_state.pos_y == CAVE_ENTRY_Y)\
                    & self.agent_state.gold_taken:
                reward += self.settings.event_rewards[Event.LEFT_WITH_GOLD.value]
                info += [event_desc[Event.LEFT_WITH_GOLD.value]]
                done = True
                game_won = True
            else:
                info += ["Can't leave yet."]
        else:
            raise Exception("Invalid action")

        self.agent_state.update_senses(self.grids, bump, scream)

        return self.get_state(), reward, done, info, game_won

    def render(self, screen, text, *args):
        if not self.assets:
            self.assets = load_assets()

        screen.fill(WHITE)

        for x in range(self.grids.shape[0]):
            for y in range(self.grids.shape[1]):
                if (x == self.agent_state.pos_x) and (y == self.agent_state.pos_y):
                    screen.blit(self.assets[self.agent_state.agent_direction], (y * FIELD_SIZE_X, x * FIELD_SIZE_Y,
                                                                                FIELD_SIZE_X, FIELD_SIZE_Y))
                elif (x == CAVE_ENTRY_X) & (y == CAVE_ENTRY_Y):
                    screen.blit(self.assets['cave_entry'], (y * FIELD_SIZE_X, x * FIELD_SIZE_Y,
                                                            FIELD_SIZE_X, FIELD_SIZE_Y))
                else:
                    obj, _, _ = self.grids.at_xy(x, y)
                    if obj == FieldType.REGULAR:
                        color = GREEN if self.grids.visited[x][y] else BROWN
                        pygame.draw.rect(screen, color, pygame.Rect(y * FIELD_SIZE_X, x * FIELD_SIZE_Y,
                                                                    FIELD_SIZE_X, FIELD_SIZE_Y))
                    else:       # pit wumpus or gold
                        screen.blit(self.assets[obj], (y * FIELD_SIZE_X, x * FIELD_SIZE_Y, FIELD_SIZE_X, FIELD_SIZE_Y))

                pygame.draw.rect(screen, BLUE, pygame.Rect(y * FIELD_SIZE_X, x * FIELD_SIZE_Y,
                                                           FIELD_SIZE_X, FIELD_SIZE_Y), 5)

        for t in range(len(text)):
            msg = self.assets['font'].render(text[t], False, BLACK)
            screen.blit(msg, (FIELD_SIZE_X * self.grids.shape[1] + 10, t * 25))

        pygame.display.flip()


def load_assets():
    assets = {}

    pit_img = pygame.image.load("assets/pit_img.png").convert()
    pit_img = pygame.transform.scale(pit_img, (FIELD_SIZE_X, FIELD_SIZE_Y))
    assets[FieldType.PIT] = pit_img

    wumpus_img = pygame.image.load("assets/wumpus_img.png").convert()
    wumpus_img = pygame.transform.scale(wumpus_img, (FIELD_SIZE_X, FIELD_SIZE_Y))
    assets[FieldType.WUMPUS] = wumpus_img

    gold_img = pygame.image.load("assets/gold_img.png").convert()
    gold_img = pygame.transform.scale(gold_img, (FIELD_SIZE_X, FIELD_SIZE_Y))
    assets[FieldType.GOLD] = gold_img

    agent_img_left = pygame.image.load("assets/arrow_img2.png").convert_alpha()
    agent_img_left = pygame.transform.scale(agent_img_left, (FIELD_SIZE_X, FIELD_SIZE_Y))
    assets[Direction.LEFT] = agent_img_left

    agent_img_down = pygame.image.load("assets/arrow_img2.png").convert_alpha()
    agent_img_down = pygame.transform.scale(agent_img_down, (FIELD_SIZE_X, FIELD_SIZE_Y))
    agent_img_down = pygame.transform.rotate(agent_img_down, 90)
    assets[Direction.DOWN] = agent_img_down

    agent_img_right = pygame.image.load("assets/arrow_img2.png").convert_alpha()
    agent_img_right = pygame.transform.scale(agent_img_right, (FIELD_SIZE_X, FIELD_SIZE_Y))
    agent_img_right = pygame.transform.rotate(agent_img_right, 180)
    assets[Direction.RIGHT] = agent_img_right

    agent_img_up = pygame.image.load("assets/arrow_img2.png").convert_alpha()
    agent_img_up = pygame.transform.scale(agent_img_up, (FIELD_SIZE_X, FIELD_SIZE_Y))
    agent_img_up = pygame.transform.rotate(agent_img_up, 270)
    assets[Direction.UP] = agent_img_up

    cave_entry_img = pygame.image.load("assets/cave_entry_img.png").convert()
    cave_entry_img = pygame.transform.scale(cave_entry_img, (FIELD_SIZE_X, FIELD_SIZE_Y))
    assets['cave_entry'] = cave_entry_img

    assets['font'] = pygame.font.SysFont('Times New Roman', 18)

    return assets
