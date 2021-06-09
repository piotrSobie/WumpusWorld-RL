import random
import numpy as np

from pygame_config import *


# 4x4 constant grid world
# agent can take actions (4): go up, down, right, left
# states (16): each grid represents different state
# stochastic environment
class FrozenLake:
    def __init__(self):
        # go up, down, right, left
        self.action_space = [0, 1, 2, 3]
        self.action_space_n = len(self.action_space)

        self.pit_field = "P"
        self.gold_field = "G"
        self.regular_field = "F"
        self.visited_field = "V"
        self.agent_field = "A"

        self.observation_space_n = None
        self.state_number = None  # array, size the same as gridworld, it looks like: [[0, 1, 2, 3], ... ,[12,13,14,15]]
        # i use it to find agent state, different way is state = 4 * agent_pos_y + agent_pos_x, but array way works, so
        # i didn't change it
        self.grid_world = self.get_new_env()
        self.cave_entry_x = 3
        self.cave_entry_y = 0
        self.agentPosXY = [self.cave_entry_x, self.cave_entry_y]

        self.intended_action_prob = 0.5
        self.side_from_intended_prob = 0.25
        self.opposite_from_intended_prob = 0.0
        assert self.intended_action_prob + 2*self.side_from_intended_prob + self.opposite_from_intended_prob == 1.0, \
            "action probabilities must sum to 1.0"

        self.action_names = {0: "up", 1: "down", 2: "right", 3: "left"}

        self.action_prob = {0: np.array([self.intended_action_prob, self.opposite_from_intended_prob, self.side_from_intended_prob, self.side_from_intended_prob]),
                            1: np.array([self.opposite_from_intended_prob, self.intended_action_prob, self.side_from_intended_prob, self.side_from_intended_prob]),
                            2: np.array([self.side_from_intended_prob, self.side_from_intended_prob, self.intended_action_prob, self.opposite_from_intended_prob]),
                            3: np.array([self.side_from_intended_prob, self.side_from_intended_prob, self.opposite_from_intended_prob, self.intended_action_prob])}

        self.living_reward = -10
        self.gold_reward = 1000
        self.death_by_pit_reward = -1000

        # for rendering
        self.assets = None

    def get_new_env(self):
        env = [[self.regular_field, self.regular_field, self.gold_field, self.pit_field],
               [self.pit_field, self.regular_field, self.regular_field, self.regular_field],
               [self.regular_field, self.regular_field, self.regular_field, self.regular_field],
               [self.regular_field, self.regular_field, self.regular_field, self.pit_field]]
        self.observation_space_n = 0

        for i in env:
            self.observation_space_n += len(i)

        state_nr = 0
        self.state_number = np.zeros((len(env), len(env[0])), dtype=np.int)
        for i in range(len(self.state_number)):
            for j in range(len(self.state_number[i])):
                self.state_number[i][j] = state_nr
                state_nr += 1

        return env

    def reset_env(self):
        self.grid_world = self.get_new_env()
        self.agentPosXY = [self.cave_entry_x, self.cave_entry_y]
        return self.state_number[self.agentPosXY[0]][self.agentPosXY[1]]

    def random_action(self):
        return random.choice(self.action_space)

    def step(self, action):
        if action not in self.action_space:
            raise Exception("Invalid action")

        reward = self.living_reward
        info = ""
        game_won = False

        if self.grid_world[self.agentPosXY[0]][self.agentPosXY[1]] == self.regular_field:
            self.grid_world[self.agentPosXY[0]][self.agentPosXY[1]] = self.visited_field

        # stochastic action
        intended_action = action
        action = np.random.choice([0, 1, 2, 3], p=self.action_prob[intended_action])

        if action == intended_action:
            info += f"Agent moved {self.action_names[intended_action]}."
        else:
            info += f"Agent wanted {self.action_names[intended_action]}, but moved {self.action_names[action]}."

        # up
        if action == 0:
            self.agentPosXY[0] -= 1
            if self.agentPosXY[0] < 0:
                self.agentPosXY[0] = 0
        # down
        elif action == 1:
            self.agentPosXY[0] += 1
            if self.agentPosXY[0] > len(self.grid_world) - 1:
                self.agentPosXY[0] = len(self.grid_world) - 1
        # right
        elif action == 2:
            self.agentPosXY[1] += 1
            if self.agentPosXY[1] > len(self.grid_world[self.agentPosXY[0]]) - 1:
                self.agentPosXY[1] = len(self.grid_world[self.agentPosXY[0]]) - 1
        # left
        elif action == 3:
            self.agentPosXY[1] -= 1
            if self.agentPosXY[1] < 0:
                self.agentPosXY[1] = 0

        new_state = self.state_number[self.agentPosXY[0]][self.agentPosXY[1]]

        if (self.grid_world[self.agentPosXY[0]][self.agentPosXY[1]] == self.regular_field) \
                | (self.grid_world[self.agentPosXY[0]][self.agentPosXY[1]] == self.visited_field):
            done = False
        else:
            done = True
            if self.grid_world[self.agentPosXY[0]][self.agentPosXY[1]] == self.gold_field:
                reward += self.gold_reward
                info += "Found gold, you won"
                game_won = True
            elif self.grid_world[self.agentPosXY[0]][self.agentPosXY[1]] == self.pit_field:
                reward += self.death_by_pit_reward
                info += "Fell into pit, you lost"
            else:
                info += "Strange ending?"

        return new_state, reward, done, info, game_won

    def render(self, screen, text, q_values=None):
        if not self.assets:
            self.assets = load_assets()

        screen.fill(WHITE)

        for i in range(len(self.grid_world)):
            for j in range(len(self.grid_world[i])):
                if self.grid_world[i][j] == self.regular_field:
                    pygame.draw.rect(screen, BROWN, pygame.Rect(j * FIELD_SIZE_X, i * FIELD_SIZE_Y,
                                                                FIELD_SIZE_X, FIELD_SIZE_Y))
                # elif self.grid_world[i][j] == self.visited_field:
                #     pygame.draw.rect(screen, green, pygame.Rect(j * FIELD_SIZE_X, i * FIELD_SIZE_Y,
                #                                                 FIELD_SIZE_X, FIELD_SIZE_Y))
                elif self.grid_world[i][j] == self.gold_field:
                    screen.blit(self.assets['gold'], (j * FIELD_SIZE_X, i * FIELD_SIZE_Y,
                                                      FIELD_SIZE_X, FIELD_SIZE_Y))
                elif self.grid_world[i][j] == self.pit_field:
                    screen.blit(self.assets['pit'], (j * FIELD_SIZE_X, i * FIELD_SIZE_Y, FIELD_SIZE_X, FIELD_SIZE_Y))

                # if (i == self.cave_entry_x) & (j == self.cave_entry_y):
                #     screen.blit(self.assets['cave_entry'], (j * FIELD_SIZE_X, i * FIELD_SIZE_Y, FIELD_SIZE_X, FIELD_SIZE_Y))
                #

                best = None
                if q_values is not None and (self.grid_world[i][j] == self.regular_field or
                                             self.grid_world[i][j] == self.visited_field):

                    state = i*4+j
                    sorted_actions = np.argsort(q_values[state])
                    if q_values[state, sorted_actions[3]] > q_values[state, sorted_actions[2]]:  # there is a clear best action
                        best = sorted_actions[3]

                    arrows = {0: self.assets['arrow_up'],
                              1: self.assets['arrow_down'],
                              2: self.assets['arrow_right'],
                              3: self.assets['arrow_left']}
                    if best is not None:
                        screen.blit(arrows[best], (j * FIELD_SIZE_X + FIELD_SIZE_X//4,
                                                   i * FIELD_SIZE_Y + FIELD_SIZE_Y//4,
                                                   FIELD_SIZE_X//2, FIELD_SIZE_Y//2))

                # draw agent
                if (i == self.agentPosXY[0]) & (j == self.agentPosXY[1]):
                    screen.blit(self.assets['agent'],
                                (j * FIELD_SIZE_X, i * FIELD_SIZE_Y, FIELD_SIZE_X, FIELD_SIZE_Y))

                pygame.draw.rect(screen, BLUE, pygame.Rect(j * FIELD_SIZE_X, i * FIELD_SIZE_Y,
                                                           FIELD_SIZE_X, FIELD_SIZE_Y), 5)

                # draw q-values at the edges of each field
                if q_values is not None and (self.grid_world[i][j] == self.regular_field or
                                             self.grid_world[i][j] == self.visited_field):

                    positions = {0: (j * FIELD_SIZE_X + FIELD_SIZE_X//2 - 20, i * FIELD_SIZE_Y + 7),
                                 1: (j * FIELD_SIZE_X + FIELD_SIZE_X//2 - 20, (i+1) * FIELD_SIZE_Y - 25),
                                 2: ((j+1) * FIELD_SIZE_X - 40, i * FIELD_SIZE_Y + FIELD_SIZE_Y//2 - 8),
                                 3: (j * FIELD_SIZE_X + 7, i * FIELD_SIZE_Y + FIELD_SIZE_Y//2 - 8)}

                    for a in range(4):
                        color = GREEN if best is not None and a == best else BLACK
                        msg = self.assets['font'].render(f"{q_values[i*4+j, a]:04.2f}", False, color)
                        screen.blit(msg, positions[a])

        for t in range(len(text)):
            msg = self.assets['font'].render(text[t], False, BLACK)
            screen.blit(msg, (FIELD_SIZE_X * len(self.grid_world[0]) + 10, t * 25))

        pygame.display.flip()
