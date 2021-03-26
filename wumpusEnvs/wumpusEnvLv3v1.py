import random


# 4x4 grid world
# agent can take actions (6): move forward, turn left, turn right, take gold, shoot, climb out of the cave
# states (32): stench (2) X breeze (2) X glitter (2) X bump (2) X scream (2)
# bad idea
class WumpusWorldLv3v1:
    def __init__(self):
        # move forward, turn left, turn right, take gold, shoot, climb out of the cave
        self.action_space = [0, 1, 2, 3, 4, 5]
        self.action_space_n = len(self.action_space)

        self.wumpus_field = "W"
        self.pit_field = "P"
        self.gold_field = "G"
        self.regular_field = "F"
        self.visited_field = "V"
        self.agent_field_turned_up = u'\u2191'
        self.agent_field_turned_right = u'\u2192'
        self.agent_field_turned_down = u'\u2193'
        self.agent_field_turned_left = u'\u2190'
        # stench, breeze, glitter, bump, scream
        self.sensed_data = [0, 0, 0, 0, 0]

        # 0 - up, 1 - right, 2 - down, 3 - left
        self.agent_direction = 1
        # 1 - has item, 0 - doesn't have item
        self.arrow = 1
        self.gold = 0

        self.cave_entry_x = 3
        self.cave_entry_y = 0
        self.wumpus_pos_x = 1
        self.wumpus_pos_y = 0
        self.gold_pos_x = 0
        self.gold_pos_y = 3

        self.stench_string = "Stench "
        self.breeze_string = "Breeze "
        self.glitter_string = "Glitter "
        self.bump_string = "Bump "
        self.scream_string = "Scream "

        self.dqn_observation_state_number = None
        self.observation_space_n = None
        self.grid_world = self.get_new_env()
        self.agentPosXY = [self.cave_entry_x, self.cave_entry_y]

        self.living_reward = -1
        self.arrow_reward = -10
        self.gold_reward = 1000
        self.death_by_wumpus_reward = -1000
        self.death_by_pit_reward = -1000

        self.wumpus_killed_reward = 0
        self.took_gold_reward = 0

    def get_new_env(self):
        env = [[self.regular_field, self.pit_field, self.regular_field, self.pit_field],
               [self.regular_field, self.regular_field, self.pit_field, self.regular_field],
               [self.regular_field, self.regular_field, self.regular_field, self.regular_field],
               [self.regular_field, self.regular_field, self.pit_field, self.regular_field]]
        env[self.wumpus_pos_x][self.wumpus_pos_y] = self.wumpus_field
        env[self.gold_pos_x][self.gold_pos_y] = self.gold_field

        # observation space = stench (2) X breeze (2) X glitter (2) X bump (2) X scream (2)
        self.observation_space_n = 2 ** len(self.sensed_data)

        return env

    def get_state(self):
        # nr 0-31
        self.use_senses()
        # stench, breeze, glitter, bump, scream
        return 16 * self.sensed_data[0] + 8 * self.sensed_data[1] + 4 * self.sensed_data[2] + 2 * self.sensed_data[3] \
               + self.sensed_data[4]

    def reset_env(self):
        self.grid_world = self.get_new_env()
        self.agentPosXY = [self.cave_entry_x, self.cave_entry_y]
        self.agent_direction = 1
        self.use_senses()
        self.arrow = 1
        self.gold = 0
        return self.get_state()

    def random_action(self):
        return random.choice(self.action_space)

    def step(self, action):
        if action not in self.action_space:
            raise Exception("Invalid action")

        reward = self.living_reward
        info = None
        done = False
        game_won = False
        # stench, breeze, glitter, bump, scream
        self.sensed_data = [0, 0, 0, 0, 0]

        if self.grid_world[self.agentPosXY[0]][self.agentPosXY[1]] == self.regular_field:
            self.grid_world[self.agentPosXY[0]][self.agentPosXY[1]] = self.visited_field

        # move forward
        if action == 0:
            info = "Agent choose to move forward - "

            # up
            if self.agent_direction == 0:
                self.agentPosXY[0] -= 1
                if self.agentPosXY[0] < 0:
                    self.sensed_data[3] = 1
                    self.agentPosXY[0] = 0
                info += "up"
            # right
            elif self.agent_direction == 1:
                self.agentPosXY[1] += 1
                if self.agentPosXY[1] > len(self.grid_world[self.agentPosXY[0]]) - 1:
                    self.sensed_data[3] = 1
                    self.agentPosXY[1] = len(self.grid_world[self.agentPosXY[0]]) - 1
                info += "right"
            # down
            elif self.agent_direction == 2:
                self.agentPosXY[0] += 1
                if self.agentPosXY[0] > len(self.grid_world) - 1:
                    self.sensed_data[3] = 1
                    self.agentPosXY[0] = len(self.grid_world) - 1
                info += "down"
            # left
            elif self.agent_direction == 3:
                self.agentPosXY[1] -= 1
                if self.agentPosXY[1] < 0:
                    self.sensed_data[3] = 1
                    self.agentPosXY[1] = 0
                info += "left"
            else:
                raise Exception("Invalid agent direction")

            # check if met Wumpus or fell into pit
            if self.grid_world[self.agentPosXY[0]][self.agentPosXY[1]] == self.wumpus_field:
                done = True
                reward += self.death_by_wumpus_reward
                info += ". Got eaten by Wumpus, you lost"
            elif self.grid_world[self.agentPosXY[0]][self.agentPosXY[1]] == self.pit_field:
                done = True
                reward += self.death_by_pit_reward
                info += ". Fell into pit, you lost"

        # turn left
        elif action == 1:
            self.agent_direction = (self.agent_direction - 1) % 4
            info = "Agent turned left"
        # turn right
        elif action == 2:
            self.agent_direction = (self.agent_direction + 1) % 4
            info = "Agent turned right"
        # take gold
        elif action == 3:
            if self.grid_world[self.agentPosXY[0]][self.agentPosXY[1]] == self.gold_field:
                self.grid_world[self.agentPosXY[0]][self.agentPosXY[1]] = self.visited_field
                self.gold = 1
                info = "Agent took gold"
                reward += self.took_gold_reward
            else:
                info = "Agent attempted to take gold, but there wasn't any"
        # shoot
        elif action == 4:
            if self.arrow == 1:
                self.arrow = 0
                reward += self.arrow_reward
                info = "Arrow was used."
                # up
                if self.agent_direction == 0:
                    if (self.agentPosXY[1] == self.wumpus_pos_y) & (self.agentPosXY[0] > self.wumpus_pos_x):
                        info += " Wumpus is dead"
                        self.grid_world[self.wumpus_pos_x][self.wumpus_pos_y] = self.regular_field
                        reward += self.wumpus_killed_reward
                        self.sensed_data[4] = 1
                # right
                elif self.agent_direction == 1:
                    if (self.agentPosXY[0] == self.wumpus_pos_x) & (self.agentPosXY[1] < self.wumpus_pos_y):
                        info += " Wumpus is dead"
                        self.grid_world[self.wumpus_pos_x][self.wumpus_pos_y] = self.regular_field
                        reward += self.wumpus_killed_reward
                        self.sensed_data[4] = 1
                # down
                elif self.agent_direction == 2:
                    if (self.agentPosXY[1] == self.wumpus_pos_y) & (self.agentPosXY[0] < self.wumpus_pos_x):
                        info += " Wumpus is dead"
                        self.grid_world[self.wumpus_pos_x][self.wumpus_pos_y] = self.regular_field
                        reward += self.wumpus_killed_reward
                        self.sensed_data[4] = 1
                # left
                elif self.agent_direction == 3:
                    if (self.agentPosXY[0] == self.wumpus_pos_x) & (self.agentPosXY[1] > self.wumpus_pos_y):
                        info += " Wumpus is dead"
                        self.grid_world[self.wumpus_pos_x][self.wumpus_pos_y] = self.regular_field
                        reward += self.wumpus_killed_reward
                        self.sensed_data[4] = 1
            else:
                info = "Arrow no available, nothing happened"
        # climb out of the cave
        elif action == 5:
            if (self.agentPosXY[0] == self.cave_entry_x) & (self.agentPosXY[1] == self.cave_entry_y) & (self.gold == 1):
                reward += 1000
                done = True
                info = "You left cave with gold, victory"
                game_won = True
            else:
                info = "Can't leave yet"
        else:
            raise Exception("Invalid action")

        new_state = self.get_state()
        return new_state, reward, done, info, game_won

    def use_senses(self):
        # stench, breeze, glitter, bump, scream
        self.sensed_data = [0, 0, 0, 0, 0]

        if self.agentPosXY[0] + 1 <= len(self.grid_world) - 1:
            if self.grid_world[self.agentPosXY[0] + 1][self.agentPosXY[1]] == self.wumpus_field:
                self.sensed_data[0] = 1
            if self.grid_world[self.agentPosXY[0] + 1][self.agentPosXY[1]] == self.pit_field:
                self.sensed_data[1] = 1

        if self.agentPosXY[0] - 1 >= 0:
            if self.grid_world[self.agentPosXY[0] - 1][self.agentPosXY[1]] == self.wumpus_field:
                self.sensed_data[0] = 1
            if self.grid_world[self.agentPosXY[0] - 1][self.agentPosXY[1]] == self.pit_field:
                self.sensed_data[1] = 1

        if self.agentPosXY[1] + 1 <= len(self.grid_world[self.agentPosXY[0]]) - 1:
            if self.grid_world[self.agentPosXY[0]][self.agentPosXY[1] + 1] == self.wumpus_field:
                self.sensed_data[0] = 1
            if self.grid_world[self.agentPosXY[0]][self.agentPosXY[1] + 1] == self.pit_field:
                self.sensed_data[1] = 1

        if self.agentPosXY[1] - 1 >= 0:
            if self.grid_world[self.agentPosXY[0]][self.agentPosXY[1] - 1] == self.wumpus_field:
                self.sensed_data[0] = 1
            if self.grid_world[self.agentPosXY[0]][self.agentPosXY[1] - 1] == self.pit_field:
                self.sensed_data[1] = 1

        if self.grid_world[self.agentPosXY[0]][self.agentPosXY[1]] == self.gold_field:
            self.sensed_data[2] = 1

    def get_sensed_string(self):
        sensed = ""
        # stench, breeze, glitter, bump, scream
        sensed += self.stench_string if self.sensed_data[0] == 1 else ""
        sensed += self.breeze_string if self.sensed_data[1] == 1 else ""
        sensed += self.glitter_string if self.sensed_data[2] == 1 else ""
        sensed += self.bump_string if self.sensed_data[3] == 1 else ""
        sensed += self.scream_string if self.sensed_data[4] == 1 else ""

        sensed += "nothing" if sensed == "" else ""

        return sensed

    def render_env(self):
        for i in range(len(self.grid_world)):
            for j in range(len(self.grid_world[i])):
                if (i == self.agentPosXY[0]) & (j == self.agentPosXY[1]):
                    if self.agent_direction == 0:
                        print(self.agent_field_turned_up, end=" ")
                    elif self.agent_direction == 1:
                        print(self.agent_field_turned_right, end=" ")
                    elif self.agent_direction == 2:
                        print(self.agent_field_turned_down, end=" ")
                    elif self.agent_direction == 3:
                        print(self.agent_field_turned_left, end=" ")
                else:
                    print(self.grid_world[i][j], end=" ")
            print()

        print(f"The agent senses: {self.get_sensed_string()}")
