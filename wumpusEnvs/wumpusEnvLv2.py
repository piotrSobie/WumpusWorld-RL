import random


# 4x4 grid world
# agent can take actions (6): move forward, turn left, turn right, take gold, shoot, climb out of the cave
# states (256): position on the grid (16) X agent's direction (4) X has gold (2) X has arrow (2)
class WumpusWorldLv2:
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

        self.stench_string = "Stench"
        self.breeze_string = "Breeze"
        self.glitter_string = "Glitter"

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
        self.observation_space_n = 0
        env[self.wumpus_pos_x][self.wumpus_pos_y] = self.wumpus_field
        env[self.gold_pos_x][self.gold_pos_y] = self.gold_field

        # observation space = position on the grid (16) X agent's direction (4) X has gold (2) X has arrow (2)
        for i in env:
            self.observation_space_n += len(i)
        self.observation_space_n = self.observation_space_n * 4 * 2 * 2

        return env

    def get_state(self):
        # nr 0-255
        return 4 * self.agentPosXY[1] + self.agentPosXY[0] + 16 * (4 * self.agent_direction + 2 * self.arrow + self.gold)

    def reset_env(self):
        self.grid_world = self.get_new_env()
        self.agentPosXY = [self.cave_entry_x, self.cave_entry_y]
        self.agent_direction = 1
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

        if self.grid_world[self.agentPosXY[0]][self.agentPosXY[1]] == self.regular_field:
            self.grid_world[self.agentPosXY[0]][self.agentPosXY[1]] = self.visited_field

        # move forward
        if action == 0:
            info = "Agent choose to move forward - "

            # up
            if self.agent_direction == 0:
                self.agentPosXY[0] -= 1
                if self.agentPosXY[0] < 0:
                    self.agentPosXY[0] = 0
                info += "up"
            # right
            elif self.agent_direction == 1:
                self.agentPosXY[1] += 1
                if self.agentPosXY[1] > len(self.grid_world[self.agentPosXY[0]]) - 1:
                    self.agentPosXY[1] = len(self.grid_world[self.agentPosXY[0]]) - 1
                info += "right"
                # down
            elif self.agent_direction == 2:
                self.agentPosXY[0] += 1
                if self.agentPosXY[0] > len(self.grid_world) - 1:
                    self.agentPosXY[0] = len(self.grid_world) - 1
                info += "down"
                # left
            elif self.agent_direction == 3:
                self.agentPosXY[1] -= 1
                if self.agentPosXY[1] < 0:
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
                # right
                elif self.agent_direction == 1:
                    if (self.agentPosXY[0] == self.wumpus_pos_x) & (self.agentPosXY[1] < self.wumpus_pos_y):
                        info += " Wumpus is dead"
                        self.grid_world[self.wumpus_pos_x][self.wumpus_pos_y] = self.regular_field
                        reward += self.wumpus_killed_reward
                # down
                elif self.agent_direction == 2:
                    if (self.agentPosXY[1] == self.wumpus_pos_y) & (self.agentPosXY[0] < self.wumpus_pos_x):
                        info += " Wumpus is dead"
                        self.grid_world[self.wumpus_pos_x][self.wumpus_pos_y] = self.regular_field
                        reward += self.wumpus_killed_reward
                # left
                elif self.agent_direction == 3:
                    if (self.agentPosXY[0] == self.wumpus_pos_x) & (self.agentPosXY[1] > self.wumpus_pos_y):
                        info += " Wumpus is dead"
                        self.grid_world[self.wumpus_pos_x][self.wumpus_pos_y] = self.regular_field
                        reward += self.wumpus_killed_reward
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
        sensed = []

        if self.agentPosXY[0] + 1 <= len(self.grid_world) - 1:
            if self.grid_world[self.agentPosXY[0] + 1][self.agentPosXY[1]] == self.wumpus_field:
                if self.stench_string not in sensed:
                    sensed.append(self.stench_string)
            if self.grid_world[self.agentPosXY[0] + 1][self.agentPosXY[1]] == self.pit_field:
                if self.breeze_string not in sensed:
                    sensed.append(self.breeze_string)

        if self.agentPosXY[0] - 1 >= 0:
            if self.grid_world[self.agentPosXY[0] - 1][self.agentPosXY[1]] == self.wumpus_field:
                if self.stench_string not in sensed:
                    sensed.append(self.stench_string)
            if self.grid_world[self.agentPosXY[0] - 1][self.agentPosXY[1]] == self.pit_field:
                if self.breeze_string not in sensed:
                    sensed.append(self.breeze_string)

        if self.agentPosXY[1] + 1 <= len(self.grid_world[self.agentPosXY[0]]) - 1:
            if self.grid_world[self.agentPosXY[0]][self.agentPosXY[1] + 1] == self.wumpus_field:
                if self.stench_string not in sensed:
                    sensed.append(self.stench_string)
            if self.grid_world[self.agentPosXY[0]][self.agentPosXY[1] + 1] == self.pit_field:
                if self.breeze_string not in sensed:
                    sensed.append(self.breeze_string)

        if self.agentPosXY[1] - 1 >= 0:
            if self.grid_world[self.agentPosXY[0]][self.agentPosXY[1] - 1] == self.wumpus_field:
                if self.stench_string not in sensed:
                    sensed.append(self.stench_string)
            if self.grid_world[self.agentPosXY[0]][self.agentPosXY[1] - 1] == self.pit_field:
                if self.breeze_string not in sensed:
                    sensed.append(self.breeze_string)

        if self.grid_world[self.agentPosXY[0]][self.agentPosXY[1]] == self.gold_field:
            sensed.append(self.glitter_string)

        return sensed

    def get_sensed_string(self):
        sesnsed_danger = self.use_senses()
        sensed = ""
        if not sesnsed_danger:
            sensed += "nothing"
        else:
            for i in sesnsed_danger:
                sensed += i + " "

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

        print(f"The agent senses: ", end="")
        sesnsed_danger = self.use_senses()
        if not sesnsed_danger:
            print("nothing")
        else:
            for i in sesnsed_danger:
                print(i, end=" ")
            print("")
