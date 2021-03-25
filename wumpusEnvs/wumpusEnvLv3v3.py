import random


# 4x4 randomized grid world
# agent can take actions (6): move forward, turn left, turn right, take gold, shoot, climb out of the cave
# agent can detect stench, breeze, bump (if he is next to end of map) nad their direction, glitter
# states (32768): agent_x (4) X agent_y (4) X agent's direction (4) X has gold (2) X has arrow (2)
#                 X stench (5) X breeze (5) X bump (5) X glitter (2)
class WumpusWorldLv3v3:
    def __init__(self, number_of_wumpuses_=1, number_of_pits_=3, number_of_golds_=1):
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
        # stench (5), breeze (5), bump (5), glitter (2)
        # for stench, breeze, bump: 0 - up, 1 - right, 2 - down, 3 - left, 4 - nothing
        # for glitter: 1- yes, 0 - no
        self.sensed_data = [4, 4, 4, 0]

        # 0 - up, 1 - right, 2 - down, 3 - left
        self.agent_direction = 1
        # 1 - has item, 0 - doesn't have item
        self.arrow = 1
        self.gold = 0

        self.cave_entry_x = 3
        self.cave_entry_y = 0
        self.wumpus_pos_x = None
        self.wumpus_pos_y = None
        self.gold_pos_x = None
        self.gold_pos_y = None

        self.stench_string = "stench "
        self.breeze_string = "breeze "
        self.glitter_string = "glitter "
        self.bump_string = "bump "
        self.scream_string = "scream "

        self.number_of_wumpuses = number_of_wumpuses_
        self.number_of_pits = number_of_pits_
        self.number_of_golds = number_of_golds_

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

    def get_random_env(self):
        size_x = 4
        size_y = 4
        env = []
        for _ in range(size_y):
            env.append([self.regular_field] * size_x)
        env[self.cave_entry_x][self.cave_entry_y] = ""

        wumpus_nr = self.number_of_wumpuses
        gold_nr = self.number_of_golds
        pit_nr = self.number_of_pits
        while (wumpus_nr > 0) | (gold_nr > 0) | (pit_nr > 0):
            random_x = random.randint(0, size_x - 1)
            random_y = random.randint(0, size_y - 1)
            # wumpus
            if wumpus_nr > 0:
                if env[random_x][random_y] == self.regular_field:
                    wumpus_nr -= 1
                    env[random_x][random_y] = self.wumpus_field
                    self.wumpus_pos_x = random_x
                    self.wumpus_pos_y = random_y
            # gold
            elif gold_nr > 0:
                if env[random_x][random_y] == self.regular_field:
                    gold_nr -= 1
                    env[random_x][random_y] = self.gold_field
                    self.gold_pos_x = random_x
                    self.gold_pos_y = random_y
            # pit
            elif pit_nr > 0:
                if env[random_x][random_y] == self.regular_field:
                    pit_nr -= 1
                    env[random_x][random_y] = self.pit_field

        env[self.cave_entry_x][self.cave_entry_y] = self.regular_field

        return env

    def get_new_env(self):
        env = self.get_random_env()

        # agent_x (4) X agent_y (4) X agent's direction (4) X has gold (2) X has arrow (2)
        # X stench (5) X breeze (5) X bump (5) X glitter (2)
        self.observation_space_n = 4 * 4 * 4 * 2 * 2 * 5 * 5 * 5 * 2

        return env

    def get_state(self):
        # nr 0-63999
        self.use_senses()
        # states (64000): agent_x (4) X agent_y (4) X agent's direction (4) X has gold (2) X has arrow (2)
        #                 X stench (5) X breeze (5) X bump (5) X glitter (2)
        # self.sensed_data = stench (5), breeze (5), bump (5), glitter (2)
        return self.agentPosXY[0] + 4 * self.agentPosXY[1] + (4 * 4) * self.agent_direction + (4 * 4 * 4) * self.gold + (4 * 4 * 4 * 2) * self.arrow + (4 * 4 * 4 * 2 * 2) * self.sensed_data[0] + (4 * 4 * 4 * 2 * 2 * 5) * self.sensed_data[1] + (4 * 4 * 4 * 2 * 2 * 5 * 5) * self.sensed_data[2] + (4 * 4 * 4 * 2 * 2 * 5 * 5 * 5) * self.sensed_data[3]

    def reset_env(self):
        self.grid_world = self.get_new_env()
        self.agentPosXY = [self.cave_entry_x, self.cave_entry_y]
        self.agent_direction = 1
        self.arrow = 1
        self.gold = 0
        self.sensed_data = [4, 4, 4, 0]
        self.use_senses()
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
        # stench (5), breeze (5), bump (5), glitter (2)
        # for stench, breeze, bump: 0 - up, 1 - right, 2 - down, 3 - left, 4 - nothing
        # for glitter: 1- yes, 0 - no
        self.sensed_data = [4, 4, 4, 0]

        if self.grid_world[self.agentPosXY[0]][self.agentPosXY[1]] == self.regular_field:
            self.grid_world[self.agentPosXY[0]][self.agentPosXY[1]] = self.visited_field

        # move forward
        if action == 0:
            info = "Agent choose to move forward - "

            # up
            if self.agent_direction == 0:
                self.agentPosXY[0] -= 1
                if self.agentPosXY[0] < 0:
                    self.sensed_data[2] = 0
                    self.agentPosXY[0] = 0
                info += "up"
            # right
            elif self.agent_direction == 1:
                self.agentPosXY[1] += 1
                if self.agentPosXY[1] > len(self.grid_world[self.agentPosXY[0]]) - 1:
                    self.sensed_data[2] = 1
                    self.agentPosXY[1] = len(self.grid_world[self.agentPosXY[0]]) - 1
                info += "right"
            # down
            elif self.agent_direction == 2:
                self.agentPosXY[0] += 1
                if self.agentPosXY[0] > len(self.grid_world) - 1:
                    self.sensed_data[2] = 2
                    self.agentPosXY[0] = len(self.grid_world) - 1
                info += "down"
            # left
            elif self.agent_direction == 3:
                self.agentPosXY[1] -= 1
                if self.agentPosXY[1] < 0:
                    self.sensed_data[2] = 3
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
                info = "Agent attempted to leave cave, but he didn't have gold or wasn't in the entry position"
        else:
            raise Exception("Invalid action")

        new_state = self.get_state()
        return new_state, reward, done, info, game_won

    def use_senses(self):
        # stench (5), breeze (5), bump (5), glitter (2)
        # for stench, breeze, bump: 0 - up, 1 - right, 2 - down, 3 - left, 4 - nothing
        # for glitter: 1- yes, 0 - no

        # down
        if self.agentPosXY[0] + 1 <= len(self.grid_world) - 1:
            if self.grid_world[self.agentPosXY[0] + 1][self.agentPosXY[1]] == self.wumpus_field:
                self.sensed_data[0] = 2
            if self.grid_world[self.agentPosXY[0] + 1][self.agentPosXY[1]] == self.pit_field:
                self.sensed_data[1] = 2

        # up
        if self.agentPosXY[0] - 1 >= 0:
            if self.grid_world[self.agentPosXY[0] - 1][self.agentPosXY[1]] == self.wumpus_field:
                self.sensed_data[0] = 0
            if self.grid_world[self.agentPosXY[0] - 1][self.agentPosXY[1]] == self.pit_field:
                self.sensed_data[1] = 0

        # right
        if self.agentPosXY[1] + 1 <= len(self.grid_world[self.agentPosXY[0]]) - 1:
            if self.grid_world[self.agentPosXY[0]][self.agentPosXY[1] + 1] == self.wumpus_field:
                self.sensed_data[0] = 1
            if self.grid_world[self.agentPosXY[0]][self.agentPosXY[1] + 1] == self.pit_field:
                self.sensed_data[1] = 1

        # left
        if self.agentPosXY[1] - 1 >= 0:
            if self.grid_world[self.agentPosXY[0]][self.agentPosXY[1] - 1] == self.wumpus_field:
                self.sensed_data[0] = 3
            if self.grid_world[self.agentPosXY[0]][self.agentPosXY[1] - 1] == self.pit_field:
                self.sensed_data[1] = 3

        if self.grid_world[self.agentPosXY[0]][self.agentPosXY[1]] == self.gold_field:
            self.sensed_data[3] = 1

    def get_sensed_string(self):
        sensed = ""
        # stench (5), breeze (5), bump (5), glitter (2)
        # for stench, breeze, bump: 0 - up, 1 - right, 2 - down, 3 - left, 4 - nothing
        # for glitter: 1- yes, 0 - no

        # stench
        if self.sensed_data[0] == 0:
            sensed += self.stench_string + " - up "
        elif self.sensed_data[0] == 1:
            sensed += self.stench_string + " - right "
        elif self.sensed_data[0] == 2:
            sensed += self.stench_string + " - down "
        elif self.sensed_data[0] == 3:
            sensed += self.stench_string + " - left "

        # breeze
        if self.sensed_data[1] == 0:
            sensed += self.breeze_string + " - up "
        elif self.sensed_data[1] == 1:
            sensed += self.breeze_string + " - right "
        elif self.sensed_data[1] == 2:
            sensed += self.breeze_string + " - down "
        elif self.sensed_data[1] == 3:
            sensed += self.breeze_string + " - left "

        # bump
        if self.sensed_data[2] == 0:
            sensed += self.bump_string + " - up "
        elif self.sensed_data[2] == 1:
            sensed += self.bump_string + " - right "
        elif self.sensed_data[2] == 2:
            sensed += self.bump_string + " - down "
        elif self.sensed_data[2] == 3:
            sensed += self.bump_string + " - left "

        # glitter
        if self.sensed_data[3] == 1:
            sensed += self.glitter_string

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
