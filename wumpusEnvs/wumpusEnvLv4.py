import random


# 4x4 randomized grid world
# agent can take actions (6): move forward, turn left, turn right, take gold, shoot, climb out of the cave
# agent can detect stench, breeze, bump (if he walks into a wall), glitter
# states = array [2^2 (direction) * 2^1 (bump) * 2^1 (scream) * 2^16 (if room visited) * 2^16 (stench) * 2^16 (breeze)
# * 2^16 (glitter) * 2^16 (where is agent) * 2^1 (has gold) * 2^1 (has arrow)]
class WumpusWorldLv4:
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

        self.cave_entry_x = 3
        self.cave_entry_y = 0
        self.wumpus_pos_x = None
        self.wumpus_pos_y = None
        self.gold_pos_x = None
        self.gold_pos_y = None

        # 0 - up, 1 - right, 2 - down, 3 - left
        self.agent_direction = 1
        # 1 - has item, 0 - doesn't have item
        self.arrow = 1
        self.gold = 0
        # whether or not the last move forward resulted in a bump, 0 - no, 1 - yes
        self.bump = 0
        # whether or not there has been a scream, 0 - no, 1 - yes
        self.scream = 0
        # whether or not the room was already visited, 0 - no, 1 - yes
        self.visited_rooms = [[0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0]]
        # whether or not the room has stench, 0 - no, 1 - yes
        self.stench_rooms = [[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]]
        # whether or not the room has breeze, 0 - no, 1 - yes
        self.breeze_rooms = [[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]]
        # whether or not the room has glitter, 0 - no, 1 - yes
        self.glitter_rooms = [[0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0]]
        # whether or not the room has glitter, 0 - no, 1 - yes
        self.agent_pos = [[0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]]

        self.visited_rooms[self.cave_entry_x][self.cave_entry_y] = 1
        self.agent_pos[self.cave_entry_x][self.cave_entry_y] = 1

        self.stench_string = "stench "
        self.breeze_string = "breeze "
        self.glitter_string = "glitter "
        self.bump_string = "bump "
        self.scream_string = "scream "

        self.number_of_wumpuses = number_of_wumpuses_
        self.number_of_pits = number_of_pits_
        self.number_of_golds = number_of_golds_

        self.observation_space_n = None
        self.dqn_observation_state_number = None
        self.grid_world = self.get_new_env()
        self.agentPosXY = [self.cave_entry_x, self.cave_entry_y]

        self.living_reward = -1
        self.arrow_reward = -10
        self.gold_reward = 1000
        self.death_by_wumpus_reward = -1000
        self.death_by_pit_reward = -1000

        self.wumpus_killed_reward = 0
        self.took_gold_reward = 0
        self.turning_reward = 0
        self.already_visited_reward = 0

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

        # 2 ^ 2(direction) * 2 ^ 1(bump) * 2 ^ 1(scream) * 2 ^ 16( if room visited) *2 ^ 16(stench) * 2 ^ 16(breeze)
        # * 2^16 (glitter) * 2^16 (where is agent) * 2^1 (has gold) * 2^1 (has arrow)
        # self.observation_space_n = 2**2 * 2**1 * 2**1 * 2**16 * 2**16 * 2**16 * 2**16 * 2**16 * 2**1 * 2**1
        self.observation_space_n = None  # not suitable for q-learn
        self.dqn_observation_state_number = 10

        return env

    def array_to_int(self, array):
        array_str = ""
        for i in range(len(array)):
            for j in range(len(array[i])):
                array_str += str(array[i][j])

        return int(array_str, 2)

    def get_state(self):
        # array:
        # [direction, bump, scream, if room visited, stench, breeze, glitter, where is agent, has gold, has arrow]
        self.use_senses()

        state = [self.agent_direction, self.bump, self.scream, self.array_to_int(self.visited_rooms),
                 self.array_to_int(self.stench_rooms), self.array_to_int(self.breeze_rooms),
                 self.array_to_int(self.glitter_rooms), self.array_to_int(self.agent_pos), self.gold, self.arrow]

        return state

    def reset_env(self):
        self.grid_world = self.get_new_env()
        self.agentPosXY = [self.cave_entry_x, self.cave_entry_y]

        self.agent_direction = 1
        self.bump = 0
        self.scream = 0
        self.gold = 0
        self.arrow = 1
        self.visited_rooms = [[0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0]]
        self.stench_rooms = [[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]]
        self.breeze_rooms = [[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]]
        self.glitter_rooms = [[0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0]]
        self.agent_pos = [[0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]]
        self.visited_rooms[self.cave_entry_x][self.cave_entry_y] = 1
        self.agent_pos[self.cave_entry_x][self.cave_entry_y] = 1

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
        self.bump = 0

        if self.visited_rooms[self.agentPosXY[0]][self.agentPosXY[1]] == 1:
            reward += self.already_visited_reward

        self.agent_pos[self.agentPosXY[0]][self.agentPosXY[1]] = 0

        if self.grid_world[self.agentPosXY[0]][self.agentPosXY[1]] == self.regular_field:
            self.grid_world[self.agentPosXY[0]][self.agentPosXY[1]] = self.visited_field

        # move forward
        if action == 0:
            info = "Agent choose to move forward - "

            # up
            if self.agent_direction == 0:
                self.agentPosXY[0] -= 1
                if self.agentPosXY[0] < 0:
                    self.bump = 1
                    self.agentPosXY[0] = 0
                info += "up"
            # right
            elif self.agent_direction == 1:
                self.agentPosXY[1] += 1
                if self.agentPosXY[1] > len(self.grid_world[self.agentPosXY[0]]) - 1:
                    self.bump = 1
                    self.agentPosXY[1] = len(self.grid_world[self.agentPosXY[0]]) - 1
                info += "right"
            # down
            elif self.agent_direction == 2:
                self.agentPosXY[0] += 1
                if self.agentPosXY[0] > len(self.grid_world) - 1:
                    self.bump = 1
                    self.agentPosXY[0] = len(self.grid_world) - 1
                info += "down"
            # left
            elif self.agent_direction == 3:
                self.agentPosXY[1] -= 1
                if self.agentPosXY[1] < 0:
                    self.bump = 1
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
            reward += self.turning_reward
        # turn right
        elif action == 2:
            self.agent_direction = (self.agent_direction + 1) % 4
            info = "Agent turned right"
            reward += self.turning_reward
        # take gold
        elif action == 3:
            if self.grid_world[self.agentPosXY[0]][self.agentPosXY[1]] == self.gold_field:
                self.grid_world[self.agentPosXY[0]][self.agentPosXY[1]] = self.visited_field
                self.gold = 1
                info = "Agent took gold"
                # finish game after taking gold
                # done = True
                # game_won = True
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
                        self.scream = 1
                        self.grid_world[self.wumpus_pos_x][self.wumpus_pos_y] = self.regular_field
                        reward += self.wumpus_killed_reward
                # right
                elif self.agent_direction == 1:
                    if (self.agentPosXY[0] == self.wumpus_pos_x) & (self.agentPosXY[1] < self.wumpus_pos_y):
                        info += " Wumpus is dead"
                        self.scream = 1
                        self.grid_world[self.wumpus_pos_x][self.wumpus_pos_y] = self.regular_field
                        reward += self.wumpus_killed_reward
                # down
                elif self.agent_direction == 2:
                    if (self.agentPosXY[1] == self.wumpus_pos_y) & (self.agentPosXY[0] < self.wumpus_pos_x):
                        info += " Wumpus is dead"
                        self.scream = 1
                        self.grid_world[self.wumpus_pos_x][self.wumpus_pos_y] = self.regular_field
                        reward += self.wumpus_killed_reward
                # left
                elif self.agent_direction == 3:
                    if (self.agentPosXY[0] == self.wumpus_pos_x) & (self.agentPosXY[1] > self.wumpus_pos_y):
                        info += " Wumpus is dead"
                        self.scream = 1
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

        self.agent_pos[self.agentPosXY[0]][self.agentPosXY[1]] = 1
        self.visited_rooms[self.agentPosXY[0]][self.agentPosXY[1]] = 1

        new_state = self.get_state()
        return new_state, reward, done, info, game_won

    def use_senses(self):
        # stench_rooms, breeze_rooms, glitter_rooms, 0 no, 1 yes

        # down
        if self.agentPosXY[0] + 1 <= len(self.grid_world) - 1:
            # stench
            if self.grid_world[self.agentPosXY[0] + 1][self.agentPosXY[1]] == self.wumpus_field:
                self.stench_rooms[self.agentPosXY[0]][self.agentPosXY[1]] = 1
            # breeze
            if self.grid_world[self.agentPosXY[0] + 1][self.agentPosXY[1]] == self.pit_field:
                self.breeze_rooms[self.agentPosXY[0]][self.agentPosXY[1]] = 1

        # up
        if self.agentPosXY[0] - 1 >= 0:
            # stench
            if self.grid_world[self.agentPosXY[0] - 1][self.agentPosXY[1]] == self.wumpus_field:
                self.stench_rooms[self.agentPosXY[0]][self.agentPosXY[1]] = 1
            # breeze
            if self.grid_world[self.agentPosXY[0] - 1][self.agentPosXY[1]] == self.pit_field:
                self.breeze_rooms[self.agentPosXY[0]][self.agentPosXY[1]] = 1

        # right
        if self.agentPosXY[1] + 1 <= len(self.grid_world[self.agentPosXY[0]]) - 1:
            # stench
            if self.grid_world[self.agentPosXY[0]][self.agentPosXY[1] + 1] == self.wumpus_field:
                self.stench_rooms[self.agentPosXY[0]][self.agentPosXY[1]] = 1
            # breeze
            if self.grid_world[self.agentPosXY[0]][self.agentPosXY[1] + 1] == self.pit_field:
                self.breeze_rooms[self.agentPosXY[0]][self.agentPosXY[1]] = 1

        # left
        if self.agentPosXY[1] - 1 >= 0:
            # stench
            if self.grid_world[self.agentPosXY[0]][self.agentPosXY[1] - 1] == self.wumpus_field:
                self.stench_rooms[self.agentPosXY[0]][self.agentPosXY[1]] = 1
            # breeze
            if self.grid_world[self.agentPosXY[0]][self.agentPosXY[1] - 1] == self.pit_field:
                self.breeze_rooms[self.agentPosXY[0]][self.agentPosXY[1]] = 1

        # glitter
        if self.grid_world[self.agentPosXY[0]][self.agentPosXY[1]] == self.gold_field:
            self.glitter_rooms[self.agentPosXY[0]][self.agentPosXY[1]] = 1
        else:
            self.glitter_rooms[self.agentPosXY[0]][self.agentPosXY[1]] = 0

        # if wumpus was killed
        if self.scream == 1:
            self.stench_rooms[self.agentPosXY[0]][self.agentPosXY[1]] = 0

    def get_sensed_string(self):
        sensed = ""
        # stench_rooms, breeze_rooms, glitter_rooms, 0 no, 1 yes
        x = self.agentPosXY[0]
        y = self.agentPosXY[1]
        sensed += self.stench_string if self.stench_rooms[x][y] == 1 else ""
        sensed += self.breeze_string if self.breeze_rooms[x][y] == 1 else ""
        sensed += self.glitter_string if self.glitter_rooms[x][y] == 1 else ""

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
