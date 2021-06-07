import random
import numpy as np


# 4x4 constant grid world
# agent can take actions (4): go up, down, right, left
# states (16): each grid represents different state
# agent doesn't make use of sensors - breeze, smell, glitter, bump, scream
class WumpusWorldLv1:
    def __init__(self):
        # go up, down, right, left
        self.action_space = [0, 1, 2, 3]
        self.action_space_n = len(self.action_space)

        self.wumpus_field = "W"
        self.pit_field = "P"
        self.gold_field = "G"
        self.regular_field = "F"
        self.visited_field = "V"
        self.agent_field = "A"

        self.stench_string = "Stench"
        self.breeze_string = "Breeze"

        self.dqn_observation_state_number = None
        self.observation_space_n = None
        self.state_number = None  # array, size the same as gridworld, it looks like: [[0, 1, 2, 3], ... ,[12,13,14,15]]
        # i use it to find agent state, different way is state = 4 * agent_pos_y + agent_pos_x, but array way works, so
        # i didn't change it
        self.grid_world = self.get_new_env()
        self.cave_entry_x = 3
        self.cave_entry_y = 0
        self.agentPosXY = [self.cave_entry_x, self.cave_entry_y]

        self.living_reward = -1
        self.arrow_reward = -10
        self.gold_reward = 1000
        self.death_by_wumpus_reward = -1000
        self.death_by_pit_reward = -1000
        self.agent_direction = None

    def get_new_env(self):
        env = [[self.regular_field, self.regular_field, self.gold_field, self.pit_field],
               [self.wumpus_field, self.regular_field, self.pit_field, self.regular_field],
               [self.regular_field, self.regular_field, self.regular_field, self.regular_field],
               [self.regular_field, self.regular_field, self.pit_field, self.regular_field]]
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

        # up
        if action == 0:
            info += "Agent moved up. "
            self.agentPosXY[0] -= 1
            if self.agentPosXY[0] < 0:
                self.agentPosXY[0] = 0
        # down
        elif action == 1:
            info += "Agent moved down. "
            self.agentPosXY[0] += 1
            if self.agentPosXY[0] > len(self.grid_world) - 1:
                self.agentPosXY[0] = len(self.grid_world) - 1
        # right
        elif action == 2:
            info += "Agent moved right. "
            self.agentPosXY[1] += 1
            if self.agentPosXY[1] > len(self.grid_world[self.agentPosXY[0]]) - 1:
                self.agentPosXY[1] = len(self.grid_world[self.agentPosXY[0]]) - 1
        # left
        elif action == 3:
            info += "Agent moved left. "
            self.agentPosXY[1] -= 1
            if self.agentPosXY[1] < 0:
                self.agentPosXY[1] = 0

        new_state = self.state_number[self.agentPosXY[0]][self.agentPosXY[1]]

        if (self.grid_world[self.agentPosXY[0]][self.agentPosXY[1]] == self.regular_field)\
                | (self.grid_world[self.agentPosXY[0]][self.agentPosXY[1]] == self.visited_field):
            done = False
        else:
            done = True
            if self.grid_world[self.agentPosXY[0]][self.agentPosXY[1]] == self.gold_field:
                reward += self.gold_reward
                info += "Found gold, you won"
                game_won = True
            elif self.grid_world[self.agentPosXY[0]][self.agentPosXY[1]] == self.wumpus_field:
                reward += self.death_by_wumpus_reward
                info += "Got eaten by Wumpus, you lost"
            elif self.grid_world[self.agentPosXY[0]][self.agentPosXY[1]] == self.pit_field:
                reward += self.death_by_pit_reward
                info += "Fell into pit, you lost"

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

        return sensed

    def get_sensed_string(self):
        return ""

    def render_env(self):
        for i in range(len(self.grid_world)):
            for j in range(len(self.grid_world[i])):
                if (i == self.agentPosXY[0]) & (j == self.agentPosXY[1]):
                    print(self.agent_field, end=" ")
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
