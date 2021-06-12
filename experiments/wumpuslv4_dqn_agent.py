from rl_alg.dqn.dqn_agent import DQNAgent
from rl_alg.dqn.dqn_network import DeepQNetwork
from wumpus_envs.wumpus_env_lv4_a import AgentState, Action, CAVE_ENTRY_X, CAVE_ENTRY_Y, Sense
import numpy as np


class WumpusBasicDQN(DQNAgent):
    def __init__(self, **kwargs):
        super().__init__(19, len(Action), gamma=0.99, lr=0.1,
                         batch_size=64, max_mem_size=20000,
                         epsilon=0.2, eps_end=0.01, eps_dec=1e-6,
                         replace_target=100, **kwargs)

        self.eye = np.eye(4, dtype=np.float32)

        # # 1 - has item, 0 - doesn't have item
        # self.arrow = 1
        # self.gold = 0
        # # whether or not the last move forward resulted in a bump, 0 - no, 1 - yes
        # self.bump = 0
        # # whether or not there has been a scream, 0 - no, 1 - yes
        # self.scream = 0
        # # whether or not the room was already visited, 0 - no, 1 - yes
        # self.visited_rooms = [[0, 0, 0, 0],
        #                       [0, 0, 0, 0],
        #                       [0, 0, 0, 0],
        #                       [0, 0, 0, 0]]
        # # whether or not the room has stench, 0 - no, 1 - yes
        # self.stench_rooms = [[0, 0, 0, 0],
        #                      [0, 0, 0, 0],
        #                      [0, 0, 0, 0],
        #                      [0, 0, 0, 0]]
        # # whether or not the room has breeze, 0 - no, 1 - yes
        # self.breeze_rooms = [[0, 0, 0, 0],
        #                      [0, 0, 0, 0],
        #                      [0, 0, 0, 0],
        #                      [0, 0, 0, 0]]
        # # whether or not the room has glitter, 0 - no, 1 - yes
        # self.glitter_rooms = [[0, 0, 0, 0],
        #                       [0, 0, 0, 0],
        #                       [0, 0, 0, 0],
        #                       [0, 0, 0, 0]]
        # self.agent_pos = [[0, 0, 0, 0],
        #                   [0, 0, 0, 0],
        #                   [0, 0, 0, 0],
        #                   [0, 0, 0, 0]]
        #
        # self.visited_rooms[CAVE_ENTRY_X][CAVE_ENTRY_Y] = 1
        # self.agent_pos[CAVE_ENTRY_X][CAVE_ENTRY_Y] = 1

    def from_state_to_input_vector(self, state: AgentState):
        is_starting_position = (state.pos_x == CAVE_ENTRY_X) and (state.pos_y == CAVE_ENTRY_Y)
        is_starting_position = np.array(is_starting_position, dtype=np.float32)
        pos_x_v = self.eye[state.pos_x]
        pos_y_v = self.eye[state.pos_y]
        dir_v = self.eye[state.agent_direction.value]
        gold = np.array(state.gold_taken, dtype=np.float32)
        n_arrows = np.array(state.arrows_left, dtype=np.float32)
        senses = np.array(state.senses, dtype=np.float32)
        # v = np.hstack((pos_x_v, pos_y_v, dir_v, gold, n_arrows, senses))
        glitter = np.array([state.senses[Sense.GLITTER.value]]*5, dtype=np.float32)
        v = np.hstack((is_starting_position, pos_x_v, pos_y_v, dir_v, gold, glitter))
        return v

    def get_network(self):
        return DeepQNetwork(self.lr, n_actions=self.n_actions,
                            input_dims=self.input_dims, fc1_dims=30, fc2_dims=20)



