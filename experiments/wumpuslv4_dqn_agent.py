import torch as T
from rl_alg.dqn.dqn_agent import DQNAgent
from rl_alg.epsilon_greedy_strategy import EpsilonGreedyStrategy
from rl_alg.q_agent import QAgent
from rl_alg.dqn.dqn_network import DeepQNetwork
from wumpus_envs.wumpus_env_lv4_a import AgentState, Action, CAVE_ENTRY_X, CAVE_ENTRY_Y, Sense
from gui.manual_pygame_agent import TurningManualControl
import numpy as np


class TakeGoldWumpusBasicDQN(DQNAgent):
    def __init__(self, **kwargs):
        super().__init__(13, len(Action), gamma=0.99, lr=0.01,
                         batch_size=64, max_mem_size=50000,
                         epsilon=0.4, eps_end=0.01, eps_dec=1e-5,
                         replace_target=50, **kwargs)

        self.eye = np.eye(4, dtype=np.float32)

    def from_state_to_input_vector(self, state: AgentState):
        pos_x_v = self.eye[state.pos_x]
        pos_y_v = self.eye[state.pos_y]
        dir_v = self.eye[state.agent_direction.value]
        n_arrows = np.array(state.arrows_left, dtype=np.float32)
        # senses = np.array(state.senses, dtype=np.float32)
        # v = np.hstack((pos_x_v, pos_y_v, dir_v, gold, n_arrows, senses))
        # bump = np.array(state.senses[Sense.BUMP.value], dtype=np.float32)
        v = np.hstack((pos_x_v, pos_y_v, dir_v, n_arrows))
        return v

    def get_network(self):
        return DeepQNetwork(self.lr, n_actions=self.n_actions,
                            input_dims=self.input_dims, fc1_dims=15, fc2_dims=10)


# version that learn to find gold (in static grid) and gets out of the cave (after ~5000 episodes)
class WumpusBasicStaticWorldDQN(DQNAgent):
    def __init__(self, **kwargs):
        super().__init__(14, len(Action), gamma=0.99, lr=0.01,
                         batch_size=64, max_mem_size=50000,
                         epsilon=0.8, eps_end=0.01, eps_dec=1e-5, **kwargs)

        self.eye = np.eye(4, dtype=np.float32)
        self.first_eps_strategy = self.action_selection_strategy
        self.second_eps_strategy = EpsilonGreedyStrategy(0.5, 0.01, 1e-5)

    def from_state_to_input_vector(self, state: AgentState):
        pos_x_v = self.eye[state.pos_x]
        pos_y_v = self.eye[state.pos_y]
        dir_v = self.eye[state.agent_direction.value]
        n_arrows = np.array(state.arrows_left, dtype=np.float32)
        has_gold = np.array(state.gold_taken, dtype=np.float32)
        # senses = np.array(state.senses, dtype=np.float32)
        # v = np.hstack((pos_x_v, pos_y_v, dir_v, gold, n_arrows, senses))
        # bump = np.array(state.senses[Sense.BUMP.value], dtype=np.float32)
        v = np.hstack((pos_x_v, pos_y_v, dir_v, n_arrows, has_gold))
        return v

    def choose_action(self, observation):
        if observation.gold_taken:
            if self.action_selection_strategy != self.second_eps_strategy:
                self.action_selection_strategy = self.second_eps_strategy
                # print(f'Switching to GOLD TAKEN strategy with eps={self.action_selection_strategy.epsilon}')
        else:
            if self.action_selection_strategy != self.first_eps_strategy:
                self.action_selection_strategy = self.first_eps_strategy
                # print(f'Switching to DEFAULT strategy with eps={self.action_selection_strategy.epsilon}')
        return super().choose_action(observation)

    def get_network(self):
        return DeepQNetwork(self.lr, n_actions=self.n_actions,
                            input_dims=self.input_dims, fc1_dims=15, fc2_dims=10)

    def get_state_dict(self):
        state_dict = super().get_state_dict()
        state_dict['epsilon'] = self.first_eps_strategy.epsilon
        state_dict['eps_dec'] = self.first_eps_strategy.eps_dec
        state_dict['eps_min'] = self.first_eps_strategy.eps_min
        state_dict['epsilon2'] = self.second_eps_strategy.epsilon
        state_dict['eps_dec2'] = self.second_eps_strategy.eps_dec
        state_dict['eps_min2'] = self.second_eps_strategy.eps_min
        return state_dict

    def load(self, load_state_path):
        state_dict = super().load(load_state_path)
        self.first_eps_strategy.epsilon = state_dict['epsilon']
        self.first_eps_strategy.eps_dec = state_dict['eps_dec']
        self.first_eps_strategy.eps_min = state_dict['eps_min']
        self.second_eps_strategy.epsilon = state_dict['epsilon2']
        self.second_eps_strategy.eps_dec = state_dict['eps_dec2']
        self.second_eps_strategy.eps_min = state_dict['eps_min2']
        self.action_selection_strategy = self.first_eps_strategy
        return state_dict


class WumpusBasicDQN(WumpusBasicStaticWorldDQN):
    def __init__(self, **kwargs):
        super().__init__(input_dims=14, n_actions=len(Action), gamma=0.99, lr=0.01,
                         batch_size=64, max_mem_size=50000,
                         epsilon=0.8, eps_end=0.01, eps_dec=1e-5, **kwargs)

        self.map = self.get_empty_map()

    def get_empty_map(self):
        """
        Returns a 6-channel 7x7 relative map in the agents mind (agent thinks that he is always in the central square
        facing north), after taking actions the map rotates nad shifts accordingly
        :return:
        """
        return np.zeros((7, 7, 6), dtype=np.float32)

    def reset_map(self):
        self.map = self.get_empty_map()

    def update_maps(self, state: AgentState) -> None:
        pass


    def from_state_to_input_vector(self, state: AgentState):
        self.update_maps(state)
        pos_x_v = self.eye[state.pos_x]
        pos_y_v = self.eye[state.pos_y]
        dir_v = self.eye[state.agent_direction.value]
        n_arrows = np.array(state.arrows_left, dtype=np.float32)
        has_gold = np.array(state.gold_taken, dtype=np.float32)
        # senses = np.array(state.senses, dtype=np.float32)
        # v = np.hstack((pos_x_v, pos_y_v, dir_v, gold, n_arrows, senses))
        # bump = np.array(state.senses[Sense.BUMP.value], dtype=np.float32)
        v = np.hstack((pos_x_v, pos_y_v, dir_v, n_arrows, has_gold))
        return v

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


# version that learn to find gold (in static grid), but does not kill Wumpus due to lack of n_arrows left information
# (after killing it (which yields positive reward), next attempts (unrecognizable by agent) gives only negative response
class VeryBasicWumpusQAgentTakeGold(QAgent):
    def __init__(self, **kwargs):
        super().__init__(4*4*4, len(Action), initial_q_value=0,
                         epsilon=0.4, eps_end=0.01, eps_dec=1e-5, **kwargs)
        self.manual_control = TurningManualControl()

    def from_state_to_idx(self, state: AgentState):
        return state.pos_x * 16 + state.pos_y * 4 + state.agent_direction.value


# version that learn to find gold (in static grid), and kills Wumpus
class BasicWumpusQAgentTakeGold(QAgent):
    def __init__(self, **kwargs):
        super().__init__(4*4*4*2, len(Action), initial_q_value=0,
                         epsilon=0.4, eps_end=0.01, eps_dec=1e-5, **kwargs)
        self.manual_control = TurningManualControl()

    def from_state_to_idx(self, state: AgentState):
        return state.arrows_left * 64 + state.pos_x * 16 + state.pos_y * 4 + state.agent_direction.value


# version that learn to find gold (in static grid), kills Wumpus and gets out of the cave
class BasicWumpusLv3QAgent(QAgent):
    def __init__(self, **kwargs):
        super().__init__(4*4*4*2*2, len(Action), initial_q_value=0,
                         epsilon=0.4, eps_end=0.01, eps_dec=1e-5, **kwargs)
        self.manual_control = TurningManualControl()

    def from_state_to_idx(self, state: AgentState):
        return state.gold_taken * 128 + state.arrows_left * 64 + state.pos_x * 16 + state.pos_y * 4 + state.agent_direction.value


class BasicWumpusQAgent(QAgent):
    def __init__(self, **kwargs):
        super().__init__(4*4*4*2*2*32, len(Action), initial_q_value=0,
                         epsilon=0.4, eps_end=0.01, eps_dec=1e-5, **kwargs)
        self.manual_control = TurningManualControl()

    def from_state_to_idx(self, state: AgentState):
        a = state.gold_taken * 128 + state.arrows_left * 64 + state.pos_x * 16 + state.pos_y * 4 + state.agent_direction.value
        senses = np.array(state.senses, dtype=np.float32)
        b = int(senses.dot(1 << np.arange(senses.size)[::-1]))
        return (a << 5) + b
