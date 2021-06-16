from enum import Enum

from rl_alg.dqn.dqn_agent import DQNAgent
from rl_alg.epsilon_greedy_strategy import EpsilonGreedyStrategy
from rl_alg.q_agent import QAgent
from rl_alg.dqn.dqn_network import DeepQNetwork, SimpleCNNQNetwork
from envs.wumpus_env import AgentState, Action, CAVE_ENTRY_X, CAVE_ENTRY_Y, Sense, Direction
from gui.manual_pygame_agent import TurningManualControl
import numpy as np


class TakeGoldWumpusBasicDQN(DQNAgent):
    def __init__(self, **kwargs):
        super().__init__(13, len(Action), gamma=0.99, lr=0.01,
                         batch_size=64, max_mem_size=50000,
                         epsilon=0.4, eps_end=0.01, eps_dec=1e-5,
                         replace_target=50, **kwargs)

        self.eye = np.eye(4, dtype=np.float32)

    def from_state_to_net_input(self, state: AgentState):
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
    def __init__(self, input_dims=14, n_actions=len(Action), **kwargs):
        super().__init__(input_dims=input_dims, n_actions=n_actions, **kwargs)

        self.eye = np.eye(4, dtype=np.float32)
        self.first_eps_strategy = self.action_selection_strategy
        self.second_eps_strategy = EpsilonGreedyStrategy(0.5, 0.01, 1e-5)

    def from_state_to_net_input(self, state: AgentState):
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
        self.maybe_switch_strategy(observation)
        return super().choose_action(observation)

    def maybe_switch_strategy(self, observation):
        if observation[-1]:     # has_gold
            if self.action_selection_strategy != self.second_eps_strategy:
                self.action_selection_strategy = self.second_eps_strategy
                # print(f'Switching to GOLD TAKEN strategy with eps={self.action_selection_strategy.epsilon}')
        else:
            if self.action_selection_strategy != self.first_eps_strategy:
                self.action_selection_strategy = self.first_eps_strategy
                # print(f'Switching to DEFAULT strategy with eps={self.action_selection_strategy.epsilon}')

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


class MapChannel(Enum):
    BREEZE = 0
    STENCH = 1
    GLITTER = 2
    BUMP = 3
    VISITED = 4
    INITIAL_POSITION = 5
    SCREAM = 6
    HAS_GOLD = 7
    N_ARROWS = 8


class FullSenseCentralizedMapDNNAgent(WumpusBasicStaticWorldDQN):
    # def __init__(self, input_dims=(len(MapChannel), 7, 7), n_actions=len(Action), **kwargs):
    def __init__(self, input_dims=6*7*7+3, n_actions=len(Action), map_dims=(6, 7, 7),
                 n_arrows=1, batch_size=64, max_mem_size=100000, **kwargs):
        super().__init__(input_dims=input_dims, n_actions=n_actions, gamma=0.99, lr=0.001,
                         batch_size=batch_size, max_mem_size=max_mem_size,
                         epsilon=1.0, eps_end=0.01, eps_dec=1e-5, **kwargs)
        self.name = 'FullSenseCentralizedMapDNNAgent'
        self.input_dims = input_dims
        self.map_dims = map_dims
        self.initial_n_arrows = n_arrows
        self.last_pos_x, self.last_pos_y, self.last_direction = self.get_initial_pose()
        self.has_gold, self.arrows_left, self.was_scream = self.get_initial_extra()
        self.map = self.get_empty_map()
        self.manual_control = TurningManualControl()

    def reset_for_new_episode(self):
        self.reset_map()
        self.reset_pose()
        self.reset_extra()

    def get_initial_pose(self):
        return CAVE_ENTRY_X, CAVE_ENTRY_Y, Direction.UP

    def get_initial_extra(self):
        return 0, self.initial_n_arrows, 0

    def get_empty_map(self):
        """
        Returns 8-channel 7x7 relative map in the agents mind (agent thinks that he is always in the central square
        facing north), after taking actions the map rotates nad shifts accordingly
        :return:
        """
        emap = np.zeros(self.map_dims, dtype=np.float32)
        emap[MapChannel.INITIAL_POSITION.value, 3, 3] = 1
        return emap

    def reset_map(self):
        self.map = self.get_empty_map()

    def reset_pose(self):
        self.last_pos_x, self.last_pos_y, self.last_direction = self.get_initial_pose()

    def reset_extra(self):
        self.has_gold, self.arrows_left, self.was_scream = self.get_initial_extra()

    def update_map_pose(self, state: AgentState) -> None:
        if (state.pos_x != self.last_pos_x) or (state.pos_y != self.last_pos_y):       # moved up
            self.map = np.hstack((np.zeros((self.map.shape[0], 1, self.map.shape[2]), dtype=np.float32),
                                  self.map[:, :-1, :]))
        elif state.agent_direction != self.last_direction:      # turning
            dir_diff = state.agent_direction.value - self.last_direction.value
            if (dir_diff == 1) or (dir_diff == -3):             # turn left
                self.map = np.rot90(self.map, 3, axes=(1, 2))
            else:                                               # turn right
                self.map = np.rot90(self.map, 1, axes=(1, 2))

    def update_pose(self, state: AgentState) -> None:
        self.last_pos_x = state.pos_x
        self.last_pos_y = state.pos_y
        self.last_direction = state.agent_direction

    def update_maps(self, state: AgentState) -> None:
        if self.map[MapChannel.BUMP.value, 3, 4] == 0 and state.senses[Sense.BUMP.value]:
            self.map[MapChannel.BUMP.value, 3, 4] = 1

        self.update_map_pose(state)

        # insert new sensations in central place of the map, except bump which places info in front
        self.map[MapChannel.BREEZE.value, 3, 3] = state.senses[Sense.BREEZE.value]
        self.map[MapChannel.STENCH.value, 3, 3] = state.senses[Sense.STENCH.value]
        self.map[MapChannel.GLITTER.value, 3, 3] = state.senses[Sense.GLITTER.value]
        self.map[MapChannel.VISITED.value, 3, 3] = 1

        self.update_pose(state)

    def from_state_to_net_input(self, state: AgentState):
        self.update_maps(state)
        # for n, m in enumerate(self.map):
        #     print(f"Map about {MapChannel(n).name}")
        #     print(m)
        # return self.map.copy()
        self.has_gold = state.gold_taken
        self.arrows_left = state.arrows_left
        self.was_scream = self.was_scream or state.senses[Sense.SCREAM.value]

        flat_map_centre = self.map.flatten()
        v = np.hstack((self.has_gold, self.arrows_left, self.was_scream, flat_map_centre)).astype(np.float32)
        return v

    def maybe_switch_strategy(self, observation):
        # if observation[MapChannel.HAS_GOLD.value, 3, 3] == 1:     # has_gold
        if observation[0] == 1:     # has_gold
            if self.action_selection_strategy != self.second_eps_strategy:
                self.action_selection_strategy = self.second_eps_strategy
                # print(f'Switching to GOLD TAKEN strategy with eps={self.action_selection_strategy.epsilon}')
        else:
            if self.action_selection_strategy != self.first_eps_strategy:
                self.action_selection_strategy = self.first_eps_strategy
                # print(f'Switching to DEFAULT strategy with eps={self.action_selection_strategy.epsilon}')

    def choose_action(self, observation):
        self.maybe_switch_strategy(observation)
        return DQNAgent.choose_action(self, observation[None, ...])

    def get_network(self):
        # return DeepQNetwork(self.lr, self.input_dims, fc1_dims=50, fc2_dims=40, n_actions=self.n_actions)
        return DeepQNetwork(self.lr, self.input_dims, fc1_dims=128, fc2_dims=64, n_actions=self.n_actions)
        # return SimpleCNNQNetwork(self.input_dims, self.n_actions, self.lr)


class FullSenseCentralizedMapCNNAgent(FullSenseCentralizedMapDNNAgent):
    def __init__(self, input_dims=(9, 7, 7), n_actions=len(Action), map_dims=(9, 7, 7), **kwargs):
        super().__init__(input_dims=input_dims, n_actions=n_actions, map_dims=map_dims, **kwargs)
        self.name = 'FullSenseCentralizedMapCNNAgent'

    def get_empty_map(self):
        """
        Returns 8-channel 7x7 relative map in the agents mind (agent thinks that he is always in the central square
        facing north), after taking actions the map rotates nad shifts accordingly
        :return:
        """
        emap = np.zeros(self.map_dims, dtype=np.float32)
        emap[MapChannel.INITIAL_POSITION.value, 3, 3] = 1
        emap[MapChannel.N_ARROWS.value, :, :] = self.initial_n_arrows
        return emap

    def update_maps(self, state: AgentState) -> None:
        super(FullSenseCentralizedMapCNNAgent, self).update_maps(state)
        self.map[MapChannel.HAS_GOLD.value, :, :] = state.gold_taken
        self.map[MapChannel.N_ARROWS.value, :, :] = state.arrows_left
        self.map[MapChannel.SCREAM.value, :, :] = self.map[MapChannel.SCREAM.value, 3, 3] or state.senses[Sense.SCREAM.value]

    def from_state_to_net_input(self, state: AgentState):
        self.update_maps(state)
        # for n, m in enumerate(self.map):
        #     print(f"Map about {MapChannel(n).name}")
        #     print(m)
        # return self.map.copy()
        # self.has_gold = state.gold_taken
        # self.arrows_left = state.arrows_left
        # self.was_scream = self.was_scream or state.senses[Sense.SCREAM.value]

        return self.map.copy()

    def maybe_switch_strategy(self, observation):
        if observation[MapChannel.HAS_GOLD.value, 3, 3] == 1:     # has_gold
            if self.action_selection_strategy != self.second_eps_strategy:
                self.action_selection_strategy = self.second_eps_strategy
                # print(f'Switching to GOLD TAKEN strategy with eps={self.action_selection_strategy.epsilon}')
        else:
            if self.action_selection_strategy != self.first_eps_strategy:
                self.action_selection_strategy = self.first_eps_strategy
                # print(f'Switching to DEFAULT strategy with eps={self.action_selection_strategy.epsilon}')

    def get_network(self):
        return SimpleCNNQNetwork(self.input_dims, self.n_actions, self.lr)


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
        return state.gold_taken * 128 + state.arrows_left * 64 + state.pos_x * 16 + state.pos_y * 4 + \
               state.agent_direction.value


class BasicWumpusQAgent(QAgent):
    def __init__(self, **kwargs):
        super().__init__(4*4*4*2*2*32, len(Action), initial_q_value=0,
                         epsilon=0.4, eps_end=0.01, eps_dec=1e-5, **kwargs)
        self.manual_control = TurningManualControl()

    def from_state_to_idx(self, state: AgentState):
        a = state.gold_taken * 128 + state.arrows_left * 64 + state.pos_x * 16 + state.pos_y * 4 + \
            state.agent_direction.value
        senses = np.array(state.senses, dtype=np.float32)
        b = int(senses.dot(1 << np.arange(senses.size)[::-1]))
        return (a << 5) + b
