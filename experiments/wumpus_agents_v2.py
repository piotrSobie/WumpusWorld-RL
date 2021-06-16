from enum import Enum
from experiments.wumpuslv4_dqn_agent import FullSenseCentralizedMapDNNAgent
from rl_alg.dqn.dqn_agent import DQNAgent
from rl_alg.dqn.dqn_network import DeepQNetwork, SimpleCNNQNetwork
from envs.wumpus_env import AgentState, Action, Sense
import numpy as np


PRESENCE = 1.
ABSENCE = -1.


class MapChannel2(Enum):
    BREEZE = 0
    STENCH = 1
    GLITTER = 2
    BUMP = 3
    INITIAL_POSITION = 4
    SCREAM = 5
    HAS_GOLD = 6
    N_ARROWS = 7


class CentralizedMapDNNAgent(FullSenseCentralizedMapDNNAgent):
    def __init__(self, input_dims=5*7*7+3, n_actions=len(Action), map_dims=(5, 9, 9),
                 n_arrows=1, **kwargs):
        super().__init__(input_dims=input_dims, n_actions=n_actions, map_dims=map_dims,
                         n_arrows=n_arrows, **kwargs)

    def get_empty_map(self):
        emap = np.zeros(self.map_dims, dtype=np.float32)
        emap[MapChannel2.INITIAL_POSITION.value, 4, 4] = PRESENCE
        return emap

    def update_maps(self, state: AgentState) -> None:
        if self.map[MapChannel2.BUMP.value, 3, 4] == 0 and state.senses[Sense.BUMP.value]:
            self.map[MapChannel2.BUMP.value, 3, 4] = PRESENCE

        self.update_map_pose(state)

        # insert new sensations in central place of the map, except bump which places info in front
        self.map[MapChannel2.BREEZE.value, 4, 4] = PRESENCE if state.senses[Sense.BREEZE.value] else ABSENCE
        self.map[MapChannel2.STENCH.value, 4, 4] = PRESENCE if state.senses[Sense.STENCH.value] else ABSENCE
        self.map[MapChannel2.GLITTER.value, 4, 4] = PRESENCE if state.senses[Sense.GLITTER.value] else ABSENCE

        self.update_pose(state)

    def from_state_to_net_input(self, state: AgentState):
        self.update_maps(state)
        # for n, m in enumerate(self.map):
        #     print(f"Map about {MapChannel2(n).name}")
        #     print(m)
        # return self.map.copy()
        self.has_gold = state.gold_taken
        self.arrows_left = state.arrows_left
        self.was_scream = self.was_scream or state.senses[Sense.SCREAM.value]

        flat_map_centre = self.map[:, 1:-1, 1:-1].flatten()
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


# class CentralizedMapCNN(CentralizedMapDNN):
#
#     def __init__(self, input_dims=(8, 7, 7), n_actions=len(Action), map_dims=(8, 7, 7), **kwargs):
#         super().__init__(input_dims=input_dims, n_actions=n_actions, map_dims=map_dims, **kwargs)
#         self.name = 'CentralizedMapCompressedCNNAgent'
#
#     def get_empty_map(self):
#         emap = np.zeros(self.map_dims, dtype=np.float32)
#         emap[MapChannel2.BUMP.value, 3, 3] = ABSENCE
#         emap[MapChannel2.INITIAL_POSITION.value, 3, 3] = PRESENCE
#         emap[MapChannel2.N_ARROWS.value, :, :] = self.initial_n_arrows
#         return emap
#
#     def update_maps(self, state: AgentState) -> None:
#         self.map[MapChannel2.HAS_GOLD.value, :, :] = state.gold_taken
#         self.map[MapChannel2.N_ARROWS.value, :, :] = state.arrows_left
#         self.map[MapChannel2.SCREAM.value, :, :] = self.map[MapChannel2.SCREAM.value, 3, 3] or state.senses[Sense.SCREAM.value]
#
#     def from_state_to_net_input(self, state: AgentState):
#         self.update_maps(state)
#         # for n, m in enumerate(self.map):
#         #     print(f"Map about {MapChannel2(n).name}")
#         #     print(m)
#         # return self.map.copy()
#         # self.has_gold = state.gold_taken
#         # self.arrows_left = state.arrows_left
#         # self.was_scream = self.was_scream or state.senses[Sense.SCREAM.value]
#
#         return self.map.copy()
#
#     def maybe_switch_strategy(self, observation):
#         if observation[MapChannel2.HAS_GOLD.value, 3, 3] == 1:     # has_gold
#             if self.action_selection_strategy != self.second_eps_strategy:
#                 self.action_selection_strategy = self.second_eps_strategy
#                 # print(f'Switching to GOLD TAKEN strategy with eps={self.action_selection_strategy.epsilon}')
#         else:
#             if self.action_selection_strategy != self.first_eps_strategy:
#                 self.action_selection_strategy = self.first_eps_strategy
#                 # print(f'Switching to DEFAULT strategy with eps={self.action_selection_strategy.epsilon}')
#
#     def get_network(self):
#         return SimpleCNNQNetwork(self.input_dims, self.n_actions, self.lr)
#
#
