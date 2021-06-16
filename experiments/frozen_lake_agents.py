import numpy as np

from rl_alg.q_agent import QAgent
from rl_alg.dqn.dqn_agent import DQNAgent
from rl_alg.dqn.dqn_network import DeepQNetwork
from gui.manual_pygame_agent import SimpleManualControl


class FrozenLakeQAgent(QAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.manual_control = SimpleManualControl()

    def from_state_to_idx(self, state):
        return state


class FrozenLakeDQNAgent(DQNAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eye = np.eye(16, dtype=np.float32)
        self.manual_control = SimpleManualControl()

    def from_state_to_net_input(self, state):
        return self.eye[state]

    def get_network(self):
        return DeepQNetwork(self.lr, n_actions=self.n_actions,
                            input_dims=self.input_dims, fc1_dims=10, fc2_dims=10)
