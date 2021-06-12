import numpy as np
from rl_base import Agent
from rl_alg.epsilon_greedy_strategy import EpsilonGreedyStrategy
from gui.manual_pygame_agent import wsad_manual_simple_action, wsad_info
from abc import abstractmethod


class QAgent(Agent):

    def __init__(self, n_states, n_actions, lr=0.1, gamma=0.99,  epsilon=0.5, eps_end=0.0001, eps_dec=5e-4,
                 initial_q_value=0.0, q_table=None, manual_action=False):
        super().__init__()
        self.gamma = gamma
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.n_states = n_states
        self.action_selection_strategy = EpsilonGreedyStrategy(epsilon, eps_end, eps_dec)
        self.q_table = q_table if q_table is not None else self.init_q_table(initial_q_value)
        self.manual_action = manual_action

    def init_q_table(self, initial_q_value=0.):
        q_table = initial_q_value * np.ones((self.n_states, len(self.action_space)))
        return q_table

    def choose_action(self, observation):
        assert 0 <= observation < self.n_states, \
            f"Bad observation. Has to be int between 0 and {self.n_states}"

        if self.manual_action:
            action = wsad_manual_simple_action()
        else:
            if np.random.random() >= self.action_selection_strategy.get_epsilon():
                action = np.argmax(self.q_table[observation, :])
            else:
                action = np.random.choice(self.action_space)
            self.action_selection_strategy.update_epsilon()

        return action

    def learn(self, observation, action, reward, new_observation, done):
        idx = self.from_state_to_idx(observation)
        new_idx = self.from_state_to_idx(new_observation)
        self.q_table[idx, action] = (1 - self.lr) * self.q_table[idx, action] + \
            self.lr * (reward + self.gamma * np.max(self.q_table[new_idx, :]))

    def save(self, path):
        np.save(path, self.q_table)

    def load(self, path):
        self.q_table = np.load(path)

    @abstractmethod
    def from_state_to_idx(self, state):
        pass

    def get_instruction_string(self):
        if self.manual_action:
            return wsad_info
        else:
            return ["Press p to on/off auto mode", "or any other key to one step"]
