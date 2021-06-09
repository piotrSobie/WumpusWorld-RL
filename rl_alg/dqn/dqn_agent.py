import numpy as np
import torch as T

from agent import Agent
from rl_alg.dqn.dqn_network import DeepQNetwork
from rl_alg.dqn.replay_memory import ReplayMemory
from rl_alg.epsilon_greedy_strategy import EpsilonGreedyStrategy

from abc import abstractmethod


class DQNAgent(Agent):
    def __init__(self, input_dims, n_actions, gamma=0.99, epsilon=0.5, lr=0.01, batch_size=64,
                 max_mem_size=500, eps_end=0.01, eps_dec=5e-4, replace_target=50,
                 replay_memory=None, net_state_dict=None, optimizer_state_dict=None):
        super().__init__()
        self.gamma = gamma
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.batch_size = batch_size
        self.iter_cntr = 0
        self.replace_target = replace_target

        self.Q_eval = DeepQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims, fc1_dims=10, fc2_dims=10)
        self.Q_next = DeepQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims, fc1_dims=10, fc2_dims=10)

        self.Q_next.load_state_dict(self.Q_eval.state_dict())
        self.Q_next.eval()

        if replay_memory is not None:
            self.memory = replay_memory
        else:
            self.memory = ReplayMemory(max_mem_size, input_dims)

        self.action_selection_strategy = EpsilonGreedyStrategy(epsilon, eps_end, eps_dec)

        if net_state_dict is not None:
            self.Q_eval.load_state_dict(net_state_dict)
            self.Q_next.load_state_dict(net_state_dict)
            self.Q_next.eval()

        if optimizer_state_dict is not None:
            self.Q_eval.optimizer.load_state_dict(optimizer_state_dict)

    def choose_action(self, observation):
        if np.random.random() >= self.action_selection_strategy.get_epsilon():
            state = T.tensor(self.from_state_to_input_vector(observation)).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        self.action_selection_strategy.update_epsilon()

        return action

    def learn(self, observation, action, reward, new_observation, done):

        obs_vector = self.from_state_to_input_vector(observation)
        new_obs_vector = self.from_state_to_input_vector(new_observation)
        self.memory.store_transitions(obs_vector, action, reward, new_obs_vector, done)

        if not self.memory.can_provide_sample(self.batch_size):
            return

        self.Q_eval.optimizer.zero_grad()

        state_batch, new_state_batch, action_batch, reward_batch, terminal_batch =\
            self.memory.get_sample(self.batch_size, self.Q_eval)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_next.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.iter_cntr += 1

        if self.iter_cntr % self.replace_target == 0:
            self.Q_next.load_state_dict(self.Q_eval.state_dict())

    @staticmethod
    def load_static(load_state_path, n_actions, n_observations):
        loaded_state = T.load(load_state_path)

        batch_size_dqn = loaded_state['batch_size']
        gamma_dqn = loaded_state['gamma']
        target_update_dqn = loaded_state['replace_target']
        lr_dqn = loaded_state['lr']
        replay_memory_dqn = loaded_state['replay_memory']
        net_state_dict_dqn = loaded_state['state_dict']
        optimizer_state_dict_dqn = loaded_state['optimizer']
        agent = DQNAgent(n_actions=n_actions, input_dims=n_observations, gamma=gamma_dqn, epsilon=0.0, lr=lr_dqn,
                         batch_size=batch_size_dqn, eps_end=0.0, eps_dec=0.0, replace_target=target_update_dqn,
                         replay_memory=replay_memory_dqn, net_state_dict=net_state_dict_dqn,
                         optimizer_state_dict=optimizer_state_dict_dqn)
        return agent

    def load(self, load_state_path):
        loaded_state = T.load(load_state_path)
        self.memory = loaded_state['replay_memory']
        self.action_selection_strategy = EpsilonGreedyStrategy(loaded_state['epsilon'],
                                                               loaded_state['eps_min'],
                                                               loaded_state['eps_dec'])
        self.Q_eval.load_state_dict(loaded_state['state_dict'])
        self.Q_next.load_state_dict(loaded_state['state_dict'])
        self.Q_next.eval()
        self.Q_eval.optimizer.load_state_dict(loaded_state['optimizer'])
        self.lr = loaded_state['lr']
        self.gamma = loaded_state['gamma']
        self.replace_target = loaded_state['replace_target']

    def save(self, save_path):
        state = {
            'state_dict': self.Q_eval.state_dict(),
            'optimizer': self.Q_eval.optimizer.state_dict(),
            'gamma': self.gamma,
            'epsilon': self.action_selection_strategy.epsilon,
            'eps_dec': self.action_selection_strategy.eps_dec,
            'eps_min': self.action_selection_strategy.eps_min,
            'batch_size': self.batch_size,
            'lr': self.lr,
            'max_mem_size': self.memory.mem_size,
            'replace_target': self.replace_target,
            'replay_memory': self.memory,

        }
        T.save(state, save_path)

    @abstractmethod
    def from_state_to_input_vector(self, state):
        pass
