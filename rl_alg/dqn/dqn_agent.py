import numpy as np
import torch as T

from rl_base import Agent
from rl_alg.dqn.replay_memory import ReplayMemory
from rl_alg.epsilon_greedy_strategy import EpsilonGreedyStrategy

from abc import abstractmethod


class DQNAgent(Agent):
    def __init__(self, input_dims, n_actions, name='DQNAgent', gamma=0.99, epsilon=0.5, lr=0.01, batch_size=64,
                 max_mem_size=500, eps_end=0.01, eps_dec=5e-4, replace_target=50, polyak=True, soft_tau=1e-2,
                 replay_memory=None, net_state_dict=None, optimizer_state_dict=None,
                 manual_action=False, manual_control=None):
        super().__init__(name)
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.gamma = gamma
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.batch_size = batch_size
        self.iter_cntr = 0
        self.replace_target = replace_target
        self.soft_tau = soft_tau
        self.polyak = polyak

        self.Q_eval = self.get_network()
        self.Q_next = self.get_network()

        self.Q_next.load_state_dict(self.Q_eval.state_dict())
        self.Q_next.eval()

        if replay_memory is not None:
            self.memory = replay_memory
        else:
            self.memory = ReplayMemory(max_mem_size, input_dims)

        self.action_selection_strategy = EpsilonGreedyStrategy(epsilon, eps_end, eps_dec)
        self.manual_action = manual_action
        self.manual_control = manual_control

        if net_state_dict is not None:
            self.Q_eval.load_state_dict(net_state_dict)
            self.Q_next.load_state_dict(net_state_dict)
            self.Q_next.eval()

        if optimizer_state_dict is not None:
            self.Q_eval.optimizer.load_state_dict(optimizer_state_dict)

    def choose_action(self, observation):
        if self.manual_action:
            action = self.manual_control.get_action()
        else:
            if np.random.random() >= self.action_selection_strategy.get_epsilon():
                state = T.tensor(observation).to(self.Q_eval.device)
                actions = self.Q_eval.forward(state)
                action = T.argmax(actions).item()
            else:
                action = np.random.choice(self.action_space)
            self.action_selection_strategy.update_epsilon()

        return action

    def learn(self, observation, action, reward, new_observation, done, combined_replay=True):

        self.memory.store_transitions(observation, action, reward, new_observation, done)

        if not self.memory.can_provide_sample(self.batch_size):
            return

        self.Q_eval.optimizer.zero_grad()

        state_batch, new_state_batch, action_batch, reward_batch, terminal_batch =\
            self.memory.get_sample(self.batch_size, self.Q_eval)

        if combined_replay:
            state_batch[0, :] = T.tensor(observation)
            new_state_batch[0, :] = T.tensor(new_observation)
            action_batch[0] = T.tensor(action)
            reward_batch[0] = T.tensor(reward)
            terminal_batch[0] = T.tensor(done)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_next.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.iter_cntr += 1

        if self.polyak:
            for target_param, param in zip(self.Q_next.parameters(), self.Q_eval.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau
                )
        else:
            if self.iter_cntr % self.replace_target == 0:
                self.Q_next.load_state_dict(self.Q_eval.state_dict())

    @staticmethod
    def load_static(load_state_path, n_actions, input_dims):
        loaded_state = T.load(load_state_path)

        batch_size_dqn = loaded_state['batch_size']
        gamma_dqn = loaded_state['gamma']
        target_update_dqn = loaded_state['replace_target']
        lr_dqn = loaded_state['lr']
        replay_memory_dqn = loaded_state['replay_memory']
        net_state_dict_dqn = loaded_state['state_dict']
        optimizer_state_dict_dqn = loaded_state['optimizer']
        agent = DQNAgent(n_actions=n_actions, input_dims=input_dims, gamma=gamma_dqn, epsilon=0.0, lr=lr_dqn,
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
        self.soft_tau = loaded_state['soft_tau']
        self.polyak = loaded_state['polyak']
        return loaded_state

    def get_state_dict(self):
        state_dict = {
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
            'soft_tau': self.soft_tau,
            'replay_memory': self.memory,
            'polyak': self.polyak
        }
        return state_dict

    def save(self, save_path):
        state_dict = self.get_state_dict()
        T.save(state_dict, save_path)

    def get_instruction_string(self):
        if self.manual_action:
            return self.manual_control.get_instruction_string()
        else:
            return ["Press p to on/off auto mode", "or any other key to one step"]

    @abstractmethod
    def from_state_to_net_input(self, state):
        pass

    def observe(self, state):
        return self.from_state_to_net_input(state)

    @abstractmethod
    def get_network(self):
        pass
