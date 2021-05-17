import numpy as np
import torch as T

from rl_alg.dqn.dqn_network import DeepQNetwork
from rl_alg.dqn.replay_memory import ReplayMemory
from rl_alg.dqn.epsilon_greedy_strategy import EpsilonGreedyStrategy


class Agent:
    def __init__(self, input_dims, n_actions, gamma=0.99, epsilon=1.0, lr=0.01, batch_size=64,
                 max_mem_size=100000, eps_end=0.01, eps_dec=5e-4, replace_target=100, loaded_state=None):
        self.gamma = gamma
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.batch_size = batch_size
        self.iter_cntr = 0
        self.replace_target = replace_target

        self.Q_eval = DeepQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims, fc1_dims=256, fc2_dims=256)
        self.Q_next = DeepQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims, fc1_dims=256, fc2_dims=256)

        self.Q_next.load_state_dict(self.Q_eval.state_dict())
        self.Q_next.eval()

        self.memory = ReplayMemory(max_mem_size, input_dims)
        self.eps_strategy = EpsilonGreedyStrategy(epsilon, eps_end, eps_dec)

        # load data
        if loaded_state is not None:
            self.gamma = loaded_state['gamma']
            self.eps_strategy = loaded_state['epsilon_strategy']
            self.batch_size = loaded_state['batch_size']
            self.lr = loaded_state['lr']
            self.replace_target = loaded_state['replace_target']
            self.memory = loaded_state['replay_memory']

            self.Q_eval.load_state_dict(loaded_state['state_dict'])
            self.Q_next.load_state_dict(loaded_state['state_dict'])
            self.Q_next.eval()
            self.Q_eval.optimizer.load_state_dict(loaded_state['optimizer'])

    def choose_action(self, observation):
        if np.random.random() >= self.eps_strategy.get_epsilon():
            state = T.tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if self.memory.can_provide_sample(self.batch_size):
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
        self.eps_strategy.update_epsilon()

        if self.iter_cntr % self.replace_target == 0:
            self.Q_next.load_state_dict(self.Q_eval.state_dict())
