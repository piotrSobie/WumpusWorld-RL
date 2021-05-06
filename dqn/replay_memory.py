import numpy as np
import torch as T


class ReplayMemory:
    def __init__(self, mem_size, input_dims):
        self.mem_size = mem_size
        self.mem_counter = 0

        self.state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transitions(self, state, action, reward, new_state, terminal):
        index = self.mem_counter % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = terminal

        self.mem_counter += 1

    def get_sample(self, batch_size, q_eval):
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        state_batch = T.tensor(self.state_memory[batch]).to(q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(q_eval.device)
        action_batch = self.action_memory[batch]
        reward_batch = T.tensor(self.reward_memory[batch]).to(q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(q_eval.device)

        return state_batch, new_state_batch, action_batch, reward_batch, terminal_batch

    def can_provide_sample(self, batch_size):
        return self.mem_counter < batch_size
