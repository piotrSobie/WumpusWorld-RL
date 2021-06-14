from abc import ABC, abstractmethod


class Agent(ABC):

    def __init__(self):
        super().__init__()
        self.manual_action = False
        self.action_selection_strategy = None

    def reset_for_new_episode(self):
        pass

    def observe(self, state):
        return state

    @abstractmethod
    def choose_action(self, observation):
        pass

    @abstractmethod
    def learn(self, observation, action, reward, new_observation, done):
        pass

    @abstractmethod
    def save(self, save_path):
        pass

    @abstractmethod
    def get_instruction_string(self):
        pass
