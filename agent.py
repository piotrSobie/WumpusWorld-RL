from abc import ABC, abstractmethod


class Agent(ABC):

    def __init__(self):
        super().__init__()
        self.manual_action = False
        self.action_selection_strategy = None

    @abstractmethod
    def choose_action(self, observation):
        pass

    @abstractmethod
    def learn(self, observation, action, reward, new_observation, done):
        pass

    @abstractmethod
    def save(self, save_path):
        pass
