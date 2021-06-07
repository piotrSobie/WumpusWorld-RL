from abc import ABC, abstractmethod


class Agent(ABC):
    @abstractmethod
    def choose_action(self, observation):
        pass

    @abstractmethod
    def learn(self, observation, action, reward, new_observation, done):
        pass
