from abc import ABC, abstractmethod


class Env(ABC):
    def __init__(self, name):
        super().__init__()
        self.name = name

    @abstractmethod
    def reset_env(self):
        pass

    @abstractmethod
    def render(self, *args, **kwargs):
        pass

    @abstractmethod
    def step(self, action):
        pass


class Agent(ABC):

    def __init__(self, name):
        super().__init__()
        self.name = name
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