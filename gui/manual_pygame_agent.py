from time import sleep

from rl_base import Agent
import pygame

from pygame.locals import (
    RLEACCEL,
    K_ESCAPE,
    K_q,
    K_w,
    K_s,
    K_d,
    K_a,
    K_g,
    K_z,
    K_c,
    KEYDOWN,
    QUIT,
)
from abc import ABC, abstractmethod


class QuitException(Exception):
    pass


class ManualActionControl(ABC):

    @staticmethod
    @abstractmethod
    def get_action():
        pass

    @staticmethod
    @abstractmethod
    def get_instruction_string():
        pass


class SimpleManualControl(ManualActionControl):

    @staticmethod
    def get_action():
        action = None
        while action is None:
            for event in pygame.event.get():
                if event.type == KEYDOWN:
                    if (event.key == K_ESCAPE) | (event.key == K_q):
                        raise QuitException()
                    elif event.key == K_w:
                        action = 0
                    elif event.key == K_s:
                        action = 1
                    elif event.key == K_a:
                        action = 3
                    elif event.key == K_d:
                        action = 2
                elif event.type == QUIT:
                    raise QuitException()
            sleep(0.05)
        return action

    @staticmethod
    def get_instruction_string():
        return ["w - move up", "a - move left", "d - move right", "s - move down"]


class TurningManualControl(ManualActionControl):

    @staticmethod
    def get_action():
        action = None
        while action is None:
            for event in pygame.event.get():
                if event.type == KEYDOWN:
                    if (event.key == K_ESCAPE) | (event.key == K_q):
                        raise QuitException()
                    elif event.key == K_w:
                        action = 0
                    elif event.key == K_a:
                        action = 1
                    elif event.key == K_d:
                        action = 2
                    elif event.key == K_g:
                        action = 3
                    elif event.key == K_z:
                        action = 4
                    elif event.key == K_c:
                        action = 5
                elif event.type == QUIT:
                    raise QuitException()
            sleep(0.05)
        return action

    @staticmethod
    def get_instruction_string():
        return ["w - forward", "a - turn left", "d - turn right", "g - take gold", "z - shoot", "c - climb out"]


class Lv1ManualPygameAgent(Agent):
    def __init__(self):
        super().__init__()
        self.manual_action = True
        self.control = SimpleManualControl()

    def learn(self, observation, action, reward, new_observation, done):
        pass

    def choose_action(self, observation):
        return self.control.get_action()

    def get_instruction_string(self):
        return self.control.get_instruction_string()

    def save(self, save_path):
        pass



class ManualPygameAgent(Agent):

    def __init__(self):
        super().__init__()
        self.manual_action = True
        self.control = TurningManualControl()

    def choose_action(self, observation):
        return self.control.get_action()

    def get_instruction_string(self):
        return self.control.get_instruction_string()

    def save(self, save_path):
        pass

    def learn(self, observation, action, reward, new_observation, done):
        pass
