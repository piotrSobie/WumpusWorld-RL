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


class QuitException(Exception):
    pass


wsad_info = ["w - move up", "a - move left", "d - move right", "s - move down"]


def wsad_manual_simple_action():
    action = None
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
    return action


class Lv1ManualPygameAgent(Agent):

    def __init__(self):
        super().__init__()
        self.manual_action = True

    def learn(self, observation, action, reward, new_observation, done):
        pass

    def choose_action(self, observation):
        return wsad_manual_simple_action()

    def get_instruction_string(self):
        return wsad_info


class ManualPygameAgent(Agent):

    def __init__(self):
        super().__init__()
        self.manual_action = True

    def choose_action(self, observation):
        action = None
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
        return action

    def get_instruction_string(self):
        return ["w - forward", "a - turn left", "d - turn right", "g - take gold", "z - shoot", "c - climb out"]

    def save(self, save_path):
        pass

    def learn(self, observation, action, reward, new_observation, done):
        pass
