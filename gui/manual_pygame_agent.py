from agent import Agent
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


class ManualPygameAgent(Agent):
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
