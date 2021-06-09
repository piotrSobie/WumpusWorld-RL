import pygame

from pygame.locals import (
    RLEACCEL,
    KEYDOWN,
    K_ESCAPE,
    K_q,
    K_p,
    QUIT,
)

RED = (153, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BROWN = (102, 51, 0)
BLUE = (0, 128, 255)
GREEN = (0, 153, 0)

FIELD_SIZE_X = 150
FIELD_SIZE_Y = 150
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 600


def load_assets():
    assets = {}

    pit_img = pygame.image.load("assets/pit_img.png").convert()
    pit_img = pygame.transform.scale(pit_img, (FIELD_SIZE_X, FIELD_SIZE_Y))
    assets['pit'] = pit_img

    gold_img = pygame.image.load("assets/gold_img.png").convert()
    gold_img = pygame.transform.scale(gold_img, (FIELD_SIZE_X, FIELD_SIZE_Y))
    assets['gold'] = gold_img

    agent_img = pygame.image.load("assets/agent_img.png").convert()
    agent_img = pygame.transform.scale(agent_img, (FIELD_SIZE_X, FIELD_SIZE_Y))
    agent_img.set_colorkey(BLACK, RLEACCEL)
    assets['agent'] = agent_img

    agent_img_left = pygame.image.load("assets/arrow_img2.png").convert_alpha()
    agent_img_left = pygame.transform.scale(agent_img_left, (FIELD_SIZE_X // 2, FIELD_SIZE_Y // 2))
    assets['arrow_left'] = agent_img_left

    agent_img_down = pygame.image.load("assets/arrow_img2.png").convert_alpha()
    agent_img_down = pygame.transform.scale(agent_img_down, (FIELD_SIZE_X // 2, FIELD_SIZE_Y // 2))
    agent_img_down = pygame.transform.rotate(agent_img_down, 90)
    assets['arrow_down'] = agent_img_down

    agent_img_right = pygame.image.load("assets/arrow_img2.png").convert_alpha()
    agent_img_right = pygame.transform.scale(agent_img_right, (FIELD_SIZE_X // 2, FIELD_SIZE_Y // 2))
    agent_img_right = pygame.transform.rotate(agent_img_right, 180)
    assets['arrow_right'] = agent_img_right

    agent_img_up = pygame.image.load("assets/arrow_img2.png").convert_alpha()
    agent_img_up = pygame.transform.scale(agent_img_up, (FIELD_SIZE_X // 2, FIELD_SIZE_Y // 2))
    agent_img_up = pygame.transform.rotate(agent_img_up, 270)
    assets['arrow_up'] = agent_img_up

    cave_entry_img = pygame.image.load("assets/cave_entry_img.png").convert()
    cave_entry_img = pygame.transform.scale(cave_entry_img, (FIELD_SIZE_X, FIELD_SIZE_Y))
    assets['cave_entry'] = cave_entry_img

    assets['font'] = pygame.font.SysFont('Times New Roman', 18)

    return assets
