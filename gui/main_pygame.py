from gui.manual_pygame_agent import ManualPygameAgent, QuitException
from rl_alg.dqn.dqn_agent import DQNAgent
from time import sleep

import pygame
from pygame.locals import (
        RLEACCEL,
)

red = (153, 0, 0)
white = (255, 255, 255)
black = (0, 0, 0)
brown = (102, 51, 0)
blue = (0, 128, 255)
green = (0, 153, 0)

FIELD_SIZE_X = 150
FIELD_SIZE_Y = 150
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 600


def draw_game_window(env, screen, assets, text, visited, sensed):

    screen.fill(white)

    agent_img = None
    if env.agent_direction == 0:
        agent_img = assets['agent_up']
    elif env.agent_direction == 1:
        agent_img = assets['agent_right']
    elif env.agent_direction == 2:
        agent_img = assets['agent_down']
    elif env.agent_direction == 3:
        agent_img = assets['agent_left']

    for i in range(len(env.grid_world)):
        for j in range(len(env.grid_world[i])):
            if env.grid_world[i][j] == env.regular_field:
                pygame.draw.rect(screen, brown, pygame.Rect(j * FIELD_SIZE_X, i * FIELD_SIZE_Y,
                                                            FIELD_SIZE_X, FIELD_SIZE_Y))
            elif env.grid_world[i][j] == env.visited_field:
                pygame.draw.rect(screen, green, pygame.Rect(j * FIELD_SIZE_X, i * FIELD_SIZE_Y,
                                                            FIELD_SIZE_X, FIELD_SIZE_Y))
            elif env.grid_world[i][j] == env.gold_field:
                screen.blit(assets['gold'], (j * FIELD_SIZE_X, i * FIELD_SIZE_Y,
                                             FIELD_SIZE_X, FIELD_SIZE_Y))
            elif env.grid_world[i][j] == env.wumpus_field:
                pygame.draw.rect(screen, red, pygame.Rect(j * FIELD_SIZE_X, i * FIELD_SIZE_Y,
                                                          FIELD_SIZE_X, FIELD_SIZE_Y))
                screen.blit(assets['wumpus'], (j * FIELD_SIZE_X, i * FIELD_SIZE_Y,
                                         FIELD_SIZE_X, FIELD_SIZE_Y))
            elif env.grid_world[i][j] == env.pit_field:
                screen.blit(assets['pit'], (j * FIELD_SIZE_X, i * FIELD_SIZE_Y, FIELD_SIZE_X, FIELD_SIZE_Y))

            if (i == env.cave_entry_x) & (j == env.cave_entry_y):
                screen.blit(assets['cave_entry'], (j * FIELD_SIZE_X, i * FIELD_SIZE_Y, FIELD_SIZE_X, FIELD_SIZE_Y))

            if (i == env.agentPosXY[0]) & (j == env.agentPosXY[1]):
                screen.blit(agent_img, (j * FIELD_SIZE_X, i * FIELD_SIZE_Y, FIELD_SIZE_X, FIELD_SIZE_Y))

            if visited[i][j] == 0:
                pygame.draw.rect(screen, white, pygame.Rect(j * FIELD_SIZE_X, i * FIELD_SIZE_Y,
                                                            FIELD_SIZE_X, FIELD_SIZE_Y))

            if (len(sensed) > 0) & (i == env.agentPosXY[0]) & (j == env.agentPosXY[1]):
                my_sensor_text = assets['font'].render(sensed, False, black)
                screen.blit(my_sensor_text, (j * FIELD_SIZE_X + 5, i * FIELD_SIZE_Y + 5))

            pygame.draw.rect(screen, blue, pygame.Rect(j * FIELD_SIZE_X, i * FIELD_SIZE_Y,
                                                       FIELD_SIZE_X, FIELD_SIZE_Y), 5)

    for t in range(len(text)):
        msg = assets['font'].render(text[t], False, black)
        screen.blit(msg, (FIELD_SIZE_X * len(env.grid_world[0]) + 10, t * 25))

    pygame.display.flip()

    return


def load_assets():
    assets = {}
    wumpus_img = pygame.image.load("assets/wumpus_img.png").convert()
    wumpus_img = pygame.transform.scale(wumpus_img, (FIELD_SIZE_X, FIELD_SIZE_Y))
    wumpus_img.set_colorkey(black, RLEACCEL)
    assets['wumpus'] = wumpus_img

    pit_img = pygame.image.load("assets/pit_img.png").convert()
    pit_img = pygame.transform.scale(pit_img, (FIELD_SIZE_X, FIELD_SIZE_Y))
    assets['pit'] = pit_img

    gold_img = pygame.image.load("assets/gold_img.png").convert()
    gold_img = pygame.transform.scale(gold_img, (FIELD_SIZE_X, FIELD_SIZE_Y))
    assets['gold'] = gold_img

    agent_img_left = pygame.image.load("assets/arrow_img.png").convert()
    agent_img_left = pygame.transform.scale(agent_img_left, (FIELD_SIZE_X, FIELD_SIZE_Y))
    assets['agent_left'] = agent_img_left

    agent_img_down = pygame.image.load("assets/arrow_img.png").convert()
    agent_img_down = pygame.transform.scale(agent_img_down, (FIELD_SIZE_X, FIELD_SIZE_Y))
    agent_img_down = pygame.transform.rotate(agent_img_down, 90)
    assets['agent_down'] = agent_img_down

    agent_img_right = pygame.image.load("assets/arrow_img.png").convert()
    agent_img_right = pygame.transform.scale(agent_img_right, (FIELD_SIZE_X, FIELD_SIZE_Y))
    agent_img_right = pygame.transform.rotate(agent_img_right, 180)
    assets['agent_right'] = agent_img_right

    agent_img_up = pygame.image.load("assets/arrow_img.png").convert()
    agent_img_up = pygame.transform.scale(agent_img_up, (FIELD_SIZE_X, FIELD_SIZE_Y))
    agent_img_up = pygame.transform.rotate(agent_img_up, 270)
    assets['agent_up'] = agent_img_up

    cave_entry_img = pygame.image.load("assets/cave_entry_img.png").convert()
    cave_entry_img = pygame.transform.scale(cave_entry_img, (FIELD_SIZE_X, FIELD_SIZE_Y))
    assets['cave_entry'] = cave_entry_img

    assets['font'] = pygame.font.SysFont('Times New Roman', 18)

    return assets


def main_pygame(wumpus_env, agent_type='dqn', show_whole_map=True):

    env = wumpus_env

    if agent_type == 'dqn':
        agent = DQNAgent()
    elif agent_type == 'manual':
        agent = ManualPygameAgent()
    else:
        raise ValueError('Unsupported agent type.')

    agent_state = env.reset_env()       # TODO this looks wrong

    # Initialize pygame
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    screen.fill(white)
    assets = load_assets()

    instruction_string = ["Goal: leave cave with gold", "Instruction:", "q | ESC - terminate program",
                          "w - move forward", "a - turn left", "d - turn right", "g - take gold",
                          "z - shoot", "c - leave cave in entry"]
    msg = instruction_string + [f"Agent state: {agent_state}"]

    visited_rooms = [[0, 0, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0],
                     [1, 0, 0, 0]]

    if show_whole_map:
        for i in range(len(visited_rooms)):
            for j in range(len(visited_rooms[i])):
                visited_rooms[j][i] = 1

    sensed_danger = env.get_sensed_string()

    draw_game_window(env, screen, assets, msg, visited_rooms, sensed_danger)

    running = True
    total_reward = 0
    # Main loop
    while running:
        observation = None
        try:
            action = agent.choose_action(observation)
        except QuitException:
            action = None
            running = False

        if action is not None:
            new_state, reward, done, info, _ = env.step(action)
            visited_rooms[env.agentPosXY[0]][env.agentPosXY[1]] = 1
            total_reward += reward
            info = info.split(".")
            msg = instruction_string + [f"Agent state: {new_state}", f"Reward this step: {reward}",
                                        f"Total reward: {total_reward}", f"Done: {done}", "Info:"] + info \
                  + ["The agent senses:"]
            sensed_danger = env.get_sensed_string()
            if not sensed_danger:
                msg += ["nothing"]
            else:
                msg += [sensed_danger]

            if done:
                msg += ["", "Game ended", "Press q or esc to leave"]
                for i in range(len(visited_rooms)):
                    for j in range(len(visited_rooms[i])):
                        visited_rooms[j][i] = 1

            draw_game_window(env, screen, assets, msg, visited_rooms, sensed_danger)

        sleep(0.05)

    return
