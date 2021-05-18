from wumpus_envs.wumpus_env_lv1 import WumpusWorldLv1

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

red = (153, 0, 0)
white = (255, 255, 255)
black = (0, 0, 0)
brown = (102, 51, 0)
blue = (0, 128, 255)
green = (0, 153, 0)


def draw_grid_world(env, screen, gold_img, wumpus_img, pit_img, agent_img, cave_entry,
                    field_size_x, field_size_y, text, font, visited, sensed):
    screen.fill(white)

    for i in range(len(env.grid_world)):
        for j in range(len(env.grid_world[i])):
            if env.grid_world[i][j] == env.regular_field:
                pygame.draw.rect(screen, brown, pygame.Rect(j * field_size_x, i * field_size_y,
                                                            field_size_x, field_size_y))
            elif env.grid_world[i][j] == env.visited_field:
                pygame.draw.rect(screen, green, pygame.Rect(j * field_size_x, i * field_size_y,
                                                            field_size_x, field_size_y))
            elif env.grid_world[i][j] == env.gold_field:
                screen.blit(gold_img, (j * field_size_x, i * field_size_y,
                                       field_size_x, field_size_y))
            elif env.grid_world[i][j] == env.wumpus_field:
                pygame.draw.rect(screen, red, pygame.Rect(j * field_size_x, i * field_size_y,
                                                          field_size_x, field_size_y))
                screen.blit(wumpus_img, (j * field_size_x, i * field_size_y,
                                         field_size_x, field_size_y))
            elif env.grid_world[i][j] == env.pit_field:
                screen.blit(pit_img, (j * field_size_x, i * field_size_y, field_size_x, field_size_y))

            if (i == env.cave_entry_x) & (j == env.cave_entry_y):
                screen.blit(cave_entry, (j * field_size_x, i * field_size_y, field_size_x, field_size_y))

            if (i == env.agentPosXY[0]) & (j == env.agentPosXY[1]):
                screen.blit(agent_img, (j * field_size_x, i * field_size_y, field_size_x, field_size_y))

            if visited[i][j] == 0:
                pygame.draw.rect(screen, white, pygame.Rect(j * field_size_x, i * field_size_y,
                                                            field_size_x, field_size_y))

            if (len(sensed) > 0) & (i == env.agentPosXY[0]) & (j == env.agentPosXY[1]):
                my_sensor_text = font.render(sensed, False, black)
                screen.blit(my_sensor_text, (j * field_size_x + 5, i * field_size_y + 5))

            pygame.draw.rect(screen, blue, pygame.Rect(j * field_size_x, i * field_size_y,
                                                       field_size_x, field_size_y), 5)

    for t in range(len(text)):
        msg = font.render(text[t], False, black)
        screen.blit(msg, (field_size_x * len(env.grid_world[0]) + 10, t * 25))

    pygame.display.flip()

    return


def manual_play_pygame_lv1(show_whole_map):
    env = WumpusWorldLv1()
    agent_state = env.reset_env()
    sesnsed_danger = env.use_senses()
    sensed_string = ""
    if not sesnsed_danger:
        sensed_string += "nothing"
    else:
        for i in sesnsed_danger:
            sensed_string += i + " "

    # Initialize pygame
    pygame.init()
    pygame.font.init()
    my_font = pygame.font.SysFont('Times New Roman', 18)

    # constants
    screen_width = 900
    screen_height = 600
    field_size_x = 150
    field_size_y = 150
    instruction_string = ["Goal: reach gold", "Instruction:", "q | ESC - terminate program", "w - move up",
                          "s - move down", "d - move right", "a - move left"]

    screen = pygame.display.set_mode((screen_width, screen_height))
    screen.fill(white)

    wumpus_img = pygame.image.load("assets/wumpus_img.png").convert()
    wumpus_img = pygame.transform.scale(wumpus_img, (field_size_x, field_size_y))
    wumpus_img.set_colorkey(black, RLEACCEL)

    pit_img = pygame.image.load("assets/pit_img.png").convert()
    pit_img = pygame.transform.scale(pit_img, (field_size_x, field_size_y))

    gold_img = pygame.image.load("assets/gold_img.png").convert()
    gold_img = pygame.transform.scale(gold_img, (field_size_x, field_size_y))

    agent_img = pygame.image.load("assets/agent_img.png").convert()
    agent_img = pygame.transform.scale(agent_img, (field_size_x, field_size_y))

    cave_entry_img = pygame.image.load("assets/cave_entry_img.png").convert()
    cave_entry_img = pygame.transform.scale(cave_entry_img, (field_size_x, field_size_y))

    msg = instruction_string + [f"Agent state: {agent_state}"]

    visited_rooms = [[0, 0, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0],
                     [1, 0, 0, 0]]

    if show_whole_map:
        for i in range(len(visited_rooms)):
            for j in range(len(visited_rooms[i])):
                visited_rooms[j][i] = 1

    draw_grid_world(env, screen, gold_img, wumpus_img, pit_img, agent_img, cave_entry_img,
                    field_size_x, field_size_y, msg, my_font, visited_rooms, sensed_string)

    running = True
    total_reward = 0
    done = False
    # Main loop
    while running:
        action = None
        for event in pygame.event.get():
            if event.type == KEYDOWN:
                if (event.key == K_ESCAPE) | (event.key == K_q) | done:
                    running = False
                elif event.key == K_w:
                    action = 0
                elif event.key == K_s:
                    action = 1
                elif event.key == K_d:
                    action = 2
                elif event.key == K_a:
                    action = 3
            elif event.type == QUIT:
                running = False

        if action is not None:
            new_state, reward, done, info, _ = env.step(action)
            visited_rooms[env.agentPosXY[0]][env.agentPosXY[1]] = 1
            total_reward += reward
            info = info.split(".")
            msg = instruction_string + [f"Agent state: {new_state}", f"Reward this step: {reward}",
                                        f"Total reward: {total_reward}", f"Done: {done}", "Info:"] + info\
                  + ["The agent senses:"]
            sesnsed_danger = env.use_senses()
            if not sesnsed_danger:
                msg += ["nothing"]
            else:
                for i in sesnsed_danger:
                    msg += [i]
            if done:
                msg += ["", "Game ended", "Press any key to leave"]

            sensed_string = ""
            if not sesnsed_danger:
                sensed_string += "nothing"
            else:
                for i in sesnsed_danger:
                    sensed_string += i + " "

            if done:
                for i in range(len(visited_rooms)):
                    for j in range(len(visited_rooms[i])):
                        visited_rooms[j][i] = 1

            draw_grid_world(env, screen, gold_img, wumpus_img, pit_img, agent_img, cave_entry_img,
                            field_size_x, field_size_y, msg, my_font, visited_rooms, sensed_string)

    return


def manual_play_pygame_lv2_plus(wumpus_env, show_whole_map):
    env = wumpus_env

    # for lv 4 only
    try:
        if env.random_grid is not None:
            env.random_grid = True
    except:
        print()

    agent_state = env.reset_env()
    sesnsed_danger = env.get_sensed_string()

    # Initialize pygame
    pygame.init()
    pygame.font.init()
    my_font = pygame.font.SysFont('Times New Roman', 18)

    # constants
    screen_width = 1000
    screen_height = 600
    field_size_x = 150
    field_size_y = 150
    instruction_string = ["Goal: leave cave with gold", "Instruction:", "q | ESC - terminate program",
                          "w - move forward", "a - turn left", "d - turn right", "g - take gold",
                          "z - shoot", "c - leave cave in entry"]

    screen = pygame.display.set_mode((screen_width, screen_height))
    screen.fill(white)

    wumpus_img = pygame.image.load("assets/wumpus_img.png").convert()
    wumpus_img = pygame.transform.scale(wumpus_img, (field_size_x, field_size_y))
    wumpus_img.set_colorkey(black, RLEACCEL)

    pit_img = pygame.image.load("assets/pit_img.png").convert()
    pit_img = pygame.transform.scale(pit_img, (field_size_x, field_size_y))

    gold_img = pygame.image.load("assets/gold_img.png").convert()
    gold_img = pygame.transform.scale(gold_img, (field_size_x, field_size_y))

    agent_img_left = pygame.image.load("assets/arrow_img.png").convert()
    agent_img_left = pygame.transform.scale(agent_img_left, (field_size_x, field_size_y))

    agent_img_down = pygame.image.load("assets/arrow_img.png").convert()
    agent_img_down = pygame.transform.scale(agent_img_down, (field_size_x, field_size_y))
    agent_img_down = pygame.transform.rotate(agent_img_down, 90)

    agent_img_right = pygame.image.load("assets/arrow_img.png").convert()
    agent_img_right = pygame.transform.scale(agent_img_right, (field_size_x, field_size_y))
    agent_img_right = pygame.transform.rotate(agent_img_right, 180)

    agent_img_up = pygame.image.load("assets/arrow_img.png").convert()
    agent_img_up = pygame.transform.scale(agent_img_up, (field_size_x, field_size_y))
    agent_img_up = pygame.transform.rotate(agent_img_up, 270)

    cave_entry_img = pygame.image.load("assets/cave_entry_img.png").convert()
    cave_entry_img = pygame.transform.scale(cave_entry_img, (field_size_x, field_size_y))

    msg = instruction_string

    agent = None
    # up
    if env.agent_direction == 0:
        agent = agent_img_up
    # right
    elif env.agent_direction == 1:
        agent = agent_img_right
    # down
    elif env.agent_direction == 2:
        agent = agent_img_down
    # left
    elif env.agent_direction == 3:
        agent = agent_img_left

    visited_rooms = [[0, 0, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0],
                     [1, 0, 0, 0]]

    if show_whole_map:
        for i in range(len(visited_rooms)):
            for j in range(len(visited_rooms[i])):
                visited_rooms[j][i] = 1

    draw_grid_world(env, screen, gold_img, wumpus_img, pit_img, agent, cave_entry_img,
                    field_size_x, field_size_y, msg, my_font, visited_rooms, sesnsed_danger)

    running = True
    total_reward = 0
    done = False
    # Main loop
    while running:
        action = None
        for event in pygame.event.get():
            if event.type == KEYDOWN:
                if (event.key == K_ESCAPE) | (event.key == K_q) | done:
                    running = False
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
                running = False

        if action is not None:
            new_state, reward, done, info, _ = env.step(action)
            visited_rooms[env.agentPosXY[0]][env.agentPosXY[1]] = 1
            total_reward += reward
            info = info.split(".")
            msg = instruction_string + [f"Reward this step: {reward}",
                                        f"Total reward: {total_reward}", f"Done: {done}", "Info:"] + info \
                  + ["The agent senses:"]
            sesnsed_danger = env.get_sensed_string()
            if not sesnsed_danger:
                msg += ["nothing"]
            else:
                msg += [sesnsed_danger]

            if done:
                msg += ["", "Game ended", "Press any key to leave"]

            agent = None
            # up
            if env.agent_direction == 0:
                agent = agent_img_up
            # right
            elif env.agent_direction == 1:
                agent = agent_img_right
            # down
            elif env.agent_direction == 2:
                agent = agent_img_down
            # left
            elif env.agent_direction == 3:
                agent = agent_img_left

            if done:
                for i in range(len(visited_rooms)):
                    for j in range(len(visited_rooms[i])):
                        visited_rooms[j][i] = 1

            draw_grid_world(env, screen, gold_img, wumpus_img, pit_img, agent, cave_entry_img,
                            field_size_x, field_size_y, msg, my_font, visited_rooms, sesnsed_danger)

    return
