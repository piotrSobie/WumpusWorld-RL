from rl_alg.q_agent import QAgent
from gui.manual_pygame_agent import QuitException
from rl_alg.epsilon_greedy_strategy import EpsilonGreedyStrategy
from time import sleep
import numpy as np
import matplotlib.pyplot as plt

import pygame
from pygame.locals import (
    KEYDOWN,
    K_ESCAPE,
    K_q,
    K_p,
    QUIT,
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


def key_logic(auto_mode, done):
    key_pressed = False
    running_episode = True
    last_episode = False
    for event in pygame.event.get():
        if event.type == QUIT:
            key_pressed = True
            running_episode = False
            last_episode = True
        elif event.type == KEYDOWN:
            key_pressed = True
            if event.key == K_ESCAPE or event.key == K_q:
                running_episode = False
                last_episode = True
            if event.key == K_p:
                auto_mode = not auto_mode
            if done:
                running_episode = False
    return key_pressed, running_episode, last_episode, auto_mode


def episode(screen, env, agent, max_ep_len, i_episode, auto=False, render=True):
    observation = env.reset_env()

    if agent.manual_action:
        auto = False

    if render:
        screen.fill(white)
        instruction_string = [f"Episode {i_episode}","Goal: step onto gold",
                              "Instruction:", "q | ESC - terminate program"]
        if agent.manual_action:
            instruction_string += ["w - move up", "a - move left", "d - move right", "s - move down"]
        if not auto:
            instruction_string += ["Press any key"]
        msg = instruction_string
        env.render(screen, msg, agent.q_table)

    n_steps = 0
    running_episode = True
    total_reward = 0
    done = False
    last_episode = False
    # Main loop
    while running_episode:
        if not done:
            try:
                action = agent.choose_action(observation)
            except QuitException:
                return total_reward, n_steps, False, auto

            if action is not None:
                new_state, reward, done, info, _ = env.step(action)
                agent.learn(observation, action, reward, new_state, done)
                observation = new_state
                total_reward += reward
                n_steps += 1
                if render:
                    info = info.split(".")
                    msg = instruction_string + [f"Agent state: {new_state}", f"Reward this step: {reward}",
                                                f"Total reward: {total_reward}", f"Step: {n_steps}", f"Done: {done}"]
                    if not agent.manual_action and isinstance(agent.action_selection_strategy, EpsilonGreedyStrategy):
                        msg += [f"Epsilon: {agent.action_selection_strategy.get_epsilon():.4f}"]
                    msg += ["Info:"]
                    msg += info
                if n_steps >= max_ep_len:
                    done = True
        else:  # done
            if auto:
                break
            if render and 'end_msg' not in locals():
                end_msg = msg + ["", "Episode ended", "Press esc/q to exit or", "any other kay to start a new episode."]
                msg = end_msg

        if render:
            key_pressed = False
            key_pressed, running_episode, last_episode, auto = key_logic(auto, done)
            if auto or agent.manual_action:
                sleep(0.05)
            else:
                while not key_pressed:
                    key_pressed, running_episode, last_episode, auto = key_logic(auto, done)
                    sleep(0.05)
            env.render(screen, msg, agent.q_table)
        else:
            if done:
                running_episode = False
    return total_reward, n_steps, not last_episode, auto


def main_pygame2(env, agent, max_ep_len=100, save_path=None, render=False):

    if not isinstance(agent, QAgent):
        raise ValueError('Unsupported agent type.')

    if render:
        # Initialize pygame
        pygame.init()
        pygame.font.init()
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        auto_end = False
    else:
        screen = None
        auto_end = True

    i_episode = 1
    running = True
    total_rewards = []
    average_rewards = []
    n_steps = []
    while running:
        tr, ns, running, auto_end = episode(screen, env, agent, max_ep_len, i_episode, auto_end, render)
        total_rewards.append(tr)
        average_rewards.append(np.mean(total_rewards[-10:]))
        n_steps.append(ns)
        i_episode += 1
        if i_episode % 10 == 0:
            print(f"After {i_episode} episodes. Last 10 avg total_rewards: {average_rewards[-1]}")
        if i_episode == 1000:
            break

    if len(average_rewards) > 10:
        plt.plot(average_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total reward')
        plt.show()

    if save_path is not None:
        agent.save_q_table(save_path)
