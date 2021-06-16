from rl_base import Agent, Env
from gui.manual_pygame_agent import QuitException
from rl_alg.epsilon_greedy_strategy import EpsilonGreedyStrategy
from time import sleep
import numpy as np
import matplotlib.pyplot as plt
import os

from pygame_config import *


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


def episode(screen, env: Env, agent: Agent, max_ep_len, i_episode, auto=False, render=True, test_mode=False):
    state = env.reset_env()
    agent.reset_for_new_episode()

    observation = agent.observe(state)

    if agent.manual_action:
        auto = False

    if render:
        screen.fill(WHITE)
        instruction_string = [f"Episode {i_episode}", "Goal: step onto gold",
                              "Instruction:", "q | ESC - terminate program"]
        instruction_string += agent.get_instruction_string()
        msg = instruction_string
        if hasattr(agent, 'q_table'):
            env.render(screen, msg, agent.q_table)
        else:
            env.render(screen, msg)

    n_steps = 0
    running_episode = True
    total_reward = 0
    done = False
    last_episode = False
    # Main loop
    while running_episode:

        if render:
            key_pressed = False
            key_pressed, running_episode, last_episode, auto = key_logic(auto, done)
            if auto or agent.manual_action:
                sleep(0.05)
            else:
                while not key_pressed:
                    key_pressed, running_episode, last_episode, auto = key_logic(auto, done)
                    sleep(0.05)

        if not done:
            try:
                action = agent.choose_action(observation)
            except QuitException:
                return total_reward, n_steps, False, auto

            if action is not None:
                new_state, reward, done, info, _ = env.step(action)
                new_observation = agent.observe(new_state)
                if not test_mode:
                    agent.learn(observation, action, reward, new_observation, done)
                observation = new_observation
                total_reward += reward
                n_steps += 1
                if render:
                    msg = instruction_string + [f"Agent state:"]
                    msg += ['    '+s for s in str(new_state).split(';')]
                    msg += [f"Reward this step: {reward}", f"Total reward: {total_reward}",
                            f"Step: {n_steps}", f"Done: {done}"]
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
            if hasattr(agent, 'q_table'):
                env.render(screen, msg, agent.q_table)
            else:
                env.render(screen, msg)
        else:
            if done:
                running_episode = False
    return total_reward, n_steps, not last_episode, auto


def main_pygame2(env: Env, agent: Agent, max_ep_len=50, save_path=None, render=False,
                 num_episodes=1000, info_after_episodes=50, test_mode=False):

    if not isinstance(agent, Agent):
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
    best_average_rew = -np.inf
    best_single = -np.inf
    n_steps = []
    average_n_steps = []
    while running:
        tr, ns, running, auto_end = episode(screen, env, agent, max_ep_len, i_episode, auto_end, render, test_mode)
        if tr > best_single:
            best_single = tr
            print(f"In {i_episode} episode, new best total_reward: {tr:05f}, in {ns} steps!")
        total_rewards.append(tr)
        avr_rew = np.mean(total_rewards[-10:])
        average_rewards.append(avr_rew)
        n_steps.append(ns)
        avr_steps = np.mean(n_steps[-10:])
        average_n_steps.append(avr_steps)
        if avr_rew > best_average_rew:
            best_average_rew = avr_rew
            if save_path is not None and not test_mode:
                agent.save(save_path + '_best')
            print(f"After {i_episode} episodes, new best last 10 ep. avg rew: {avr_rew:05f}, avg steps/ep: {avr_steps:.2f}")
        i_episode += 1
        if i_episode % info_after_episodes == 0:
            if hasattr(agent, "second_eps_strategy"):
                print(f"After {i_episode} episodes. Last 10 avg total_rewards: {avr_rew:05f}, avg steps/ep: {avr_steps:.2f}, "
                      f"Eps={agent.first_eps_strategy.get_epsilon()}, Eps2={agent.second_eps_strategy.get_epsilon()}")
            else:
                print(f"After {i_episode} episodes. Last 10 avg total_rewards: {avr_rew:05f}, avg steps/ep: {avr_steps:.2f}, "
                      f"Eps={agent.action_selection_strategy.get_epsilon()}")
        if i_episode == num_episodes:
            break

    if len(average_rewards) > 10:
        plt.plot(average_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total reward')
        if save_path is not None and not test_mode:
            plt.savefig(os.path.dirname(save_path) + '/avg_rewards.png')
        plt.show()

    if save_path is not None and not test_mode:
        agent.save(save_path)

