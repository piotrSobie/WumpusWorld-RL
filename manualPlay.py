from wumpusEnvs.wumpusEnvLv1 import WumpusWorldLv1
from msvcrt import getch
import sys


def manual_play_lv1():
    env = WumpusWorldLv1()
    agent_state = env.reset_env()

    print("Initial environment")
    env.render_env()
    print(f"Agent state: {agent_state}\n")

    total_reward_manual = 0

    print("*****Instruction*****")
    print("q | ESC - terminate program")
    print("w - move up")
    print("s - move down")
    print("d - move right")
    print("a - move left")

    while True:
        key = ord(getch())
        action = None
        # q | ESC - terminate program
        if (key == 113) | (key == 27):
            print("q | ESC - terminate program")
            break
        # w - move up
        elif key == 119:
            print("w - move up")
            action = 0
        # s - move down
        elif key == 115:
            print("s - move down")
            action = 1
        # d - move right
        elif key == 100:
            print("d - move right")
            action = 2
        # a - move left
        elif key == 97:
            print("a - move left")
            action = 3

        if action is not None:
            new_state, reward, done, info, _ = env.step(action)
            env.render_env()
            total_reward_manual += reward
            print(f"Agent state: {new_state}")
            print(f"Reward this step: {reward}")
            print(f"Total reward: {total_reward_manual}")
            print(f"Done: {done}")
            print(f"Info {info}")
            print("-----------------------------")
            if done:
                print("*****************************")
                print(f"Game ended, total reward: {total_reward_manual}")
                print("Terminate program")
                print("*****************************")
                sys.exit()
        else:
            print("Invalid action")


def manual_play_lv2_plus(wumpus_env):
    env = wumpus_env
    agent_state = env.reset_env()

    print("Initial environment")
    env.render_env()
    print(f"Agent state: {agent_state}\n")

    total_reward_manual = 0

    print("*****Instruction*****")
    print("q | ESC - terminate program")
    print("w - move forward")
    print("a - turn left")
    print("d - turn right")
    print("g - take gold")
    print("z - shoot")
    print("c - climb out of the cave")

    while True:
        key = ord(getch())
        action = None
        # q | esc
        if (key == 113) | (key == 27):
            print("q | ESC - terminate program")
            break
        # w - move forward
        elif key == 119:
            print("w - move forward")
            action = 0
        # a - turn left
        elif key == 97:
            print("a - turn left")
            action = 1
        # d - turn right
        elif key == 100:
            print("d - turn right")
            action = 2
        # g - take gold
        elif key == 103:
            print("g - take gold")
            action = 3
        # z - shoot
        elif key == 122:
            print("z - shoot")
            action = 4
        # c - climb out of the cave
        elif key == 99:
            print("c - climb out of the cave")
            action = 5

        if action is not None:
            new_state, reward, done, info, _ = env.step(action)
            env.render_env()
            total_reward_manual += reward
            print(f"Agent state: {new_state}")
            print(f"Reward this step: {reward}")
            print(f"Total reward: {total_reward_manual}")
            print(f"Done: {done}")
            print(f"Info: {info}")
            print("-----------------------------")
            if done:
                print("*****************************")
                print(f"Game ended, total reward: {total_reward_manual}")
                print("Terminate program")
                print("*****************************")
                sys.exit()
        else:
            print("Invalid action")