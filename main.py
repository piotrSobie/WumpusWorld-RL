from wumpus_envs.wumpus_env_lv1 import WumpusWorldLv1
from wumpus_envs.wumpus_env_lv2 import WumpusWorldLv2
from wumpus_envs.wumpus_env_lv3v1 import WumpusWorldLv3v1
from wumpus_envs.wumpus_env_lv3v2 import WumpusWorldLv3v2
from wumpus_envs.wumpus_env_lv3v3 import WumpusWorldLv3v3
from wumpus_envs.wumpus_env_lv4 import WumpusWorldLv4

from manual_play.manual_play_cmd import manual_play_lv1, manual_play_lv2_plus
from manual_play.manual_play_pygame import manual_play_pygame_lv1, manual_play_pygame_lv2_plus
from rl_alg.q_learn import q_learn
from rl_alg.dqn.dqn_alg import dqn_algorithm

import argparse
import time

if __name__ == '__main__':
    # default values
    # q-learn & dqn hyper parameters
    examined_env = None
    mode = None
    num_episodes = 10000
    max_steps_per_episode = 100
    learning_rate = 0.1
    discount_rate = 0.9
    epsilon_start = 1
    epsilon_decay = 0.001
    epsilon_min = 0.01
    # q-learn only
    show_actions_plot = True
    show_reward_plot = True
    show_games_won_plot = True
    show_learned_path = True
    # dqn only
    batch_size = 256
    target_update = 10
    memory_size = 100000

    time_for_reading_set_parameters = 3

    parser = argparse.ArgumentParser()
    # required arguments
    required_args = parser.add_argument_group("required named arguments")
    required_args.add_argument("--env", help="Required, choose environment, possible values:"
                                      " lv1, lv2, lv3v1, lv3v2, lv3v3, lv4", required=True)
    required_args.add_argument("--mode", help="Required, choose mode, possible values: manual, manual-cmd q-learn, dqn",
                               required=True)
    # optional arguments
    parser.add_argument("--num_episodes", help=f"Number of learning episodes, default={num_episodes}")
    parser.add_argument("--max_steps_per_episode", help=f"Maximum number of steps per episode, "
                                                        f"default={max_steps_per_episode}")
    parser.add_argument("--lr", help=f"Learning rate, should be in <0, 1>, default={learning_rate}")
    parser.add_argument("--discount", help=f"Discount rate (gamma), should be in <0, 1>, default={discount_rate}")
    parser.add_argument("--eps_start", help=f"Epsilon starting value, should be in <0, 1>, default={epsilon_start}")
    parser.add_argument("--eps_decay", help=f"Epsilon decay rate, default={epsilon_decay}")
    parser.add_argument("--eps_min", help=f"Epsilon min, should be in <0, 1>, default={epsilon_min}")
    parser.add_argument("--show_actions_plot", help=f"Show plot with number of actions, default={show_actions_plot}")
    parser.add_argument("--show_reward_plot", help=f"Show plot with rewards, default={show_reward_plot}")
    parser.add_argument("--show_games_won_plot", help=f"Show plot with games won, default={show_games_won_plot}")
    parser.add_argument("--show_learned_path", help=f"Show learned path after learning, default={show_learned_path}")
    parser.add_argument("--batch_size", help=f"Used in DQN replay memory, default={batch_size}")
    parser.add_argument("--target_update", help=f"Used in DQN, tells how often target network should be updated, "
                                                f"default={target_update}")
    parser.add_argument("--memory_size", help=f"Used in DQN, set replay memory size, default={memory_size}")

    args = parser.parse_args()
    exception_msg = "Invalid environment, try python main.py --help"

    if args.env == "lv1":
        examined_env = WumpusWorldLv1()
    elif args.env == "lv2":
        examined_env = WumpusWorldLv2()
    elif args.env == "lv3v1":
        examined_env = WumpusWorldLv3v1()
    elif args.env == "lv3v2":
        examined_env = WumpusWorldLv3v2()
    elif args.env == "lv3v3":
        examined_env = WumpusWorldLv3v3()
    elif args.env == "lv4":
        examined_env = WumpusWorldLv4()
    else:
        raise Exception(exception_msg)

    if args.num_episodes is not None:
        num_episodes = int(args.num_episodes)

    if args.max_steps_per_episode is not None:
        max_steps_per_episode = int(args.max_steps_per_episode)

    if args.lr is not None:
        learning_rate = float(args.lr)

    if args.discount is not None:
        discount_rate = float(args.discount)

    if args.eps_start is not None:
        epsilon_start = float(args.eps_start)

    if args.eps_decay is not None:
        epsilon_decay = float(args.eps_decay)

    if args.eps_min is not None:
        epsilon_min = float(args.eps_min)

    if args.show_actions_plot is not None:
        if args.show_actions_plot.lower() == "true":
            show_actions_plot = True
        elif args.show_actions_plot.lower() == "false":
            show_actions_plot = False
        else:
            show_actions_plot = bool(int(args.show_actions_plot))

    if args.show_reward_plot is not None:
        if args.show_reward_plot.lower() == "true":
            show_reward_plot = True
        elif args.show_reward_plot.lower() == "false":
            show_reward_plot = False
        else:
            show_reward_plot = bool(int(args.show_reward_plot))

    if args.show_games_won_plot is not None:
        if args.show_games_won_plot.lower() == "true":
            show_games_won_plot = True
        elif args.show_games_won_plot.lower() == "false":
            show_games_won_plot = False
        else:
            show_games_won_plot = bool(int(args.show_games_won_plot))

    if args.show_learned_path is not None:
        if args.show_learned_path.lower() == "true":
            show_learned_path = True
        elif args.show_learned_path.lower() == "false":
            show_learned_path = False
        else:
            show_learned_path = bool(int(args.show_learned_path))

    if args.batch_size is not None:
        batch_size = int(args.batch_size)

    if args.target_update is not None:
        target_update = int(args.target_update)

    if args.memory_size is not None:
        memory_size = int(args.memory_size)

    if args.mode == "manual":
        mode = "manual"
    elif args.mode == "manual-cmd":
        mode = "manual-cmd"
    elif args.mode == "q-learn":
        mode = "q-learn"
    elif args.mode == "dqn":
        mode = "dqn"
    else:
        raise Exception(exception_msg)

    print("\nStarting with following parameters:")
    print("\nQ-learning & DQN parameters:")
    print(f"Env name: {examined_env.__class__.__name__}")
    print(f"Mode: {mode}")
    print(f"Num episodes: {num_episodes}")
    print(f"Max steps per episodes: {max_steps_per_episode}")
    print(f"Learning rate: {learning_rate}")
    print(f"Discount rate (gamma): {discount_rate}")
    print(f"Epsilon start: {epsilon_start}")
    print(f"Epsilon decay: {epsilon_decay}")
    print(f"Epsilon min: {epsilon_min}")
    print("\nQ-learning only parameters:")
    print(f"Show number actions plot: {show_actions_plot}")
    print(f"Show reward plot: {show_reward_plot}")
    print(f"Show games won plot: {show_games_won_plot}")
    print(f"Show learned path: {show_learned_path}")
    print("\nDQN only parameters:")
    print(f"Batch size: {batch_size}")
    print(f"Target update: {target_update}")
    print(f"Memory size: {memory_size}")
    print(f"\nWaiting {time_for_reading_set_parameters} seconds...")
    time.sleep(time_for_reading_set_parameters)

    if mode == "manual":
        if args.env == "lv1":
            manual_play_pygame_lv1()
        else:
            manual_play_pygame_lv2_plus(examined_env)
    elif mode == "manual-cmd":
        if args.env == "lv1":
            manual_play_lv1()
        else:
            manual_play_lv2_plus(examined_env)
    elif mode == "q-learn":
        q_learn(examined_env, num_episodes_=num_episodes, max_step_per_episode_=max_steps_per_episode,
                learning_rate_=learning_rate, discount_rate_=discount_rate, start_exploration_rate=epsilon_start,
                max_exploration_rate_=epsilon_start, min_exploration_rate_=epsilon_min,
                exploration_decay_rate_=epsilon_decay, show_number_actions_plot=show_actions_plot,
                show_reward_plot=show_reward_plot, show_games_won_plot=show_games_won_plot,
                show_learned_path=show_learned_path)
    elif mode == "dqn":
        dqn_algorithm(examined_env, batch_size_=batch_size, gamma_=discount_rate, eps_start_=epsilon_start,
                      eps_end_=epsilon_min, eps_decay_=epsilon_decay, target_update_=target_update,
                      memory_size_=memory_size, lr_=learning_rate, num_episodes_=num_episodes,
                      max_steps_per_episode_=max_steps_per_episode)

    # manual_play_lv1()

    # examined_env = WumpusWorldLv1()
    # examined_env = WumpusWorldLv2()
    # examined_env = WumpusWorldLv3v1()
    # examined_env = WumpusWorldLv3v2()
    # examined_env = WumpusWorldLv3v3()
    # examined_env = WumpusWorldLv4()

    # manual_play_lv2_plus(examined_env)
    # q_learn(examined_env, show_learned_path=False)
    # dqn_algorithm(examined_env)
