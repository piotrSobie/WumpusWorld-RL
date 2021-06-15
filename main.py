from gui.manual_pygame_agent import ManualPygameAgent, Lv1ManualPygameAgent, TurningManualControl,\
    SimpleManualControl
from rl_alg.dqn.dqn_agent import DQNAgent
from wumpus_envs.wumpus_env_lv1 import WumpusWorldLv1
from wumpus_envs.wumpus_env_lv2 import WumpusWorldLv2
from wumpus_envs.wumpus_env_lv3v1 import WumpusWorldLv3v1
from wumpus_envs.wumpus_env_lv3v2 import WumpusWorldLv3v2
from wumpus_envs.wumpus_env_lv3v3 import WumpusWorldLv3v3
from wumpus_envs.wumpus_env_lv4 import WumpusWorldLv4
from wumpus_envs.wumpus_env_lv4_a import WumpusWorldLv4a
from envs.frozen_lake import FrozenLake

from manual_play.manual_play_cmd import manual_play_lv1, manual_play_lv2_plus
from gui.main_pygame import main_pygame
from gui.main_pygame2 import main_pygame2
from rl_alg.q_learn import q_learn
from rl_alg.dqn.dqn_alg import dqn_algorithm
from rl_alg.dqn.test_agent import test_agent_dqn
from rl_alg.dqn.dqn_default_params import DqnDefaultParams
from rl_alg.q_agent import QAgent
from rl_alg.dqn.dqn_agent import DQNAgent
from rl_alg.dqn.dqn_network import DeepQNetwork
from experiments.wumpuslv4_dqn_agent import WumpusSenseMapStaticDQN, BasicWumpusQAgent, WumpusBasicStaticWorldDQN

import argparse
import time
import sys

if __name__ == '__main__':
    # default values
    default_params = DqnDefaultParams()

    # q-learn & dqn hyper parameters
    examined_env = None
    mode = None

    num_episodes = default_params.NUM_EPISODES
    max_steps_per_episode = default_params.MAX_STEPS_PER_EPISODE
    learning_rate = default_params.LEARNING_RATE
    discount_rate = default_params.DISCOUNT_RATE
    epsilon_start = default_params.EPSILON_START
    epsilon_decay = default_params.EPSILON_DECAY
    epsilon_min = default_params.EPSILON_MIN

    # q-learn only
    show_actions_plot = True
    show_reward_plot = True
    show_games_won_plot = True
    show_learned_path = True

    # dqn only
    batch_size = default_params.BATCH_SIZE
    target_update = default_params.TARGET_UPDATE
    memory_size = default_params.MEMORY_SIZE
    save_every = default_params.SAVE_EVERY
    state_path = None

    # gui only
    show_whole_map = False

    time_for_reading_set_parameters = 1

    parser = argparse.ArgumentParser()
    # required arguments
    required_args = parser.add_argument_group("required named arguments")
    required_args.add_argument("--env", help="Required, choose environment, possible values:"
                                      " lv1, lv2, lv3v1, lv3v2, lv3v3, lv4", required=True)
    required_args.add_argument("--mode", help="Required, choose mode, possible values: manual, manual-cmd q-learn, "
                                              "dqn, test (with test --state_path must be specified),"
                                              "test-gui", required=True)
    # optional arguments
    parser.add_argument("--num_episodes", help=f"Number of learning episodes, default={num_episodes}", default=num_episodes, type=int)
    parser.add_argument("--max_steps_per_episode", help=f"Maximum number of steps per episode, "
                                                        f"default={max_steps_per_episode}", default=max_steps_per_episode)
    parser.add_argument("--lr", help=f"Learning rate, should be in <0, 1>, default={learning_rate}", default=learning_rate)
    parser.add_argument("--discount", help=f"Discount rate (gamma), should be in <0, 1>, default={discount_rate}", default=discount_rate)
    parser.add_argument("--eps_start", help=f"Epsilon starting value, should be in <0, 1>, default={epsilon_start}", default=epsilon_start, type=float)
    parser.add_argument("--eps_decay", help=f"Epsilon decay rate, default={epsilon_decay}", default=epsilon_decay, type=float)
    parser.add_argument("--eps_min", help=f"Epsilon min, should be in <0, 1>, default={epsilon_min}", default=epsilon_min, type=float)
    parser.add_argument("--show_whole_map", help=f"Show whole map in manual play, default={show_whole_map}", default=show_whole_map, type=bool)
    parser.add_argument("--show_actions_plot", help=f"Show plot with number of actions, default={show_actions_plot}", default=show_actions_plot, type=bool)
    parser.add_argument("--show_reward_plot", help=f"Show plot with rewards, default={show_reward_plot}", default=show_reward_plot, type=bool)
    parser.add_argument("--show_games_won_plot", help=f"Show plot with games won, default={show_games_won_plot}", default=show_games_won_plot, type=bool)
    parser.add_argument("--show_learned_path", help=f"Show learned path after learning, default={show_learned_path}", default=show_learned_path, type=bool)
    parser.add_argument("--batch_size", help=f"Used in DQN replay memory, default={batch_size}", default=batch_size)
    parser.add_argument("--target_update", help=f"Used in DQN, tells how often target network should be updated, "
                                                f"default={target_update}", default=target_update)
    parser.add_argument("--memory_size", help=f"Used in DQN, set replay memory size, default={memory_size}", default=memory_size)
    parser.add_argument("--state_path", help=f"Loading state from /saved_models/PATH, PATH must be specified")
    parser.add_argument("--save_every", help=f"Saving checkpoint at specified frequency, default={save_every}", default=save_every)
    parser.add_argument("--no_render", help=f"Whether to display pygame", dest='render', action='store_false')
    parser.add_argument("--no_random_grid", help=f"Whether grid is random or not", dest='random_grid', action='store_false')
    parser.add_argument("--test", help=f"Whether this is test mode", dest='test_mode', action='store_true')
    parser.add_argument("--reset_eps", help=f"Whether this eps-greedy strategy should start from {epsilon_start} again.",
                        dest='reset_eps', action='store_true')

    parser.set_defaults(render=True)
    parser.set_defaults(random_grid=True)
    parser.set_defaults(test_mode=False)
    parser.set_defaults(reset_eps=False)

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
        examined_env = WumpusWorldLv4a(args.random_grid)
    elif args.env == "lake":
        examined_env = FrozenLake()
    else:
        raise Exception(exception_msg)

    num_episodes = args.num_episodes
    max_steps_per_episode = args.max_steps_per_episode
    learning_rate = args.lr
    discount_rate = args.discount
    epsilon_start = args.eps_start
    epsilon_decay = args.eps_decay
    epsilon_min = args.eps_min
    batch_size = args.batch_size
    target_update = args.target_update
    memory_size = args.memory_size
    save_every = args.save_every
    show_whole_map = args.show_whole_map
    show_actions_plot = args.show_actions_plot
    show_learned_path = args.show_learned_path
    show_reward_plot = args.show_reward_plot
    show_games_won_plot = args.show_games_won_plot

    if args.batch_size is not None:
        batch_size = int(args.batch_size)

    if args.target_update is not None:
        target_update = int(args.target_update)

    if args.memory_size is not None:
        memory_size = int(args.memory_size)

    if args.save_every is not None:
        save_every = float(args.save_every)

    # print("\nSet parameters:")
    # print("\nQ-learning & DQN parameters:")
    print(f"Env name: {examined_env.__class__.__name__}")
    print(f"Mode: {args.mode}")
    # print(f"Num episodes: {num_episodes}")
    # print(f"Max steps per episodes: {max_steps_per_episode}")
    # print(f"Learning rate: {learning_rate}")
    # print(f"Discount rate (gamma): {discount_rate}")
    # print(f"Epsilon start: {epsilon_start}")
    # print(f"Epsilon decay: {epsilon_decay}")
    # print(f"Epsilon min: {epsilon_min}")
    # print("\nQ-learning only parameters:")
    # print(f"Show number actions plot: {show_actions_plot}")
    # print(f"Show reward plot: {show_reward_plot}")
    # print(f"Show games won plot: {show_games_won_plot}")
    # print(f"Show learned path: {show_learned_path}")
    # print("\nDQN only parameters:")
    # print(f"Batch size: {batch_size}")
    # print(f"Target update: {target_update}")
    # print(f"Memory size: {memory_size}")
    if args.state_path is not None:
        state_path = args.state_path
    print(f"Load state path: {state_path}")
    print(f"\nWaiting {time_for_reading_set_parameters} seconds...")
    time.sleep(time_for_reading_set_parameters)

    if args.env == "lake":
        if args.mode == "dqn":
            import numpy as np

            class FrozenLakeDQNAgent(DQNAgent):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.eye = np.eye(16, dtype=np.float32)

                def from_state_to_input_vector(self, state):
                    return self.eye[state]

                def get_network(self):
                    return DeepQNetwork(self.lr, n_actions=self.n_actions,
                                        input_dims=self.input_dims, fc1_dims=10, fc2_dims=10)

            agent = FrozenLakeDQNAgent(16, 4)
            save_path = 'saved_models/dqn_agent'
        elif args.mode == "manual":
            agent = ManualPygameAgent()
            save_path = None
        elif args.mode.startswith("q-learn"):
            class FrozenLakeQAgent(QAgent):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.manual_control = SimpleManualControl()

                def from_state_to_idx(self, state):
                    return state
            manual = args.mode.endswith("m")
            agent = FrozenLakeQAgent(16, 4, manual_action=manual)
            save_path = 'saved_models/q_table'
        else:
            raise NotImplementedError

        if args.test_mode:
            state_path = save_path
            if args.mode.startswith("q-learn"):
                state_path += '.npy'

        if state_path is not None:
            print(f"Loading agent state from {state_path}")
            agent.load(state_path)

        main_pygame2(examined_env, agent, save_path=save_path, render=args.render,
                     num_episodes=num_episodes, test_mode=args.test_mode)

    elif args.env == "lv4":
        if args.mode.startswith("dqn"):
            manual = args.mode.endswith("m")
            # agent = WumpusBasicStaticWorldDQN()
            agent = WumpusSenseMapStaticDQN(manual_action=manual)
            save_path = 'saved_models/dqn_agent_lv5'
        elif args.mode == "manual":
            agent = ManualPygameAgent()
            save_path = None
        elif args.mode.startswith("q-learn"):
            manual = args.mode.endswith("m")
            save_path = 'saved_models/q_table_lv4'
            agent = BasicWumpusQAgent(manual_action=manual)
        else:
            raise NotImplementedError

        if args.test_mode:
            state_path = save_path
            if args.mode.startswith("q-learn"):
                state_path += '.npy'

        if state_path is not None:
            print(f"Loading agent state from {state_path}")
            agent.load(state_path)

        if args.reset_eps:
            print(f"WARINING: reseting eps-greedy params to: eps_start={epsilon_start}, eps_decay={epsilon_decay}")
            agent.action_selection_strategy.epsilon = epsilon_start
            agent.action_selection_strategy.eps_dec = epsilon_decay
            agent.action_selection_strategy.eps_min = epsilon_min

        main_pygame2(examined_env, agent, save_path=save_path, render=args.render,
                     num_episodes=num_episodes, test_mode=args.test_mode)

    elif mode == "manual":
        if args.env == "lv1" or args.env == "lake":
            agent = Lv1ManualPygameAgent()
        else:
            agent = ManualPygameAgent()
        main_pygame(examined_env, agent, show_whole_map)
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
                      max_steps_per_episode_=max_steps_per_episode, save_weights_every_=save_every,
                      load_state_path=state_path)
    elif mode == "test" or mode == "test-gui":
        if state_path is None:
            print("State path must be specified in this mode")
            sys.exit()

        if mode == "test":
            test_agent_dqn(examined_env, load_state_path=state_path)
        else:   # test-gui
            # # load agent from file
            agent = DQNAgent.load(state_path, examined_env.action_space_n,
                                  examined_env.dqn_observation_state_number)
            main_pygame(examined_env, agent, show_whole_map)
