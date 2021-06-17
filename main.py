import os
from gui.manual_pygame_agent import ManualPygameAgent
from envs.wumpus_env import WumpusWorld, wumpus_settings
from envs.frozen_lake import FrozenLake
from gui.main_pygame2 import main_pygame2
from rl_alg.dqn.dqn_default_params import DqnDefaultParams
from experiments.wumpuslv4_dqn_agent import FullSenseCentralizedMapDNNAgent, BasicWumpusQAgent, \
    FullSenseCentralizedMapCNNAgent
from experiments.frozen_lake_agents import FrozenLakeDQNAgent, FrozenLakeQAgent
from experiments.wumpus_agents_v2 import CentralizedMapDNNAgent, CentralizedMapCNNAgent
from glob import glob

import argparse
import time
import datetime

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
    parser.add_argument("--num_episodes", help=f"Number of learning episodes, default={num_episodes}", type=int)
    parser.add_argument("--max_steps_per_episode", help=f"Maximum number of steps per episode, "
                                                        f"default={max_steps_per_episode}", type=int)
    # parser.add_argument("--discount", help=f"Discount rate (gamma), should be in <0, 1>, default={discount_rate}", type=float)
    parser.add_argument("--lr", help=f"Learning rate, should be in <0, 1>, default={learning_rate}", type=float)
    parser.add_argument("--eps_start", help=f"Epsilon starting value, should be in <0, 1>, default={epsilon_start}", type=float)
    # parser.add_argument("--eps_decay", help=f"Epsilon decay rate, default={epsilon_decay}", type=float)
    # parser.add_argument("--eps_min", help=f"Epsilon min, should be in <0, 1>, default={epsilon_min}", type=float)
    # parser.add_argument("--batch_size", help=f"Used in DQN replay memory, default={batch_size}", type=int)
    # parser.add_argument("--target_update", help=f"Used in DQN, tells how often target network should be updated, "
    #                                             f"default={target_update}", type=int)
    parser.add_argument("--memory_size", help=f"Used in DQN, set replay memory size, default={memory_size}", type=int)
    parser.add_argument("--state_path", help=f"Loading state from /saved_models/PATH, PATH must be specified")
    parser.add_argument("--save_every", help=f"Saving checkpoint at specified frequency, default={save_every}", type=int)

    parser.add_argument("--show_whole_map", help=f"Show whole map in manual play, default={show_whole_map}", type=bool)
    # parser.add_argument("--show_actions_plot", help=f"Show plot with number of actions, default={show_actions_plot}", type=bool)
    # parser.add_argument("--show_reward_plot", help=f"Show plot with rewards, default={show_reward_plot}", type=bool)
    # parser.add_argument("--show_games_won_plot", help=f"Show plot with games won, default={show_games_won_plot}", type=bool)
    # parser.add_argument("--show_learned_path", help=f"Show learned path after learning, default={show_learned_path}", type=bool)
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

    if args.env in wumpus_settings:
        examined_env = WumpusWorld(settings=wumpus_settings[args.env], name=args.env)
    elif args.env == "lake":
        examined_env = FrozenLake()
    else:
        raise Exception(exception_msg)

    print(f"Env name: {examined_env.__class__.__name__}")
    print(f"Mode: {args.mode}")

    print(f"\nWaiting {time_for_reading_set_parameters} seconds...")
    time.sleep(time_for_reading_set_parameters)

    manual = args.mode.endswith("m")

    if args.env == "lake":
        if args.mode.startswith("dqn"):
            agent = FrozenLakeDQNAgent(16, 4, manual_action=manual)
        elif args.mode == "manual":
            agent = ManualPygameAgent()
        elif args.mode.startswith("q-learn"):
            agent = FrozenLakeQAgent(16, 4, manual_action=manual)
        else:
            raise NotImplementedError
    elif args.env in wumpus_settings:
        if args.mode.startswith("dqn"):
            agent = CentralizedMapDNNAgent(manual_action=manual)
        elif args.mode.startswith("cnn_dqn"):
             agent = CentralizedMapCNNAgent(manual_action=manual)
        elif args.mode == "manual":
            agent = ManualPygameAgent()
        elif args.mode.startswith("q-learn"):
            agent = BasicWumpusQAgent(manual_action=manual)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    date = 'run-{date:%Y-%m-%d_%H:%M:%S}'.format(date=datetime.datetime.now()).replace(':', '-')
    save_path_dir = '/'.join(['saved_models', examined_env.name, agent.name, date])
    save_path = save_path_dir + '/model'

    def get_prev_run_model(curr_dir):
        dirs = glob(os.path.dirname(curr_dir) + '/*')
        dirs.sort(reverse=True)
        return dirs[0] + '/model'

    if args.test_mode:
        if args.state_path is None or args.state_path == 'latest':
            state_path = get_prev_run_model(save_path_dir)
            print(f"Testing model from latest run.")
        else:
            state_path = args.state_path
    else:
        if args.state_path == 'latest':
            state_path = get_prev_run_model(save_path_dir)
        elif args.state_path:
            state_path = args.state_path

    if state_path is not None:
        print(f"Loading agent state from {state_path}")
        agent.load(state_path)

    if args.reset_eps:
        print(f"WARINING: reseting eps-greedy params to: eps_start={epsilon_start}, eps_decay={epsilon_decay}")
        agent.action_selection_strategy.epsilon = epsilon_start
        agent.action_selection_strategy.eps_dec = epsilon_decay
        agent.action_selection_strategy.eps_min = epsilon_min
        if hasattr(agent, "second_eps_strategy"):
            agent.second_eps_strategy.epsilon = 0.5
            agent.second_eps_strategy.eps_min = 0.01
            agent.second_eps_strategy.eps_dec = 1e-5

    if args.test_mode:
        agent.action_selection_strategy.epsilon = 0
        agent.action_selection_strategy.eps_min = 0
        agent.action_selection_strategy.eps_dec = 0
        if hasattr(agent, "second_eps_strategy"):
            agent.second_eps_strategy.epsilon = 0
            agent.second_eps_strategy.eps_min = 0
            agent.second_eps_strategy.eps_dec = 0
        print(f"TEST MODE, greedy action selection, eps={agent.action_selection_strategy.get_epsilon()}")
    else:
        os.makedirs(save_path_dir)

    if args.lr is not None:
        agent.lr = args.lr
        print(f"Forcing LR set to {agent.lr}")
    if args.eps_start is not None:
        agent.action_selection_strategy.epsilon = args.eps_start
        print(f"Forcing EPS_START set to {agent.action_selection_strategy.epsilon}")
    if args.num_episodes is not None:
        num_episodes = args.num_episodes

    main_pygame2(examined_env, agent, save_path=save_path, render=args.render,
                 num_episodes=num_episodes, test_mode=args.test_mode)
