import torch as T
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from rl_alg.dqn.agent import Agent
from rl_alg.dqn.utils import plot_learning


def test_agent_dqn(wumpus_env, load_state_path):
    env = wumpus_env

    if env.dqn_observation_state_number is None:
        raise Exception("Environment not suitable for dqn")

    models_directory = 'saved_models'
    loaded_state = T.load(f"{models_directory}/{load_state_path}")

    # load data from file if specified
    start_episode = loaded_state['episode']
    batch_size_dqn = loaded_state['batch_size']
    gamma_dqn = loaded_state['gamma']
    eps_start_dqn = loaded_state['epsilon']
    eps_end_dqn = loaded_state['eps_min']
    eps_decay_dqn = loaded_state['eps_dec']
    target_update_dqn = loaded_state['replace_target']
    lr_dqn = loaded_state['lr']
    num_episodes_dqn = loaded_state['num_episodes']
    max_steps_per_episode_dqn = loaded_state['max_step_per_episode']
    save_weights_every_dqn = loaded_state['save_weights_every']
    replay_memory_dqn = loaded_state['replay_memory']
    net_state_dict_dqn = loaded_state['state_dict']
    optimizer_state_dict_dqn = loaded_state['optimizer']

    env_action_n = env.action_space_n
    env_obs_n = env.dqn_observation_state_number
    agent = Agent(n_actions=env_action_n, input_dims=env_obs_n, gamma=gamma_dqn, epsilon=0.0, lr=lr_dqn,
                  batch_size=batch_size_dqn, eps_end=0.0, eps_dec=0.0, replace_target=target_update_dqn,
                  replay_memory=replay_memory_dqn, net_state_dict=net_state_dict_dqn,
                  optimizer_state_dict=optimizer_state_dict_dqn)
    max_step = loaded_state['max_step_per_episode']

    # test
    # in testing i set epsilon to 0 and test on random grids
    agent.eps_strategy.epsilon = 0
    agent.eps_strategy.eps_min = 0
    scores_test, eps_history_test = [], []
    n_games = 100
    env.random_grid = True
    comment = f"testing, batch_size={agent.batch_size}, lr={agent.lr}, random_grid={env.random_grid}"
    tb_test = SummaryWriter(comment=comment)
    #
    for test_episode in range(n_games):
        score = 0
        done = False
        observation = env.reset_env()
        current_step = 0

        while not done:
            action = agent.choose_action(observation)
            new_observation, reward, done, _, _ = env.step(action)
            score += reward
            observation = new_observation

            current_step += 1

            if current_step > max_step:
                break

        scores_test.append(score)
        eps_history_test.append(agent.eps_strategy.get_epsilon())

        avg_score = np.mean(scores_test[-100:])

        tb_test.add_scalar('Score in testing', score, test_episode)
        tb_test.add_scalar('Avg score in testing', avg_score, test_episode)

        print(f"Test {test_episode}, score {score}, average score {avg_score}, "
              f"epsilon {agent.eps_strategy.get_epsilon()}\n")

    x = [i + 1 for i in range(n_games)]
    filename = 'wumpus_test' + comment + '.png'
    plot_learning(x, scores_test, eps_history_test, filename)

    tb_test.close()
    return
