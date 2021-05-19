from torch.utils.tensorboard import SummaryWriter
import numpy as np

from rl_alg.dqn.dqn_agent import DQNAgent
from rl_alg.dqn.utils import plot_learning


def test_agent_dqn(wumpus_env, load_state_path, random_grid=True, max_step=100):
    env = wumpus_env

    if env.dqn_observation_state_number is None:
        raise Exception("Environment not suitable for dqn")

    # # load agent from file
    agent = DQNAgent.load_agent(load_state_path, env.action_space_n, env.dqn_observation_state_number)

    # test
    # in testing i set epsilon to 0 and test on random grids
    agent.eps_strategy.epsilon = 0
    agent.eps_strategy.eps_min = 0
    scores_test, eps_history_test = [], []
    n_games = 100
    env.random_grid = random_grid
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
