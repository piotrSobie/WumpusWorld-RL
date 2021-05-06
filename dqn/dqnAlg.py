import numpy as np
from torch.utils.tensorboard import SummaryWriter
from itertools import product

from dqn.agent import Agent
from dqn.utils import plot_learning
from wumpusEnvs.wumpusEnvLv4 import WumpusWorldLv4


def dqn_algorithm(wumpus_env, batch_size_=64, gamma_=0.999, eps_start_=1, eps_end_=0.01, eps_decay_=5e-5,
                  target_update_=100, memory_size_=100000, lr_=0.001, num_episodes_=500, max_steps_per_episode_=200):

    env = WumpusWorldLv4()
    print("dqn start")

    env.reset_env()
    env_action_n = env.action_space_n
    env_obs_n = 85
    n_games = num_episodes_
    max_step = max_steps_per_episode_

    parameters = dict(
        gamma=[gamma_],
        epsilon=[eps_start_],
        eps_dec=[eps_decay_],
        eps_end=[eps_end_],
        batch_size=[batch_size_],
        lr=[lr_],
        max_mem_size=[memory_size_],
        replace_target=[target_update_]
    )

    param_values = [v for v in parameters.values()]
    loop_nr = 1
    max_nr_of_loops = 1
    for e in param_values:
        max_nr_of_loops *= len(e)

    for gamma, epsilon, eps_dec, eps_end, batch_size, lr, max_mem_size, replace_target in product(*param_values):
        agent = Agent(gamma=gamma, epsilon=epsilon, eps_dec=eps_dec, eps_end=eps_end, batch_size=batch_size,
                      lr=lr, max_mem_size=max_mem_size, replace_target=replace_target,
                      n_actions=env_action_n, input_dims=env_obs_n)

        scores, eps_history = [], []
        comment = f"batch_size={batch_size}, lr={lr}"
        tb = SummaryWriter(comment=comment)

        # learn
        for episode in range(n_games):
            score = 0
            done = False
            observation = env.reset_env()
            current_step = 0

            while not done:
                action = agent.choose_action(observation)
                new_observation, reward, done, _, _ = env.step(action)
                score += reward
                agent.memory.store_transitions(observation, action, reward, new_observation, done)
                agent.learn()
                observation = new_observation

                current_step += 1

                if current_step > max_step:
                    break

                # to compare loss with different batch_size
                # total_loss = loss * batch_size

            scores.append(score)
            eps_history.append(agent.eps_strategy.get_epsilon())

            avg_score = np.mean(scores[-100:])

            tb.add_scalar('Score', score, episode)
            tb.add_scalar('Avg score', avg_score, episode)
            tb.add_scalar('Epsilon', agent.eps_strategy.get_epsilon(), episode)

            for name, weight in agent.Q_eval.named_parameters():
                tb.add_histogram(name, weight, episode)
                # tb.add_histogram(f"{name}.grad", weight.grad, episode)

            print("works")
            print(f"{loop_nr} in {max_nr_of_loops}")
            print(f"Examined values: {comment}")
            print(
                f"Episode {episode}, score {score}, average score {avg_score},"
                f" epsilon {agent.eps_strategy.get_epsilon()}\n")

        x = [i + 1 for i in range(n_games)]
        filename = 'frozen_lake_' + comment + ".png"
        plot_learning(x, scores, eps_history, filename)

        # test
        agent.eps_strategy.epsilon = 0
        agent.eps_strategy.eps_min = 0
        scores_test, eps_history_test = [], []
        n_games = 100
        env.random_grid = True

        for i in range(n_games):
            score = 0
            done = False
            observation = env.reset_env()
            current_step = 0

            while not done:
                action = agent.choose_action(observation)
                new_observation, reward, done, _, _ = env.step(action)
                score += reward
                # agent.memory.store_transitions(observation, action, reward, new_observation, done)
                # agent.learn()
                observation = new_observation

                if current_step > max_step:
                    break

            scores_test.append(score)
            eps_history_test.append(agent.eps_strategy.get_epsilon())

            avg_score = np.mean(scores_test[-100:])

            print(f"Test {i}, score {score}, average score {avg_score}, epsilon {agent.eps_strategy.get_epsilon()}")

        x = [i + 1 for i in range(n_games)]
        filename = 'lunar_lander_test.png'
        plot_learning(x, scores_test, eps_history_test, filename)

        tb.close()

        loop_nr += 1
