import numpy as np
import torch as T
from torch.utils.tensorboard import SummaryWriter
from itertools import product
import datetime
import os

from rl_alg.dqn.dqn_agent import Agent
from rl_alg.dqn.utils import plot_learning


def dqn_algorithm(wumpus_env, batch_size_=64, gamma_=0.999, eps_start_=1, eps_end_=0.01, eps_decay_=5e-5,
                  target_update_=100, memory_size_=100000, lr_=0.001, num_episodes_=500, max_steps_per_episode_=200,
                  save_weights_every_=50, load_state_path=None):

    env = wumpus_env
    if env.dqn_observation_state_number is None:
        raise Exception("Environment not suitable for dqn")

    models_directory = 'saved_models'
    now = datetime.datetime.now()
    start_date = f"start_learning_{now.year}-{now.month}-{now.day}_{now.hour}-{now.minute}-{now.second}"
    loaded_state = None

    if load_state_path is not None:
        start_date = load_state_path.split('/')[0]
        if start_date == "":
            start_date = load_state_path.split('/')[1]
        loaded_state = T.load(f"{models_directory}/{load_state_path}")

    if not os.path.exists(models_directory):
        os.mkdir(models_directory)

    if not os.path.exists(f"{models_directory}/{start_date}"):
        os.mkdir(f"{models_directory}/{start_date}")

    env.reset_env()
    env_action_n = env.action_space_n
    env_obs_n = env.dqn_observation_state_number
    n_games = num_episodes_
    max_step = max_steps_per_episode_
    start_episode = 0
    save_weights_every = save_weights_every_

    if loaded_state is not None:
        start_episode = loaded_state['episode']
        n_games = loaded_state['num_episodes']
        max_step = loaded_state['max_step_per_episode']
        save_weights_every = loaded_state['save_weights_every']

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
    loop_nr_hyper_parameters_variations = 1
    max_nr_of_loops_hyper_parameters_variations = 1
    for e in param_values:
        max_nr_of_loops_hyper_parameters_variations *= len(e)

    for gamma, epsilon, eps_dec, eps_end, batch_size, lr, max_mem_size, replace_target in product(*param_values):
        agent = Agent(gamma=gamma, epsilon=epsilon, eps_dec=eps_dec, eps_end=eps_end, batch_size=batch_size,
                      lr=lr, max_mem_size=max_mem_size, replace_target=replace_target,
                      n_actions=env_action_n, input_dims=env_obs_n, loaded_state=loaded_state)

        scores, eps_history = [], []
        best_avg_score = None

        # learn
        comment = f"learning, batch_size={batch_size}, lr={lr}, random_grid={env.random_grid}"
        tb_train = SummaryWriter(comment=comment)
        for episode in range(start_episode, n_games):
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

                # in the future maybe
                # to compare loss with different batch_size
                # total_loss = loss * batch_size

            scores.append(score)
            eps_history.append(agent.eps_strategy.get_epsilon())

            avg_score = np.mean(scores[-100:])
            if best_avg_score is None:
                best_avg_score = avg_score

            tb_train.add_scalar('Score in training', score, episode)
            tb_train.add_scalar('Avg score in training', avg_score, episode)
            tb_train.add_scalar('Epsilon in training', agent.eps_strategy.get_epsilon(), episode)

            for name, weight in agent.Q_eval.named_parameters():
                tb_train.add_histogram(name, weight, episode)
                # tb.add_histogram(f"{name}.grad", weight.grad, episode)

            print(f"Hyper parameters variations {loop_nr_hyper_parameters_variations} in "
                  f"{max_nr_of_loops_hyper_parameters_variations}")
            print(f"Examined values: {comment}")
            print(
                f"Episode {episode} in {n_games}, score {score}, average score {avg_score},"
                f" epsilon {agent.eps_strategy.get_epsilon()}")

            # saving state
            if (episode % save_weights_every == 0) & (episode > 0):
                state = {
                    'episode': episode,
                    'state_dict': agent.Q_eval.state_dict(),
                    'optimizer': agent.Q_eval.optimizer.state_dict(),
                    'gamma': gamma,
                    'epsilon_strategy': agent.eps_strategy,
                    'batch_size': batch_size,
                    'lr': lr,
                    'max_mem_size': max_mem_size,
                    'replace_target': replace_target,
                    'replay_memory': agent.memory,
                    'num_episodes': num_episodes_,
                    'max_step_per_episode': max_steps_per_episode_,
                    'save_weights_every': save_weights_every
                }
                now = datetime.datetime.now()
                date_string = f"{now.year}-{now.month}-{now.day}_{now.hour}-{now.minute}-{now.second}"

                T.save(state, f'saved_models/{start_date}/{date_string}')

                print(f"Latest checkpoint state saved, {date_string}")

                if avg_score > best_avg_score:
                    best_avg_score = avg_score
                    T.save(state, f'saved_models/{start_date}/best')
                    print(f"New best avg score saved")

            print()

        x = [i + 1 for i in range(n_games)]
        filename = 'wumpus_' + comment + ".png"
        plot_learning(x, scores, eps_history, filename)

        tb_train.close()

        loop_nr_hyper_parameters_variations += 1
