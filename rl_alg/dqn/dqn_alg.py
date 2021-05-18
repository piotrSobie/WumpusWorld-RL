import numpy as np
import torch as T
from torch.utils.tensorboard import SummaryWriter
from itertools import product
import datetime
import os

from rl_alg.dqn.agent import Agent
from rl_alg.dqn.utils import plot_learning
from rl_alg.dqn.dqn_default_params import DqnDefaultParams


def dqn_algorithm(wumpus_env, batch_size_=64, gamma_=0.999, eps_start_=1.0, eps_end_=0.01, eps_decay_=5e-5,
                  target_update_=100, memory_size_=100000, lr_=0.001, num_episodes_=500, max_steps_per_episode_=200,
                  save_weights_every_=50, load_state_path=None):

    # default parameters
    default_params = DqnDefaultParams()

    batch_size_dqn = default_params.BATCH_SIZE
    gamma_dqn = default_params.DISCOUNT_RATE
    eps_start_dqn = default_params.EPSILON_START
    eps_end_dqn = default_params.EPSILON_MIN
    eps_decay_dqn = default_params.EPSILON_DECAY
    target_update_dqn = default_params.TARGET_UPDATE
    memory_size_dqn = default_params.MEMORY_SIZE
    lr_dqn = default_params.LEARNING_RATE
    num_episodes_dqn = default_params.NUM_EPISODES
    max_steps_per_episode_dqn = default_params.MAX_STEPS_PER_EPISODE
    save_weights_every_dqn = default_params.SAVE_EVERY

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
    start_episode = 0
    replay_memory_dqn = None
    net_state_dict_dqn = None
    optimizer_state_dict_dqn = None

    # load data from file if specified
    if loaded_state is not None:
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

    # overwrite settings with cmd if not none
    if batch_size_ is not None:
        batch_size_dqn = int(batch_size_)

    if gamma_ is not None:
        gamma_dqn = float(gamma_)

    if eps_start_ is not None:
        eps_start_dqn = float(eps_start_)

    if eps_end_ is not None:
        eps_end_dqn = float(eps_end_dqn)

    if eps_decay_ is not None:
        eps_decay_dqn = float(eps_decay_)

    if target_update_ is not None:
        target_update_dqn = int(target_update_)

    if memory_size_ is not None:
        memory_size_dqn = int(memory_size_)

    if lr_ is not None:
        lr_dqn = float(lr_)

    if num_episodes_ is not None:
        num_episodes_dqn = int(num_episodes_)

    if max_steps_per_episode_ is not None:
        max_steps_per_episode_dqn = int(max_steps_per_episode_)

    if save_weights_every_ is not None:
        save_weights_every_dqn = int(save_weights_every_)

    n_games = num_episodes_dqn
    max_step = max_steps_per_episode_dqn
    save_weights_every = save_weights_every_dqn

    parameters = dict(
        gamma=[gamma_dqn],
        epsilon=[eps_start_dqn],
        eps_dec=[eps_decay_dqn],
        eps_end=[eps_end_dqn],
        batch_size=[batch_size_dqn],
        lr=[lr_dqn],
        max_mem_size=[memory_size_dqn],
        replace_target=[target_update_dqn]
    )

    param_values = [v for v in parameters.values()]
    loop_nr_hyper_parameters_variations = 1
    max_nr_of_loops_hyper_parameters_variations = 1
    for e in param_values:
        max_nr_of_loops_hyper_parameters_variations *= len(e)

    for gamma, epsilon, eps_dec, eps_end, batch_size, lr, max_mem_size, replace_target in product(*param_values):
        agent = Agent(gamma=gamma, epsilon=epsilon, eps_dec=eps_dec, eps_end=eps_end, batch_size=batch_size,
                      lr=lr, max_mem_size=max_mem_size, replace_target=replace_target,
                      n_actions=env_action_n, input_dims=env_obs_n, replay_memory=replay_memory_dqn,
                      net_state_dict=net_state_dict_dqn, optimizer_state_dict=optimizer_state_dict_dqn)

        scores, eps_history = [], []
        best_avg_score = None

        # learn
        # in learning i use stable grid (only 1, not random)
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
            agent.eps_strategy.update_epsilon()
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
                    'epsilon': agent.eps_strategy.epsilon,
                    'eps_dec': agent.eps_strategy.eps_dec,
                    'eps_min': agent.eps_strategy.eps_min,
                    'batch_size': batch_size,
                    'lr': lr,
                    'max_mem_size': max_mem_size,
                    'replace_target': replace_target,
                    'replay_memory': agent.memory,
                    'num_episodes': n_games,
                    'max_step_per_episode': max_step,
                    'save_weights_every': save_weights_every,

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
