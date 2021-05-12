import numpy as np
import random
import matplotlib.pyplot as plt


def q_learn(wumpus_env, num_episodes_=10000, max_step_per_episode_=100, learning_rate_=0.1, discount_rate_=0.9,
            start_exploration_rate=1, max_exploration_rate_=1, min_exploration_rate_=0.01,
            exploration_decay_rate_=0.001, show_number_actions_plot=True, show_reward_plot=True,
            show_games_won_plot=True, show_learned_path=True):

    env = wumpus_env

    print("Initial environment")
    env.render_env()

    action_space_size = env.action_space_n
    state_space_size = env.observation_space_n
    if state_space_size is None:
        raise Exception("Environment not suitable for q-learning")
    q_table = np.zeros((state_space_size, action_space_size))
    print(f"Action space size: {action_space_size}")
    print(f"State space size: {state_space_size}")
    print("Start Q table")
    print(q_table, "\n\n")

    # hyper parameters
    num_episodes = num_episodes_
    max_steps_per_episode = max_step_per_episode_
    learning_rate = learning_rate_
    discount_rate = discount_rate_

    exploration_rate = start_exploration_rate
    max_exploration_rate = max_exploration_rate_
    min_exploration_rate = min_exploration_rate_
    exploration_decay_rate = exploration_decay_rate_

    rewards_all_episodes = []
    actions_all_episodes = []
    game_won_all_episodes = []

    for episode in range(num_episodes):
        state = env.reset_env()

        rewards_current_episode = 0
        actions_current_episode = 0
        game_won = False

        for step in range(max_steps_per_episode):
            # exploration-exploitation trade-off
            exploration_rate_threshold = random.uniform(0, 1)
            if exploration_rate_threshold > exploration_rate:
                action = np.argmax(q_table[state, :])
            else:
                action = env.random_action()

            new_state, reward, done, info, game_won = env.step(action)

            # Update Q-table for Q(s, a)
            q_table[state, action] = q_table[state, action] * (1 - learning_rate) \
                                     + learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

            state = new_state
            rewards_current_episode += reward
            actions_current_episode += 1

            if done:
                break

        # Exploration rate decay
        exploration_rate = min_exploration_rate + \
                           (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)

        rewards_all_episodes.append(rewards_current_episode)
        actions_all_episodes.append(actions_current_episode)
        game_won_all_episodes.append(game_won)

    # number of actions per episode plot
    if show_number_actions_plot:
        plt.plot(actions_all_episodes)
        plt.ylabel("Liczba akcji")
        plt.xlabel("Epizod")
        plt.title(f"Liczba wykonanych akcji w poszczeglnych epizodach,\n klasa: {env.__class__.__name__}")
        plt.show()

    # show average reward per 1000 episodes
    if show_reward_plot:
        count = 1000
        rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), num_episodes / count)
        sum_rewards_per_thousand_episodes = []
        print(f"*****Average reward per {count} episodes*****\n")
        i = count
        for r in rewards_per_thousand_episodes:
            print(i, "\t: ", str(sum(r / count)))
            sum_rewards_per_thousand_episodes.append(sum(r / count))
            i += count

        plt.plot(sum_rewards_per_thousand_episodes)
        plt.ylabel("Nagroda")
        plt.xlabel(f"{count} epizodów")
        plt.title(f"Średnia nagroda w każdych kolejnych {count} epizodów,\n klasa: {env.__class__.__name__}")
        plt.xticks(np.arange(0, num_episodes / count, step=1))
        plt.show()

    # show number of won games per 1000 episodes
    if show_games_won_plot:
        count = 1000
        game_won_per_thousand_episodes = np.split(np.array(game_won_all_episodes), num_episodes / count)
        sum_game_won_per_thousand_episodes = []
        for g in game_won_per_thousand_episodes:
            sum_game_won_per_thousand_episodes.append(np.sum(g))

        plt.plot(sum_game_won_per_thousand_episodes)
        plt.ylabel("Ilość wygranych gier")
        plt.xlabel(f"{count} epizodów")
        plt.title(f"Ilość wygranych gier w każdych kolejnych {count} epizodów,\n klasa: {env.__class__.__name__}")
        plt.xticks(np.arange(0, num_episodes / count, step=1))
        plt.show()

    np.set_printoptions(suppress=True)
    # Print updated Q-table
    print("\n\n*****Learned Q-table*****\n")
    print(q_table)

    print(f"\nNr of actions in last episode: {actions_all_episodes[-1]}")

    # show best learned sequence of actions
    if show_learned_path:
        state = env.reset_env()
        done = False
        action_nr = 0

        print("\n\n*****Best learned sequence of actions*****\n")
        print(f"Initial grid state: ")
        env.render_env()

        while not done:
            action_nr += 1
            print(f"Action {action_nr}")

            action = np.argmax(q_table[state, :])
            new_state, reward, done, info, _ = env.step(action)

            print(f"Info: {info}")
            env.render_env()

            state = new_state

    return
