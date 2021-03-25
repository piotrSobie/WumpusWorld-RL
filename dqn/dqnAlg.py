from dqn.Agent import Agent
from dqn.EpsilonGreedyStrategy import EpsilonGreedyStrategy
from dqn.ReplayMemory import ReplayMemory
from dqn.DQN import DQN
from dqn.utils import Experience, plot, extract_tensors, QValues
import torch
import torch.optim as optim
import torch.nn.functional as F


def dqn_algorithm(wumpus_env, batch_size_=256, gamma_=0.999, eps_start_=1, eps_end_=0.01, eps_decay_=0.001,
                  target_update_=10, memory_size_=100000, lr_=0.001, num_episodes_=1000, max_steps_per_episode_=100):
    env = wumpus_env

    # hyper parameters
    batch_size = batch_size_
    gamma = gamma_
    eps_start = eps_start_
    eps_end = eps_end_
    eps_decay = eps_decay_
    target_update = target_update_  # how frequently (episodes) will update target network' weights with policy network
    memory_size = memory_size_  # capacity of replay memory
    lr = lr_
    num_episodes = num_episodes_
    max_steps_per_episode = max_steps_per_episode_

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device {device}")
    strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
    agent = Agent(strategy, env.action_space_n, device)
    memory = ReplayMemory(memory_size)

    if env.dqn_observation_state_number is None:
        raise Exception("Environment not suitable for dqn")

    policy_net = DQN(env.dqn_observation_state_number, env.action_space_n).to(device)
    target_net = DQN(env.dqn_observation_state_number, env.action_space_n).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()  # tells that this network is not in training mode
    optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)

    rewards_all_episodes = []
    for episode in range(num_episodes):
        state = env.reset_env()
        current_reward = 0

        for time_step in range(max_steps_per_episode):
            # print(f"Episode: {episode}, time_step: {time_step}")
            action = agent.select_action(state, policy_net)
            next_state, reward, done, info, _ = env.step(action.item())
            memory.push(Experience(
                torch.tensor([state], dtype=torch.float),
                action,
                torch.tensor([next_state], dtype=torch.float),
                torch.tensor([reward], dtype=torch.float)))

            state = next_state
            current_reward += reward

            if memory.can_provide_sample(batch_size):
                experiences = memory.sample(batch_size)
                states, actions, rewards, next_states = extract_tensors(experiences)

                current_q_values = QValues.get_current(policy_net, states, actions)
                next_q_values = QValues.get_next(target_net, next_states)
                target_q_values = (next_q_values * gamma) + rewards

                loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                rewards_all_episodes.append(current_reward)
                plot(rewards_all_episodes, 100)
                break

        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

    return
