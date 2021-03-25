import random
import torch


class Agent:
    def __init__(self, strategy, num_actions, device):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device

    # action
    def select_action(self, state, policy_net):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate > random.random():
            action = random.randrange(self.num_actions)
            return torch.tensor([action]).to(self.device)
        else:
            state = torch.tensor([state], dtype=torch.float)
            with torch.no_grad():
                return policy_net(state).argmax(dim=1).to(self.device)
