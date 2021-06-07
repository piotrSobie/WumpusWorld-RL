class EpsilonGreedyStrategy:
    def __init__(self, eps_start, eps_end, eps_dec):
        self.epsilon = eps_start
        self.eps_min = eps_end
        self.eps_dec = eps_dec

    def get_epsilon(self):
        return self.epsilon

    def update_epsilon(self):
        if self.epsilon > self.eps_min:
            self.epsilon = self.epsilon - self.eps_dec
        else:
            self.epsilon = self.eps_min
