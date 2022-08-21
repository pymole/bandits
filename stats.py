from typing import Protocol


class Stats(Protocol):
    def update(self, reward: float):
        ...


class StandardStats(Stats):
    def __init__(self):
        self.n = 0
        self.cum_reward = 0.0
        self.history = []

    def update(self, reward: float):
        self.n += 1
        self.cum_reward += reward
        self.history.append(reward)

    def avg(self):
        # TODO: cache
        if self.n == 0:
            return 0.0
        return self.cum_reward / self.n
