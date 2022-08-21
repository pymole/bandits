import math
import random

from bandit import MultiArmedBandit


class Random(MultiArmedBandit):
    def strategy(self) -> int:
        return random.randint(0, len(self.arms) - 1)


def e_greedy(e: float, bandit: MultiArmedBandit):
    if random.random() >= e:
        return random.randint(0, len(bandit.arms) - 1)

    best_arm_index = 0
    highest_avg = float("-inf")

    for i, arm in enumerate(bandit.arms):
        standard, = bandit.arms_stats[i]
        avg = standard.avg()
        if avg > highest_avg:
            best_arm_index = i
            highest_avg = avg

    return best_arm_index


class EGreedy(MultiArmedBandit):
    def __init__(self, e: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert 0.0 <= e <= 1.0
        self.e = e

    def strategy(self) -> int:
        return e_greedy(self.e, self)


class EGreedyDecay(MultiArmedBandit):
    def strategy(self) -> int:
        return e_greedy(1 / (self.n + 1), self)


class UCB(MultiArmedBandit):
    def __init__(self, c=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c = c

    def strategy(self) -> int:
        best_arm_index = 0
        highest_ucb = float("-inf")
        log_n = math.log(self.n + 1)

        for i, arm in enumerate(self.arms):
            standard, = self.arms_stats[i]
            if standard.n == 0:
                return i

            ucb = standard.avg() + self.c * math.sqrt(log_n / standard.n)
            if ucb > highest_ucb:
                best_arm_index = i
                highest_ucb = ucb

        return best_arm_index


class ThompsonSampling(MultiArmedBandit):
    def strategy(self) -> int:
        best_arm_index = 0
        highest = float("-inf")
        for i, arm in enumerate(self.arms):
            standard, = self.arms_stats[i]

            value = random.betavariate(standard.cum_reward + 1.0, standard.n - standard.cum_reward + 1.0)
            if value > highest:
                best_arm_index = i
                highest = value

        return best_arm_index

