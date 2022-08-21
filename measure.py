from typing import Protocol, Type

from arms import TrueMean
from bandit import MultiArmedBandit


class Measure(Protocol):
    def __init__(self, bandit: MultiArmedBandit):
        ...

    def update(self, arm_index: int, reward: float):
        ...


class CumulativeRegret(Measure):
    def __init__(self, bandit: MultiArmedBandit):
        max_mean = float("-inf")
        for arm in bandit.arms:
            assert isinstance(arm, TrueMean)
            true_mean = arm.get_true_mean()
            if true_mean > max_mean:
                max_mean = true_mean

        self.max_mean = max_mean
        self.arms: list[TrueMean] = bandit.arms
        self.total = 0.0
        self.history = []

    def update(self, arm_index: int, reward: float):
        regret = self.max_mean - self.arms[arm_index].get_true_mean()
        self.total += regret
        self.history.append(regret)


def measure(measure_cls: Type[Measure], *args, **kwargs):
    def finalize(bandit):
        return measure_cls(*args, bandit=bandit, **kwargs)

    return finalize
