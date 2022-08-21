from __future__ import annotations

from arms import Arm
from stats import Stats, StandardStats


class MultiArmedBandit(Arm, StandardStats):
    def __init__(self, arms: list[Arm], measures: list = None):
        super().__init__()
        self.arms = arms
        self.arms_stats: list[list[Stats]] = [
            [
                arm if isinstance(arm, MultiArmedBandit) else StandardStats(),
                *self.get_additional_arm_stats(arm),
            ]
            for arm in arms
        ]

        if measures:
            measures = [measure(self) for measure in measures]
        else:
            measures = []

        self.measures = measures

    def pull(self) -> float:
        arm_index = self.strategy()
        arm = self.arms[arm_index]
        reward = arm.pull()
        self.update(reward)

        standard, *other = self.arms_stats[arm_index]
        if not isinstance(arm, MultiArmedBandit):
            standard.update(reward)

        for additional in other:
            additional.update(reward)

        for measure in self.measures:
            measure.update(arm_index, reward)

        return reward

    def strategy(self) -> int:
        raise NotImplementedError

    def get_additional_arm_stats(self, arm) -> list[Stats]:
        return []


def make_bandit(bandit_factory, arms: Arm | list):
    if isinstance(arms, Arm):
        return arms

    arms = [
        make_bandit(bandit_factory, arm)
        for arm in arms
    ]

    return bandit_factory(arms=arms)


def bandit_factory(bandit_cls, **kwargs):
    def finalize(**finalize_kwargs):
        return bandit_cls(**kwargs, **finalize_kwargs)

    return finalize
