from __future__ import annotations

import random


class Arm:
    def pull(self) -> float:
        raise NotImplementedError


class TrueMean:
    def get_true_mean(self) -> float:
        ...


class ConstantArm(Arm, TrueMean):
    def __init__(self, c: float):
        self.c = c

    def pull(self) -> float:
        return self.c

    def get_true_mean(self):
        return self.c


class BernoulliArm(Arm, TrueMean):
    def __init__(self, p):
        self.p = p

    def pull(self) -> float:
        if random.random() < self.p:
            return 1.0
        return 0.0

    def get_true_mean(self):
        return self.p


# TODO: true mean for normal and gauss
class NormalArm(Arm):
    def __init__(self, mu: float, sigma: float):
        self.mu = mu
        self.sigma = sigma

    def pull(self) -> float:
        return random.normalvariate(self.mu, self.sigma)


class GaussArm(Arm):
    def __init__(self, mu: float, sigma: float):
        self.mu = mu
        self.sigma = sigma

    def pull(self) -> float:
        return random.gauss(self.mu, self.sigma)
