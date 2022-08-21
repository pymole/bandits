import random

import plotly.graph_objs as go

from arms import BernoulliArm
from bandit import bandit_factory, make_bandit
from measure import measure, CumulativeRegret
from strategies import UCB, EGreedy, EGreedyDecay, Random, ThompsonSampling


def make_root_regret_bandit(bandit_factory, arms):
    arms = [make_bandit(bandit_factory, arm) for arm in arms]
    bandit = bandit_factory(
        arms=arms,
        measures=[measure(CumulativeRegret)]
    )

    return bandit


def get_regret_y(history):
    y = [history[0]]
    for i in range(1, len(history)):
        y.append(history[i] + y[i - 1])

    return y


def get_some_arms():
    arms = [
        BernoulliArm(0.5), BernoulliArm(0.8), BernoulliArm(0.2),
        BernoulliArm(0.5), BernoulliArm(0.7), BernoulliArm(0.2),
        BernoulliArm(0.5), BernoulliArm(0.2), BernoulliArm(0.9),
        BernoulliArm(0.5), BernoulliArm(0.6), BernoulliArm(0.2),
        BernoulliArm(0.5), BernoulliArm(0.4), BernoulliArm(0.1),
    ]
    return arms


# Bandit factories
ucb = bandit_factory(UCB, c=2.0)
e_greedy = bandit_factory(EGreedy, e=0.1)
e_greedy_decay = bandit_factory(EGreedyDecay)
random_strat = bandit_factory(Random)
ts = bandit_factory(ThompsonSampling)

strats = [
    ("UCB", ucb),
    ("E-Greedy 0.1", e_greedy),
    ("E-Greedy Decay", e_greedy_decay),
    ("Random", random_strat),
    ("Thompson Sampling", ts),
]

pulls = 10000

fig = go.Figure()

for name, strat in strats:
    random.seed(10)
    arms = get_some_arms()
    bandit = make_root_regret_bandit(strat, arms)
    for _ in range(pulls):
        bandit.pull()
    regret_history = bandit.measures[0].history
    fig.add_trace(go.Scatter(
        y=get_regret_y(regret_history),
        mode='lines',
        name=name,
    ))

fig.update_xaxes(
    tickangle=90,
    title_text="t",
    title_font={"size": 20},
    title_standoff=25
)

fig.update_yaxes(
    title_text="Cumulative regret",
    title_standoff=25
)

fig.show()
