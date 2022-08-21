import random

from plotly.subplots import make_subplots
import plotly.graph_objs as go

from arms import BernoulliArm
from bandit import make_bandit, bandit_factory
from strategies import UCB


def get_some_arms():
    arms = [
        BernoulliArm(0.5), BernoulliArm(0.8), BernoulliArm(0.2),
        BernoulliArm(0.5), BernoulliArm(0.7), BernoulliArm(0.2),
        BernoulliArm(0.5), BernoulliArm(0.2), BernoulliArm(0.9),
        BernoulliArm(0.5), BernoulliArm(0.6), BernoulliArm(0.2),
        BernoulliArm(0.5), BernoulliArm(0.4), BernoulliArm(0.1),
    ]
    return arms


random.seed(10)
pulls = 1000
ucb = bandit_factory(UCB, c=2.0)
bandit = make_bandit(ucb, get_some_arms())

for _ in range(pulls):
    bandit.pull()

fig = make_subplots(
    rows=len(bandit.arms),
    cols=1,
    subplot_titles=[f"Arm {i}" for i in range(len(bandit.arms))],
)
for row, (standard,) in enumerate(bandit.arms_stats, start=1):
    fig.add_trace(
        go.Histogram(x=standard.history),
        row=row,
        col=1,
    )

fig.show()
