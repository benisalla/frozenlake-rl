"""Every method must actually learn: solve the deterministic 4x4 lake in a few hundred episodes."""

import numpy as np
import pytest

from frozenlake_rl.config import default_config
from frozenlake_rl.train import train

TABULAR = ["mc-first-visit", "mc-every-visit", "sarsa", "expected-sarsa", "q-learning", "double-q-learning",
           "n-step-sarsa", "sarsa-lambda", "q-lambda", "dyna-q"]


@pytest.mark.parametrize("method", TABULAR)
def test_tabular_method_solves_the_4x4_lake(method):
    res = train(default_config("fl4", method, episodes=600, eval_episodes=20, eval_points=20), verbose=False)
    assert res.summary["exact"]["success"] == 1.0
    assert res.summary["solved_at"] is not None


def test_mc_exploring_starts_learns():
    res = train(default_config("fl4", "mc-first-visit", episodes=600, exploring_starts=True, eval_episodes=20),
                verbose=False)
    assert res.summary["exact"]["success"] == 1.0
    assert (res.extras["visits"] > 0).sum() > 30           # many (s, a) pairs tried thanks to random starts


def test_dqn_learns_the_4x4_lake():
    cfg = default_config("fl4", "dqn", episodes=250, learning_starts=200, eval_episodes=20, eval_points=10)
    assert train(cfg, verbose=False).summary["exact"]["success"] == 1.0


def test_reinforce_learns_the_4x4_lake():
    cfg = default_config("fl4", "reinforce", episodes=400, episodes_per_update=2, eval_episodes=20, eval_points=10)
    assert train(cfg, verbose=False).summary["exact"]["success"] == 1.0


def test_sarsa_is_safer_than_q_learning_on_the_cliff():
    """Sutton & Barto example 6.6: with epsilon = 0.1 SARSA learns the safe path and earns more online."""
    q = train(default_config("cliff", "q-learning", episodes=300, eval_episodes=10), verbose=False)
    s = train(default_config("cliff", "sarsa", episodes=300, eval_episodes=10), verbose=False)
    assert np.mean(s.train_returns[-100:]) > np.mean(q.train_returns[-100:])
    assert q.summary["exact"]["return"] == -13.0              # q-learning's greedy policy is the optimal edge path


def test_dyna_planning_beats_q_learning_with_the_same_400_episodes():
    """Same real experience; the planner replays it 10x and gets much closer to V* from DP."""
    v_star = train(default_config("fl4s", "value-iteration", eval_episodes=10), verbose=False).V
    q = train(default_config("fl4s", "q-learning", episodes=400, eval_episodes=10, eval_points=10), verbose=False)
    d = train(default_config("fl4s", "dyna-q", episodes=400, eval_episodes=10, eval_points=10), verbose=False)
    assert d.summary["exact"]["success"] > 0.7 > q.summary["exact"]["success"]
    assert np.abs(d.V - v_star).max() < np.abs(q.V - v_star).max() / 3


def test_runs_are_reproducible():
    a = train(default_config("fl4s", "q-learning", episodes=300, eval_episodes=10), verbose=False)
    b = train(default_config("fl4s", "q-learning", episodes=300, eval_episodes=10), verbose=False)
    assert np.array_equal(a.Q, b.Q)
