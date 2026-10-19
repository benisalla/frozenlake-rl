import numpy as np

from frozenlake_rl.common import (
    Schedules,
    argmax_random,
    epsilon_greedy,
    epsilon_greedy_probs,
    linear_schedule,
)
from frozenlake_rl.config import METHODS, Config, default_config


def test_linear_schedule_reaches_end_and_stays():
    assert linear_schedule(1.0, 0.1, 0.5, 0.0) == 1.0
    assert abs(linear_schedule(1.0, 0.1, 0.5, 0.25) - 0.55) < 1e-12
    assert linear_schedule(1.0, 0.1, 0.5, 0.9) == 0.1


def test_schedules_follow_config():
    s = Schedules(Config(episodes=100, alpha=0.5, alpha_end=0.1, epsilon_start=1, epsilon_end=0, epsilon_decay=1))
    assert s.epsilon(0) == 1 and abs(s.epsilon(50) - 0.5) < 1e-12
    assert s.alpha(0) == 0.5 and abs(s.alpha(100) - 0.1) < 1e-12
    assert Schedules(Config(alpha=0.3)).alpha(10) == 0.3


def test_epsilon_greedy_probs_and_sampling():
    q = np.array([0.0, 2.0, 2.0, 1.0])
    p = epsilon_greedy_probs(q, 0.2)
    assert abs(p.sum() - 1) < 1e-12 and abs(p[1] - (0.05 + 0.4)) < 1e-12 and p[0] == 0.05
    rng = np.random.default_rng(0)
    picks = [epsilon_greedy(q, 0.0, rng) for _ in range(200)]
    assert set(picks) == {1, 2}                      # ties are broken at random
    assert argmax_random(np.array([0.0, 5.0, 1.0]), rng) == 1


def test_default_config_for_every_method_and_overrides():
    for m in METHODS:
        cfg = default_config("taxi", m)
        assert cfg.method == m and cfg.env == "taxi"
    assert default_config("fl4", "sarsa", alpha=0.7).alpha == 0.7
    assert default_config("cliff", "sarsa").epsilon_end == 0.1
