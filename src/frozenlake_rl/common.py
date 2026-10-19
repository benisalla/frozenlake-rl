"""Pieces every method shares: schedules, epsilon-greedy action choice, greedy policies.

Kept tiny on purpose: the methods differ only in *what target* they move Q(s, a) towards,
and everything around that (exploration, step size, evaluation) is the same code for all.
"""

from __future__ import annotations

import os

import numpy as np


def linear_schedule(start: float, end: float, fraction: float, progress: float) -> float:
    """Value at `progress` in [0, 1]: goes from start to end over the first `fraction`, then flat."""
    if fraction <= 0:
        return end
    t = progress / fraction
    return end if t >= 1.0 else start + t * (end - start)


class Schedules:
    """epsilon(episode) and alpha(episode) for a run of `episodes` episodes."""

    def __init__(self, cfg, episodes: int | None = None):
        self.cfg = cfg
        self.episodes = max(episodes or cfg.episodes, 1)

    def epsilon(self, episode: int) -> float:
        c = self.cfg
        return linear_schedule(c.epsilon_start, c.epsilon_end, c.epsilon_decay, episode / self.episodes)

    def alpha(self, episode: int) -> float:
        c = self.cfg
        if c.alpha_end is None:
            return c.alpha
        return linear_schedule(c.alpha, c.alpha_end, 1.0, episode / self.episodes)


def argmax_random(values: np.ndarray, rng: np.random.Generator) -> int:
    """argmax with random tie-breaking (all-zero Q tables must not always pick action 0)."""
    best = np.flatnonzero(values == values.max())
    return int(best[0]) if len(best) == 1 else int(rng.choice(best))


def epsilon_greedy(q_row: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    """With probability epsilon a uniformly random action, otherwise a greedy one."""
    if rng.random() < epsilon:
        return int(rng.integers(len(q_row)))
    return argmax_random(q_row, rng)


def epsilon_greedy_probs(q_row: np.ndarray, epsilon: float) -> np.ndarray:
    """pi(a|s) of the epsilon-greedy policy (ties share the greedy mass), used by expected SARSA."""
    n = len(q_row)
    probs = np.full(n, epsilon / n)
    best = np.flatnonzero(q_row == q_row.max())
    probs[best] += (1.0 - epsilon) / len(best)
    return probs


def greedy_policy(Q: np.ndarray) -> np.ndarray:
    """Deterministic greedy policy: one action per state (first max on ties)."""
    return np.asarray(Q).argmax(axis=1).astype(np.int64)


def set_threads(threads: int = 1) -> None:
    """One thread by default: these problems are tiny and the box is shared."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, str(threads))
    try:
        import torch

        torch.set_num_threads(threads)
    except ImportError:  # pragma: no cover
        pass
