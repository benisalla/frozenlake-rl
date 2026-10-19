"""Dyna-Q: Q-learning plus planning with a model learned from experience (Sutton 1990).

After every real step the agent
1. does the normal Q-learning update with the real transition,
2. records the outcome in its model: counts of (s', r, terminated) observed after (s, a),
3. performs `planning_steps` extra Q-learning updates on transitions *simulated* from that
   model, at previously visited (s, a) pairs chosen uniformly at random.

Each real step is reused many times, so Dyna needs far fewer real episodes than Q-learning.
The model stores outcome counts, so sampling from it reproduces the slippery ice's
probabilities instead of the "last seen outcome" of the deterministic textbook version.
`planning_steps = 0` is exactly Q-learning.
"""

from __future__ import annotations

import numpy as np

from frozenlake_rl.common import Schedules, epsilon_greedy, greedy_policy
from frozenlake_rl.envs import get_spec, is_success, n_states_actions


class CountModel:
    """Empirical model: for each visited (s, a), how often each (s', r, terminated) followed."""

    def __init__(self):
        self.outcomes: dict[tuple[int, int], dict[tuple[int, float, bool], int]] = {}
        self.visited: list[tuple[int, int]] = []

    def update(self, s: int, a: int, s2: int, r: float, terminated: bool) -> None:
        key = (s, a)
        if key not in self.outcomes:
            self.outcomes[key] = {}
            self.visited.append(key)
        table = self.outcomes[key]
        out = (s2, r, terminated)
        table[out] = table.get(out, 0) + 1

    def sample(self, rng: np.random.Generator) -> tuple[int, int, int, float, bool]:
        s, a = self.visited[int(rng.integers(len(self.visited)))]
        table = self.outcomes[(s, a)]
        outcomes = list(table)
        cumulative = np.cumsum(list(table.values()))     # sample an outcome with probability count / total
        i = int(np.searchsorted(cumulative, rng.random() * cumulative[-1], side="right"))
        s2, r, terminated = outcomes[i]
        return s, a, s2, r, terminated


def dyna_q(env, cfg, rng, tracker) -> dict:
    spec = get_spec(cfg.env)
    S, A = n_states_actions(env)
    Q = np.zeros((S, A))
    model = CountModel()
    sched = Schedules(cfg)
    gamma = cfg.gamma
    tracker.start(greedy_policy(Q))
    for episode in range(cfg.episodes):
        eps, alpha = sched.epsilon(episode), sched.alpha(episode)
        s, _ = env.reset(seed=cfg.seed if episode == 0 else None)
        total, steps = 0.0, 0
        while True:
            a = epsilon_greedy(Q[s], eps, rng)
            s2, r, terminated, truncated, _ = env.step(a)
            # (1) direct RL: Q-learning on the real transition
            Q[s, a] += alpha * (r + (0.0 if terminated else gamma * Q[s2].max()) - Q[s, a])
            # (2) model learning
            model.update(int(s), int(a), int(s2), float(r), bool(terminated))
            # (3) planning: replay simulated transitions from the model
            for _ in range(cfg.planning_steps):
                ps, pa, ps2, pr, pterm = model.sample(rng)
                Q[ps, pa] += alpha * (pr + (0.0 if pterm else gamma * Q[ps2].max()) - Q[ps, pa])
            total += r
            steps += 1
            if terminated or truncated:
                break
            s = s2
        tracker.end_episode(episode + 1, total, steps, is_success(spec, terminated, total),
                            lambda: greedy_policy(Q))
    return {"Q": Q, "V": Q.max(1)}
