"""Training history shared by every method: one `Tracker` per run.

Each algorithm calls `tracker.end_episode(...)` after every episode (DP calls `end_sweep` after
every sweep). The tracker keeps the online training returns and, at `eval_points` evenly spaced
moments, scores the current greedy policy exactly with `ExactEvaluator`. From that history come
the convergence curves and the "episodes to solve" column of the comparison table.
"""

from __future__ import annotations

import time

import numpy as np

from frozenlake_rl.envs import get_spec
from frozenlake_rl.evaluate import ExactEvaluator, solved


class Tracker:
    def __init__(self, env_name: str, total: int, eval_points: int = 100):
        self.spec = get_spec(env_name)
        self.evaluator = ExactEvaluator(env_name)
        self.total = max(int(total), 1)
        self.eval_at = set(np.unique(np.linspace(0, self.total, eval_points + 1).astype(int)).tolist())
        self.history: dict[str, list] = {"x": [], "env_steps": [], "success": [], "return": []}
        self.train_returns: list[float] = []
        self.train_success: list[bool] = []
        self.env_steps = 0
        self.solved_at: int | None = None        # episodes (or sweeps for DP) until the target is met
        self.solved_steps: int | None = None
        self.solved_time: float | None = None
        self.start_time = time.perf_counter()

    def _evaluate(self, x: int, policy: np.ndarray) -> None:
        metrics = self.evaluator(policy)
        self.history["x"].append(int(x))
        self.history["env_steps"].append(int(self.env_steps))
        self.history["success"].append(metrics["success"])
        self.history["return"].append(metrics["return"])
        if self.solved_at is None and solved(self.spec, metrics):
            self.solved_at = int(x)
            self.solved_steps = int(self.env_steps)
            self.solved_time = time.perf_counter() - self.start_time

    def start(self, policy: np.ndarray) -> None:
        """Score the initial (untrained) policy at x = 0."""
        self._evaluate(0, policy)

    def end_episode(self, episode: int, ret: float, length: int, success: bool, policy_fn) -> None:
        """`episode` is the number of finished episodes (1-based); policy_fn() -> greedy action table."""
        self.train_returns.append(float(ret))
        self.train_success.append(bool(success))
        self.env_steps += int(length)
        if episode in self.eval_at:
            self._evaluate(episode, policy_fn())

    def end_sweep(self, sweep: int, policy: np.ndarray) -> None:
        self._evaluate(sweep, policy)

    def elapsed(self) -> float:
        return time.perf_counter() - self.start_time


def moving_average(values, window: int = 100) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return values
    window = max(1, min(window, len(values)))
    cumsum = np.cumsum(np.insert(values, 0, 0.0))
    out = (cumsum[window:] - cumsum[:-window]) / window
    head = cumsum[1:window] / np.arange(1, window)
    return np.concatenate([head, out])
