"""One frozen dataclass holds every hyperparameter, with good defaults per environment.

`default_config(env, method)` returns settings that solve (or come close to the optimum on)
each environment in seconds to a few minutes on one CPU thread. Every field can be overridden
from the CLI (`frozenlake-train --alpha 0.2 --episodes 5000 ...`) or from Python
(`default_config("taxi", "sarsa", alpha=0.2)`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace

DP_METHODS = ("value-iteration", "policy-iteration")
MC_METHODS = ("mc-first-visit", "mc-every-visit")
TD_METHODS = ("sarsa", "expected-sarsa", "q-learning", "double-q-learning", "n-step-sarsa",
              "sarsa-lambda", "q-lambda")
PLANNING_METHODS = ("dyna-q",)
NEURAL_METHODS = ("dqn", "reinforce")
METHODS = DP_METHODS + MC_METHODS + TD_METHODS + PLANNING_METHODS + NEURAL_METHODS


@dataclass(frozen=True)
class Config:
    env: str = "frozenlake-4x4-slippery"
    method: str = "q-learning"
    seed: int = 0
    gamma: float = 0.99

    # experience budget (ignored by dynamic programming)
    episodes: int = 10_000

    # step size: constant, or decayed linearly from `alpha` to `alpha_end` over all episodes
    alpha: float = 0.1
    alpha_end: float | None = None

    # exploration: epsilon decays linearly from epsilon_start to epsilon_end over
    # `epsilon_decay` (fraction of the episodes), then stays at epsilon_end
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.5

    # method-specific
    n: int = 4                       # n-step SARSA
    lam: float = 0.9                 # SARSA(lambda), Q(lambda)
    planning_steps: int = 10         # Dyna-Q: simulated updates per real step
    exploring_starts: bool = False   # Monte Carlo: random first state and action
    sample_average: bool = False     # Monte Carlo: 1/N(s,a) step instead of alpha
    theta: float = 1e-8              # dynamic programming stopping threshold
    max_sweeps: int = 10_000         # dynamic programming safety cap

    # neural methods (DQN, REINFORCE) on one-hot states
    hidden: int = 64
    lr: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 20_000
    target_update: int = 100         # DQN: copy online -> target every this many updates
    train_every: int = 4             # DQN: one gradient step every this many env steps
    learning_starts: int = 500       # DQN: random steps before the first update
    entropy_coef: float = 0.01       # REINFORCE: entropy bonus
    episodes_per_update: int = 10    # REINFORCE: episodes averaged in one gradient step

    # evaluation
    eval_points: int = 50            # how many times the greedy policy is scored during training
    eval_episodes: int = 1000        # final Monte Carlo evaluation (real gymnasium episodes)

    threads: int = 1

    def to_dict(self) -> dict:
        return asdict(self)


# Per-environment defaults for the model-free methods. Tuned for "solves it in seconds".
_ENV_DEFAULTS: dict[str, dict] = {
    "frozenlake-4x4": dict(episodes=1_000, alpha=0.5, alpha_end=0.01),
    "frozenlake-4x4-slippery": dict(episodes=10_000, alpha=0.1, alpha_end=0.01),
    "frozenlake-8x8": dict(episodes=3_000, alpha=0.5, alpha_end=0.01),
    "frozenlake-8x8-slippery": dict(episodes=20_000, alpha=0.1, alpha_end=0.01, epsilon_decay=0.7),
    "taxi": dict(episodes=3_000, alpha=0.5, alpha_end=0.05, epsilon_end=0.01),
    # the classic Sutton & Barto setting: constant epsilon 0.1, alpha 0.5
    "cliffwalking": dict(episodes=500, alpha=0.5, epsilon_start=0.1, epsilon_end=0.1, gamma=1.0),
}

# Per-(env, method) adjustments on top of the env defaults: Monte Carlo needs smaller steps and
# more episodes (its targets are noisy full returns), Dyna-Q fewer episodes (each one is replayed
# by the planner), and the neural methods their own learning rates and budgets.
_OVERRIDES: dict[tuple[str, str], dict] = {
    ("frozenlake-4x4-slippery", "mc-first-visit"): dict(episodes=20_000, alpha=0.05, alpha_end=0.005),
    ("frozenlake-4x4-slippery", "mc-every-visit"): dict(episodes=20_000, alpha=0.05, alpha_end=0.005),
    ("frozenlake-8x8-slippery", "mc-first-visit"): dict(episodes=40_000, alpha=0.05, alpha_end=0.005),
    ("frozenlake-8x8-slippery", "mc-every-visit"): dict(episodes=40_000, alpha=0.05, alpha_end=0.005),
    ("frozenlake-4x4-slippery", "dyna-q"): dict(episodes=2_000),
    ("frozenlake-8x8-slippery", "dyna-q"): dict(episodes=2_500),
    ("taxi", "dyna-q"): dict(episodes=1_000),
    # larger n needs a smaller step size (Sutton & Barto figure 7.2)
    ("taxi", "n-step-sarsa"): dict(episodes=5_000, alpha=0.1, alpha_end=None),
    ("frozenlake-4x4", "dqn"): dict(episodes=1_000),
    ("frozenlake-4x4-slippery", "dqn"): dict(episodes=1_500, lr=5e-4),
    ("frozenlake-8x8", "dqn"): dict(episodes=2_000),
    ("frozenlake-8x8-slippery", "dqn"): dict(episodes=2_000, lr=5e-4),
    ("taxi", "dqn"): dict(episodes=1_500, epsilon_end=0.01, hidden=128),
    ("cliffwalking", "dqn"): dict(episodes=500),
    ("frozenlake-4x4", "reinforce"): dict(episodes=1_000, lr=1e-2),
    ("frozenlake-4x4-slippery", "reinforce"): dict(episodes=8_000, lr=1e-2),
    ("frozenlake-8x8", "reinforce"): dict(episodes=4_000, lr=1e-2),
    ("frozenlake-8x8-slippery", "reinforce"): dict(episodes=15_000, lr=1e-2),
    ("taxi", "reinforce"): dict(episodes=5_000, lr=1e-2, hidden=128),
    ("cliffwalking", "reinforce"): dict(episodes=2_000, lr=1e-2),
}


def default_config(env: str = "frozenlake-4x4-slippery", method: str = "q-learning", **overrides) -> Config:
    """The tuned defaults for (env, method), with any field overridden by keyword."""
    from frozenlake_rl.envs import get_spec

    env = get_spec(env).name
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    values: dict = dict(env=env, method=method)
    values.update(_ENV_DEFAULTS.get(env, {}))
    values.update(_OVERRIDES.get((env, method), {}))
    known = {f.name for f in fields(Config)}
    unknown = set(overrides) - known
    if unknown:
        raise ValueError(f"unknown config fields: {sorted(unknown)}")
    values.update({k: v for k, v in overrides.items() if v is not None})
    return Config(**values)


def with_overrides(cfg: Config, **overrides) -> Config:
    return replace(cfg, **{k: v for k, v in overrides.items() if v is not None})
