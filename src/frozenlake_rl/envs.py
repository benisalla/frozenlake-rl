"""The environments: FrozenLake (4x4, 8x8, slippery or not), Taxi and CliffWalking.

All of them are gymnasium "toy-text" environments with a finite number of states and actions,
and all of them expose their full model in `env.unwrapped.P`:

    P[s][a] = [(probability, next_state, reward, terminated), ...]

Model-based methods (value / policy iteration) read P directly; every other method only
calls `reset()` and `step()`, exactly like an agent that does not know the rules.

An environment is chosen with one string, the `--env` flag of every CLI:

| name                      | gymnasium id         | states | what "solved" means                    |
|---------------------------|----------------------|--------|----------------------------------------|
| `frozenlake-4x4`          | FrozenLake-v1        | 16     | reach the goal on every episode        |
| `frozenlake-4x4-slippery` | FrozenLake-v1        | 16     | success rate close to the optimal 74 % |
| `frozenlake-8x8`          | FrozenLake-v1 (8x8)  | 64     | reach the goal on every episode        |
| `frozenlake-8x8-slippery` | FrozenLake-v1 (8x8)  | 64     | success rate close to the optimal 91 % |
| `taxi`                    | Taxi-v3 / Taxi-v4    | 500    | mean return close to the optimal ~8    |
| `cliffwalking`            | CliffWalking-v0 / v1 | 48     | mean return -13 (the shortest path)    |
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

warnings.filterwarnings("ignore", message=".*is out of date.*")


@dataclass(frozen=True)
class EnvSpec:
    """Everything the rest of the code needs to know about one environment."""

    name: str
    ids: tuple[str, ...]          # gymnasium ids to try, newest first (versions differ across gymnasium)
    kwargs: dict
    max_steps: int                # episode time limit
    target_metric: str            # "success" or "return": what "solved" is measured with
    target: float                 # threshold on that metric for "episodes to solve"
    grid: tuple[int, int] | None  # (rows, cols) when states are cells of a grid (heatmaps / arrows)
    gamma: float = 0.99


ENVS: dict[str, EnvSpec] = {
    "frozenlake-4x4": EnvSpec("frozenlake-4x4", ("FrozenLake-v1",), {"map_name": "4x4", "is_slippery": False},
                              100, "success", 1.0, (4, 4)),
    "frozenlake-4x4-slippery": EnvSpec("frozenlake-4x4-slippery", ("FrozenLake-v1",),
                                       {"map_name": "4x4", "is_slippery": True}, 100, "success", 0.70, (4, 4)),
    "frozenlake-8x8": EnvSpec("frozenlake-8x8", ("FrozenLake-v1",), {"map_name": "8x8", "is_slippery": False},
                              200, "success", 1.0, (8, 8)),
    "frozenlake-8x8-slippery": EnvSpec("frozenlake-8x8-slippery", ("FrozenLake-v1",),
                                       {"map_name": "8x8", "is_slippery": True}, 200, "success", 0.80, (8, 8)),
    "taxi": EnvSpec("taxi", ("Taxi-v4", "Taxi-v3"), {}, 200, "return", 7.0, None),
    "cliffwalking": EnvSpec("cliffwalking", ("CliffWalking-v1", "CliffWalking-v0"), {}, 200, "return", -13.0,
                            (4, 12)),
}

ALIASES = {"fl4": "frozenlake-4x4", "fl4s": "frozenlake-4x4-slippery", "fl8": "frozenlake-8x8",
           "fl8s": "frozenlake-8x8-slippery", "frozenlake": "frozenlake-4x4-slippery", "cliff": "cliffwalking"}

ACTION_ARROWS = {
    # FrozenLake / CliffWalking action order differs: FrozenLake 0=left 1=down 2=right 3=up,
    # CliffWalking 0=up 1=right 2=down 3=left.
    "frozenlake": ["←", "↓", "→", "↑"],
    "cliffwalking": ["↑", "→", "↓", "←"],
    "taxi": ["S", "N", "E", "W", "P", "D"],
}


def get_spec(name: str) -> EnvSpec:
    name = ALIASES.get(name.lower(), name.lower())
    if name not in ENVS:
        raise ValueError(f"unknown env {name!r}; choose from {sorted(ENVS)} or aliases {sorted(ALIASES)}")
    return ENVS[name]


def make_env(name: str, render_mode: str | None = None) -> gym.Env:
    """Create the gymnasium env for `name` with its time limit (tries each known version id)."""
    spec = get_spec(name)
    last_error: Exception | None = None
    for env_id in spec.ids:
        try:
            return gym.make(env_id, max_episode_steps=spec.max_steps, render_mode=render_mode, **spec.kwargs)
        except Exception as error:  # deprecated / unknown version in this gymnasium release
            last_error = error
    raise RuntimeError(f"could not create {name}: {last_error}")


def env_family(name: str) -> str:
    return get_spec(name).name.split("-")[0]


def n_states_actions(env: gym.Env) -> tuple[int, int]:
    return int(env.observation_space.n), int(env.action_space.n)


def transition_model(env: gym.Env) -> dict:
    """The full MDP model P[s][a] = [(p, s', r, terminated), ...] of a toy-text env."""
    return env.unwrapped.P


def terminal_states(env: gym.Env) -> np.ndarray:
    """Boolean mask of absorbing states: every action loops back to the state itself.

    On FrozenLake these are the holes and the goal; the episode ends when the agent enters them,
    so their value is 0 by definition. Taxi has none (the episode ends on the drop-off transition).
    """
    P = transition_model(env)
    n_states, _ = n_states_actions(env)
    return np.array([
        all(s2 == s for trans in P[s].values() for _p, s2, _r, _d in trans) for s in range(n_states)
    ])


def is_success(spec: EnvSpec, terminated: bool, total_reward: float) -> bool:
    """FrozenLake: reached the goal (reward 1). Taxi: passenger delivered. Cliff: reached the goal."""
    if spec.name.startswith("frozenlake"):
        return total_reward > 0
    return bool(terminated)


def one_hot(state: int, n_states: int) -> np.ndarray:
    x = np.zeros(n_states, dtype=np.float32)
    x[state] = 1.0
    return x
