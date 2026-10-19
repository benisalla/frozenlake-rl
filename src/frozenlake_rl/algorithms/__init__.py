"""Every method, one function each, all with the same signature:

    fn(env, cfg, rng, tracker) -> {"Q": (S, A) array, "V": (S,) array, ...}

`ALGORITHMS[name]` maps the `--method` flag to the function.
"""

from frozenlake_rl.algorithms.dp import policy_iteration, value_iteration
from frozenlake_rl.algorithms.monte_carlo import mc_control
from frozenlake_rl.algorithms.neural import dqn, reinforce
from frozenlake_rl.algorithms.planning import dyna_q
from frozenlake_rl.algorithms.td import (
    double_q_learning,
    expected_sarsa,
    n_step_sarsa,
    q_lambda,
    q_learning,
    sarsa,
    sarsa_lambda,
)

ALGORITHMS = {
    "value-iteration": value_iteration,
    "policy-iteration": policy_iteration,
    "mc-first-visit": lambda env, cfg, rng, tracker: mc_control(env, cfg, rng, tracker, first_visit=True),
    "mc-every-visit": lambda env, cfg, rng, tracker: mc_control(env, cfg, rng, tracker, first_visit=False),
    "sarsa": sarsa,
    "expected-sarsa": expected_sarsa,
    "q-learning": q_learning,
    "double-q-learning": double_q_learning,
    "n-step-sarsa": n_step_sarsa,
    "sarsa-lambda": sarsa_lambda,
    "q-lambda": q_lambda,
    "dyna-q": dyna_q,
    "dqn": dqn,
    "reinforce": reinforce,
}

__all__ = ["ALGORITHMS"]
