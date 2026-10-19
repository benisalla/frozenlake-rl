"""Evaluation: the same two measurements for every method.

1. `ExactEvaluator`: the success probability and expected (undiscounted) return of a policy,
   computed *exactly* from the model P with the environment's time limit. It is used for the
   learning curves because it has no sampling noise and costs a few matrix-vector products.
   The agent never sees P (except DP); only the experimenter uses it to score the agent.

2. `rollout_evaluate`: N real gymnasium episodes with the greedy policy, the number reported in
   the results tables (what you would measure without knowing the model).

With enough episodes the two agree to within the Monte Carlo error sqrt(p(1-p)/N).

CLI: `frozenlake-eval runs/frozenlake-4x4-slippery/q-learning --episodes 1000`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from frozenlake_rl.envs import get_spec, is_success, make_env, n_states_actions, transition_model


def policy_matrix(policy: np.ndarray, n_actions: int) -> np.ndarray:
    """(S,) action table -> (S, A) one-hot probabilities; (S, A) probabilities pass through."""
    policy = np.asarray(policy)
    if policy.ndim == 2:
        return policy.astype(np.float64)
    pi = np.zeros((len(policy), n_actions))
    pi[np.arange(len(policy)), policy.astype(int)] = 1.0
    return pi


class ExactEvaluator:
    """Finite-horizon policy evaluation on the true model.

    For a horizon of H steps (the time limit) and policy pi:

        v_0 = 0,    v_k(s) = sum_a pi(a|s) sum_{s'} p(s'|s,a) [ r + (1 - done) v_{k-1}(s') ]

    v_H at the start distribution is the expected return of one episode; replacing r by the
    indicator "this transition is a success" gives the success probability.
    """

    def __init__(self, env_name: str):
        self.spec = get_spec(env_name)
        env = make_env(env_name)
        P = transition_model(env)
        S, A = n_states_actions(env)
        self.n_states, self.n_actions = S, A
        self.R = np.zeros((S, A))          # expected immediate reward
        self.SU = np.zeros((S, A))         # probability that this step is a success
        self.M = np.zeros((S, A, S))       # probability of continuing to s'
        for s in range(S):
            for a in range(A):
                for p, s2, r, done in P[s][a]:
                    self.R[s, a] += p * r
                    self.SU[s, a] += p * is_success(self.spec, done, r)
                    if not done:
                        self.M[s, a, s2] += p
        self.start = np.asarray(env.unwrapped.initial_state_distrib, dtype=np.float64)
        env.close()

    def __call__(self, policy: np.ndarray) -> dict:
        pi = policy_matrix(policy, self.n_actions)
        r_pi = (pi * self.R).sum(1)
        s_pi = (pi * self.SU).sum(1)
        m_pi = np.einsum("sa,sat->st", pi, self.M)
        v_ret = np.zeros(self.n_states)
        v_suc = np.zeros(self.n_states)
        for _ in range(self.spec.max_steps):
            v_ret = r_pi + m_pi @ v_ret
            v_suc = s_pi + m_pi @ v_suc
        return {"success": float(self.start @ v_suc), "return": float(self.start @ v_ret)}


    def best(self) -> dict:
        """The best score any policy can reach within the time limit (finite-horizon DP).

        Maximises the env's target metric directly: v_k(s) = max_a [ score(s, a) + sum_s' M v_{k-1}(s') ]
        with score = success indicator (FrozenLake) or reward (Taxi, CliffWalking). This is the
        ceiling of the evaluation metric; the gamma-discounted optimum of value iteration can sit
        slightly below it (gamma = 0.99 trades a little success for speed on the 8x8 lake).
        """
        score = self.SU if self.spec.target_metric == "success" else self.R
        v = np.zeros(self.n_states)
        for _ in range(self.spec.max_steps):
            v = (score + self.M @ v).max(1)
        return {self.spec.target_metric: float(self.start @ v)}


def rollout_evaluate(env_name: str, policy: np.ndarray, episodes: int = 1000, seed: int = 10_000) -> dict:
    """Run `episodes` real episodes with a deterministic action table; report the usual metrics."""
    spec = get_spec(env_name)
    env = make_env(env_name)
    policy = np.asarray(policy)
    if policy.ndim == 2:
        policy = policy.argmax(1)
    returns, lengths, successes = [], [], []
    for i in range(episodes):
        state, _ = env.reset(seed=seed + i)
        total, steps, terminated, truncated = 0.0, 0, False, False
        while not (terminated or truncated):
            state, reward, terminated, truncated, _ = env.step(int(policy[state]))
            total += reward
            steps += 1
        returns.append(total)
        lengths.append(steps)
        successes.append(is_success(spec, terminated, total))
    env.close()
    returns = np.array(returns)
    return {
        "episodes": episodes,
        "success_rate": float(np.mean(successes)),
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "mean_length": float(np.mean(lengths)),
    }


def solved(spec, metrics: dict) -> bool:
    """Has a policy reached the env's target (success probability or mean return)?"""
    value = metrics["success"] if spec.target_metric == "success" else metrics["return"]
    return value >= spec.target - 1e-9


def load_run(run_dir: str | Path) -> tuple[dict, dict]:
    """(arrays, summary) saved by `frozenlake-train`: arrays has Q / V / policy."""
    run_dir = Path(run_dir)
    arrays = dict(np.load(run_dir / "agent.npz"))
    summary = json.loads((run_dir / "summary.json").read_text())
    return arrays, summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained policy (greedy) on real episodes.")
    parser.add_argument("run_dir", help="directory written by frozenlake-train")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=10_000)
    args = parser.parse_args(argv)
    arrays, summary = load_run(args.run_dir)
    env_name = summary["config"]["env"]
    rollout = rollout_evaluate(env_name, arrays["policy"], args.episodes, args.seed)
    exact = ExactEvaluator(env_name)(arrays["policy"])
    print(f"{summary['config']['method']} on {env_name}")
    print(f"  rollouts ({args.episodes} episodes): success {rollout['success_rate']:.3f}  "
          f"return {rollout['mean_return']:.2f} +- {rollout['std_return']:.2f}  length {rollout['mean_length']:.1f}")
    print(f"  exact (model, time limit):   success {exact['success']:.3f}  return {exact['return']:.2f}")


if __name__ == "__main__":
    main()
