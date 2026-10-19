"""Dynamic programming: solve the MDP from its model P, no episodes at all.

Both methods use the Bellman equations with the true transition probabilities:

    Q(s, a) = sum_{s'} p(s'|s,a) [ r + gamma * (1 - done) * V(s') ]

* value iteration applies the Bellman *optimality* backup V(s) <- max_a Q(s, a) until the
  largest change of a sweep is below theta;
* policy iteration alternates full policy evaluation (V <- V^pi, iterated to theta) and greedy
  policy improvement pi(s) <- argmax_a Q(s, a), until the policy no longer changes.

The tracker scores the greedy policy after every sweep (value iteration) or every improvement
step (policy iteration), so their curves have sweeps, not episodes, on the x axis.
"""

from __future__ import annotations

import numpy as np

from frozenlake_rl.envs import n_states_actions, transition_model


def model_arrays(env) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """P as dense arrays: T[s, a, s'] (continuation prob.), R[s, a] (expected reward)."""
    P = transition_model(env)
    S, A = n_states_actions(env)
    T = np.zeros((S, A, S))
    R = np.zeros((S, A))
    for s in range(S):
        for a in range(A):
            for p, s2, r, done in P[s][a]:
                R[s, a] += p * r
                if not done:              # no future value after a terminal transition
                    T[s, a, s2] += p
    return T, R


def q_from_v(T: np.ndarray, R: np.ndarray, V: np.ndarray, gamma: float) -> np.ndarray:
    """One-step lookahead: Q(s, a) = R(s, a) + gamma * sum_s' T(s, a, s') V(s')."""
    return R + gamma * T @ V


def value_iteration(env, cfg, rng, tracker) -> dict:
    T, R = model_arrays(env)
    V = np.zeros(T.shape[0])
    tracker.start(q_from_v(T, R, V, cfg.gamma).argmax(1))
    deltas = []
    for sweep in range(1, cfg.max_sweeps + 1):
        Q = q_from_v(T, R, V, cfg.gamma)
        V_new = Q.max(1)                          # Bellman optimality backup, all states at once
        delta = float(np.abs(V_new - V).max())
        V = V_new
        deltas.append(delta)
        tracker.end_sweep(sweep, Q.argmax(1))
        if delta < cfg.theta:
            break
    Q = q_from_v(T, R, V, cfg.gamma)
    return {"Q": Q, "V": V, "sweeps": len(deltas), "deltas": np.array(deltas)}


def policy_evaluation(T, R, policy, gamma, theta, max_sweeps, V=None) -> tuple[np.ndarray, int]:
    """Iterative evaluation of a deterministic policy: V <- R_pi + gamma T_pi V."""
    S = T.shape[0]
    idx = np.arange(S)
    T_pi, R_pi = T[idx, policy], R[idx, policy]
    V = np.zeros(S) if V is None else V.copy()
    for sweep in range(1, max_sweeps + 1):
        V_new = R_pi + gamma * T_pi @ V
        delta = float(np.abs(V_new - V).max())
        V = V_new
        if delta < theta:
            return V, sweep
    return V, max_sweeps


def policy_iteration(env, cfg, rng, tracker) -> dict:
    T, R = model_arrays(env)
    S = T.shape[0]
    policy = np.zeros(S, dtype=np.int64)
    V = np.zeros(S)
    tracker.start(policy)
    eval_sweeps = []
    for step in range(1, cfg.max_sweeps + 1):
        V, n = policy_evaluation(T, R, policy, cfg.gamma, cfg.theta, cfg.max_sweeps, V)   # evaluation
        eval_sweeps.append(n)
        Q = q_from_v(T, R, V, cfg.gamma)
        # improvement; keep the old action on (numerical) ties so the loop terminates
        best = Q.max(1)
        keep = Q[np.arange(S), policy] >= best - 1e-12
        new_policy = np.where(keep, policy, Q.argmax(1))
        tracker.end_sweep(step, new_policy)
        if np.array_equal(new_policy, policy):
            break
        policy = new_policy
    return {"Q": Q, "V": V, "policy": policy, "improvements": len(eval_sweeps),
            "sweeps": int(sum(eval_sweeps)), "eval_sweeps": np.array(eval_sweeps)}
