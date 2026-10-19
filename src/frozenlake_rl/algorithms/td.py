"""Temporal-difference control: learn Q(s, a) from single transitions while acting.

All of them move Q(s, a) a step alpha towards a *target* built from the next reward and the
current estimate of what comes after (bootstrapping):

    Q(s, a) <- Q(s, a) + alpha * (target - Q(s, a))

| method            | target                                          | learns about           |
|-------------------|-------------------------------------------------|------------------------|
| SARSA             | r + gamma Q(s', a'),  a' ~ epsilon-greedy       | the policy it follows  |
| expected SARSA    | r + gamma sum_a' pi(a'|s') Q(s', a')            | the policy it follows  |
| Q-learning        | r + gamma max_a' Q(s', a')                      | the greedy policy      |
| double Q-learning | r + gamma Q2(s', argmax_a' Q1(s', a'))          | the greedy policy, no max bias |
| n-step SARSA      | r_1 + ... + gamma^{n-1} r_n + gamma^n Q(s_n, a_n) | the policy it follows |
| SARSA(lambda)     | TD error spread backwards along eligibility traces | the policy it follows |
| Q(lambda)         | Watkins: traces cut after an exploratory action | the greedy policy      |

On a terminal transition the bootstrap term is 0; on a time-limit truncation it is kept (the
state still has a future, the clock just ran out).
"""

from __future__ import annotations

import numpy as np

from frozenlake_rl.common import Schedules, epsilon_greedy, epsilon_greedy_probs, greedy_policy
from frozenlake_rl.envs import get_spec, is_success, n_states_actions


def _reset(env, cfg, episode: int) -> int:
    state, _ = env.reset(seed=cfg.seed if episode == 0 else None)
    return int(state)


def _one_step_td(env, cfg, rng, tracker, target_fn) -> dict:
    """The loop shared by SARSA, expected SARSA and Q-learning; only `target_fn` differs.

    target_fn(Q, s2, a2, epsilon) returns the value of the next state used in the target.
    """
    spec = get_spec(cfg.env)
    S, A = n_states_actions(env)
    Q = np.zeros((S, A))
    sched = Schedules(cfg)
    tracker.start(greedy_policy(Q))
    for episode in range(cfg.episodes):
        eps, alpha = sched.epsilon(episode), sched.alpha(episode)
        s = _reset(env, cfg, episode)
        a = epsilon_greedy(Q[s], eps, rng)
        total, steps = 0.0, 0
        while True:
            s2, r, terminated, truncated, _ = env.step(a)
            a2 = epsilon_greedy(Q[s2], eps, rng)
            future = 0.0 if terminated else target_fn(Q, s2, a2, eps)
            Q[s, a] += alpha * (r + cfg.gamma * future - Q[s, a])
            total += r
            steps += 1
            if terminated or truncated:
                break
            s, a = s2, a2
        tracker.end_episode(episode + 1, total, steps, is_success(spec, terminated, total),
                            lambda: greedy_policy(Q))
    return {"Q": Q, "V": Q.max(1)}


def sarsa(env, cfg, rng, tracker) -> dict:
    """On-policy TD: bootstrap on the action a' the agent will actually take next."""
    return _one_step_td(env, cfg, rng, tracker, lambda Q, s2, a2, eps: Q[s2, a2])


def expected_sarsa(env, cfg, rng, tracker) -> dict:
    """SARSA with the sampled a' replaced by its expectation under epsilon-greedy: less variance."""
    return _one_step_td(env, cfg, rng, tracker,
                        lambda Q, s2, a2, eps: float(epsilon_greedy_probs(Q[s2], eps) @ Q[s2]))


def q_learning(env, cfg, rng, tracker) -> dict:
    """Off-policy TD: bootstrap on the best next action, whatever the agent does next."""
    return _one_step_td(env, cfg, rng, tracker, lambda Q, s2, a2, eps: Q[s2].max())


def double_q_learning(env, cfg, rng, tracker) -> dict:
    """Two tables: one picks the argmax, the other evaluates it (van Hasselt 2010).

    max_a Q(s', a) over noisy estimates is biased upwards; decoupling selection from
    evaluation removes that maximisation bias. Behaviour is epsilon-greedy on Q1 + Q2.
    """
    spec = get_spec(cfg.env)
    S, A = n_states_actions(env)
    Q1, Q2 = np.zeros((S, A)), np.zeros((S, A))
    sched = Schedules(cfg)
    tracker.start(greedy_policy(Q1 + Q2))
    for episode in range(cfg.episodes):
        eps, alpha = sched.epsilon(episode), sched.alpha(episode)
        s = _reset(env, cfg, episode)
        total, steps = 0.0, 0
        while True:
            a = epsilon_greedy(Q1[s] + Q2[s], eps, rng)
            s2, r, terminated, truncated, _ = env.step(a)
            if rng.random() < 0.5:
                update, other = Q1, Q2
            else:
                update, other = Q2, Q1
            future = 0.0 if terminated else other[s2, int(update[s2].argmax())]
            update[s, a] += alpha * (r + cfg.gamma * future - update[s, a])
            total += r
            steps += 1
            if terminated or truncated:
                break
            s = s2
        tracker.end_episode(episode + 1, total, steps, is_success(spec, terminated, total),
                            lambda: greedy_policy(Q1 + Q2))
    Q = (Q1 + Q2) / 2
    return {"Q": Q, "V": Q.max(1), "Q1": Q1, "Q2": Q2}


def n_step_sarsa(env, cfg, rng, tracker) -> dict:
    """n-step SARSA (Sutton & Barto, section 7.2): n real rewards, then bootstrap.

    n = 1 is SARSA; n = infinity is Monte Carlo. The update of time tau happens at time
    tau + n, once the n rewards after it are known.
    """
    spec = get_spec(cfg.env)
    S, A = n_states_actions(env)
    Q = np.zeros((S, A))
    n, gamma = cfg.n, cfg.gamma
    sched = Schedules(cfg)
    tracker.start(greedy_policy(Q))
    for episode in range(cfg.episodes):
        eps, alpha = sched.epsilon(episode), sched.alpha(episode)
        states = [_reset(env, cfg, episode)]
        actions = [epsilon_greedy(Q[states[0]], eps, rng)]
        rewards = [0.0]                                   # rewards[t] is R_t (R_0 unused)
        T, t, bootstrap_end, terminated = np.inf, 0, False, False
        while True:
            if t < T:
                s2, r, terminated, truncated, _ = env.step(actions[t])
                states.append(int(s2))
                rewards.append(float(r))
                if terminated or truncated:
                    T = t + 1
                    bootstrap_end = not terminated          # truncated: S_T still has a future
                actions.append(epsilon_greedy(Q[s2], eps, rng))
            tau = t - n + 1                                 # the time whose estimate is updated
            if tau >= 0:
                end = int(min(tau + n, T))
                G = sum(gamma ** (i - tau - 1) * rewards[i] for i in range(tau + 1, end + 1))
                if tau + n < T or bootstrap_end:
                    G += gamma ** (end - tau) * Q[states[end], actions[end]]
                sa = (states[tau], actions[tau])
                Q[sa] += alpha * (G - Q[sa])
            if tau == T - 1:
                break
            t += 1
        total = float(sum(rewards))
        tracker.end_episode(episode + 1, total, int(T), is_success(spec, terminated, total),
                            lambda: greedy_policy(Q))
    return {"Q": Q, "V": Q.max(1)}


def _lambda_control(env, cfg, rng, tracker, watkins: bool) -> dict:
    """SARSA(lambda) or Watkins's Q(lambda) with replacing eligibility traces.

    The trace E(s, a) remembers how recently (s, a) was visited; every TD error delta updates
    *all* pairs at once, Q += alpha * delta * E, then traces fade: E *= gamma * lambda. A single
    reward thus reaches every state of the path in one episode, like Monte Carlo, while still
    learning online, like TD. Q(lambda) bootstraps on max Q and cuts the traces when the agent
    takes an exploratory (non-greedy) action, because the path after it is not the greedy one.
    """
    spec = get_spec(cfg.env)
    S, A = n_states_actions(env)
    Q = np.zeros((S, A))
    E = np.zeros((S, A))
    decay = cfg.gamma * cfg.lam
    sched = Schedules(cfg)
    tracker.start(greedy_policy(Q))
    for episode in range(cfg.episodes):
        eps, alpha = sched.epsilon(episode), sched.alpha(episode)
        E[:] = 0.0
        s = _reset(env, cfg, episode)
        a = epsilon_greedy(Q[s], eps, rng)
        total, steps = 0.0, 0
        while True:
            s2, r, terminated, truncated, _ = env.step(a)
            a2 = epsilon_greedy(Q[s2], eps, rng)
            next_is_greedy = Q[s2, a2] >= Q[s2].max()
            if terminated:
                future = 0.0
            else:
                future = Q[s2].max() if watkins else Q[s2, a2]
            delta = r + cfg.gamma * future - Q[s, a]
            E[s, :] = 0.0                       # replacing trace (and clear the other actions of s)
            E[s, a] = 1.0
            Q += alpha * delta * E
            if watkins and not next_is_greedy:
                E[:] = 0.0                      # exploratory action: the greedy path is broken
            else:
                E *= decay
            total += r
            steps += 1
            if terminated or truncated:
                break
            s, a = s2, a2
        tracker.end_episode(episode + 1, total, steps, is_success(spec, terminated, total),
                            lambda: greedy_policy(Q))
    return {"Q": Q, "V": Q.max(1)}


def sarsa_lambda(env, cfg, rng, tracker) -> dict:
    return _lambda_control(env, cfg, rng, tracker, watkins=False)


def q_lambda(env, cfg, rng, tracker) -> dict:
    return _lambda_control(env, cfg, rng, tracker, watkins=True)
