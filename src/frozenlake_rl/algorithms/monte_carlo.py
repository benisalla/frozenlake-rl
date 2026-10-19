"""Monte Carlo control: learn Q(s, a) from complete episodes, no bootstrapping.

Play a whole episode with the epsilon-greedy policy, then walk it backwards computing the
return G_t = r_{t+1} + gamma * G_{t+1}, and average G_t into Q(s_t, a_t):

    Q(s, a) += alpha * (G - Q(s, a))          (constant-alpha MC, the default)
    N(s, a) += 1,  Q(s, a) += (G - Q(s, a)) / N(s, a)   (sample average, `sample_average=True`)

The sample average is the textbook estimator of q_pi for a *fixed* pi; in control the policy
keeps changing, and averaging in returns of the early random policy forever makes Q lag behind.
A constant (or slowly decaying) step size forgets old returns, which works much better here.

* first-visit: only the first occurrence of (s, a) in an episode is averaged
  (unbiased estimates of q_pi);
* every-visit: every occurrence is averaged (biased but consistent; more updates per episode).

`exploring_starts=True` starts each episode in a uniformly random non-terminal state with a
random first action, the textbook way to guarantee every (s, a) is tried (Sutton & Barto 5.3).

The returns are the true discounted sums of real rewards, so the estimates are unbiased but
their variance grows with the episode length: on long or sparse problems Monte Carlo needs
many more episodes than TD.
"""

from __future__ import annotations

import numpy as np

from frozenlake_rl.common import Schedules, epsilon_greedy, greedy_policy
from frozenlake_rl.envs import get_spec, is_success, n_states_actions, terminal_states


def mc_control(env, cfg, rng, tracker, first_visit: bool = True) -> dict:
    spec = get_spec(cfg.env)
    S, A = n_states_actions(env)
    Q = np.zeros((S, A))
    N = np.zeros((S, A))
    sched = Schedules(cfg)
    starts = np.flatnonzero(~terminal_states(env))
    tracker.start(greedy_policy(Q))
    for episode in range(cfg.episodes):
        eps, alpha = sched.epsilon(episode), sched.alpha(episode)
        s, _ = env.reset(seed=cfg.seed if episode == 0 else None)
        first_action = None
        if cfg.exploring_starts:
            s = int(rng.choice(starts))
            env.unwrapped.s = s                     # teleport: legal for toy-text envs
            first_action = int(rng.integers(A))

        # 1. generate an episode with the current epsilon-greedy policy
        trajectory = []                             # (s, a, r)
        terminated = truncated = False
        while not (terminated or truncated):
            a = first_action if (first_action is not None and not trajectory) else epsilon_greedy(Q[s], eps, rng)
            s2, r, terminated, truncated, _ = env.step(a)
            trajectory.append((int(s), int(a), float(r)))
            s = s2

        # 2. walk it backwards and average the returns into Q
        if first_visit:
            first_index = {}
            for t, (st, at, _r) in enumerate(trajectory):
                first_index.setdefault((st, at), t)
        G = 0.0
        for t in range(len(trajectory) - 1, -1, -1):
            st, at, r = trajectory[t]
            G = r + cfg.gamma * G
            if first_visit and first_index[(st, at)] != t:
                continue
            N[st, at] += 1
            step = 1.0 / N[st, at] if cfg.sample_average else alpha
            Q[st, at] += step * (G - Q[st, at])

        total = sum(r for _s, _a, r in trajectory)
        tracker.end_episode(episode + 1, total, len(trajectory), is_success(spec, terminated, total),
                            lambda: greedy_policy(Q))
    return {"Q": Q, "V": Q.max(1), "visits": N}
