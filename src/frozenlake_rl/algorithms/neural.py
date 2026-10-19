"""Neural methods on one-hot states: DQN and REINFORCE with a learned baseline.

A tabular problem is the easiest possible input for a network: state s becomes the one-hot
vector e_s, so a linear layer on it *is* a table, and the hidden layer only adds capacity. The
point is to see the deep-RL machinery (replay buffer, target network, policy gradient) work on
a problem where the exact answer is known from dynamic programming.

DQN (Mnih et al. 2015): Q_theta(s, .) is an MLP; transitions go into a replay buffer; every
`train_every` steps a random batch is regressed on the Q-learning target computed with a
frozen copy of the network (the target network, synced every `target_update` updates):

    loss = Huber( Q_theta(s, a),  r + gamma (1 - done) max_a' Q_target(s', a') )

REINFORCE (Williams 1992) with a value baseline: after every `episodes_per_update` episodes,
averaged over all their steps,

    loss = - log pi_theta(a_t|s_t) (G_t - V_phi(s_t))  +  0.5 (G_t - V_phi(s_t))^2  - beta H(pi(.|s_t))

The baseline does not change the expected gradient but removes most of its variance.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from frozenlake_rl.common import Schedules, epsilon_greedy
from frozenlake_rl.envs import get_spec, is_success, n_states_actions


def mlp(n_in: int, hidden: int, n_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, n_out))


class ReplayBuffer:
    """Fixed-size circular buffer of (s, a, r, s', terminated) as integer / float arrays."""

    def __init__(self, capacity: int):
        self.s = np.zeros(capacity, dtype=np.int64)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.s2 = np.zeros(capacity, dtype=np.int64)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.capacity, self.size, self.pos = capacity, 0, 0

    def add(self, s, a, r, s2, done) -> None:
        i = self.pos
        self.s[i], self.a[i], self.r[i], self.s2[i], self.done[i] = s, a, r, s2, done
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch: int, rng: np.random.Generator):
        idx = rng.integers(self.size, size=batch)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.done[idx]


def dqn(env, cfg, rng, tracker) -> dict:
    spec = get_spec(cfg.env)
    S, A = n_states_actions(env)
    torch.manual_seed(cfg.seed)
    eye = torch.eye(S)                               # row s is the one-hot encoding of state s
    online, target = mlp(S, cfg.hidden, A), mlp(S, cfg.hidden, A)
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=cfg.lr)
    buffer = ReplayBuffer(cfg.buffer_size)
    sched = Schedules(cfg)

    @torch.no_grad()
    def q_table() -> np.ndarray:
        return online(eye).numpy().astype(np.float64)

    tracker.start(q_table().argmax(1))
    q_cache = q_table()                              # Q of all states, refreshed after each update
    total_steps = updates = 0
    losses = []
    for episode in range(cfg.episodes):
        eps = sched.epsilon(episode)
        s, _ = env.reset(seed=cfg.seed if episode == 0 else None)
        total, steps = 0.0, 0
        while True:
            a = epsilon_greedy(q_cache[s], eps, rng)
            s2, r, terminated, truncated, _ = env.step(a)
            buffer.add(s, a, r, s2, float(terminated))
            total += r
            steps += 1
            total_steps += 1
            if buffer.size >= cfg.learning_starts and total_steps % cfg.train_every == 0:
                bs, ba, br, bs2, bd = buffer.sample(cfg.batch_size, rng)
                with torch.no_grad():
                    y = torch.from_numpy(br) + cfg.gamma * (1 - torch.from_numpy(bd)) * target(eye[bs2]).max(1).values
                q = online(eye[bs]).gather(1, torch.from_numpy(ba)[:, None]).squeeze(1)
                loss = nn.functional.smooth_l1_loss(q, y)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), 10.0)
                optimizer.step()
                updates += 1
                losses.append(loss.item())
                if updates % cfg.target_update == 0:
                    target.load_state_dict(online.state_dict())
                q_cache = q_table()
            if terminated or truncated:
                break
            s = s2
        tracker.end_episode(episode + 1, total, steps, is_success(spec, terminated, total),
                            lambda: q_cache.argmax(1))
    Q = q_table()
    return {"Q": Q, "V": Q.max(1), "losses": np.array(losses), "model": online.state_dict()}


def reinforce(env, cfg, rng, tracker) -> dict:
    spec = get_spec(cfg.env)
    S, A = n_states_actions(env)
    torch.manual_seed(cfg.seed)
    eye = torch.eye(S)
    policy_net, value_net = mlp(S, cfg.hidden, A), mlp(S, cfg.hidden, 1)
    optimizer = torch.optim.Adam(list(policy_net.parameters()) + list(value_net.parameters()), lr=cfg.lr)

    @torch.no_grad()
    def probs_table() -> np.ndarray:
        return torch.softmax(policy_net(eye), -1).numpy().astype(np.float64)

    probs = probs_table()
    tracker.start(probs.argmax(1))
    batch: list[tuple[list, list, np.ndarray]] = []          # episodes waiting for the next update
    for episode in range(cfg.episodes):
        s, _ = env.reset(seed=cfg.seed if episode == 0 else None)
        states, actions, rewards = [], [], []
        terminated = truncated = False
        while not (terminated or truncated):
            p = probs[s]
            a = int(rng.choice(A, p=p / p.sum()))        # sample from pi(.|s): exploration is built in
            s2, r, terminated, truncated, _ = env.step(a)
            states.append(int(s))
            actions.append(a)
            rewards.append(float(r))
            s = s2
        # returns-to-go G_t, computed backwards
        G, returns = 0.0, np.zeros(len(rewards), dtype=np.float32)
        for t in range(len(rewards) - 1, -1, -1):
            G = rewards[t] + cfg.gamma * G
            returns[t] = G
        batch.append((states, actions, returns))
        if len(batch) == cfg.episodes_per_update or episode == cfg.episodes - 1:
            x = eye[np.concatenate([b[0] for b in batch])]
            a_t = torch.tensor(np.concatenate([b[1] for b in batch]))[:, None]
            G_t = torch.from_numpy(np.concatenate([b[2] for b in batch]))
            log_pi = torch.log_softmax(policy_net(x), -1)
            values = value_net(x).squeeze(1)
            advantage = (G_t - values).detach()               # baseline: what the critic expected
            policy_loss = -(log_pi.gather(1, a_t).squeeze(1) * advantage).mean()
            entropy = -(log_pi.exp() * log_pi).sum(1).mean()
            value_loss = ((G_t - values) ** 2).mean()
            loss = policy_loss + 0.5 * value_loss - cfg.entropy_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            probs = probs_table()
            batch = []
        total = float(sum(rewards))
        tracker.end_episode(episode + 1, total, len(rewards), is_success(spec, terminated, total),
                            lambda: probs.argmax(1))
    with torch.no_grad():
        V = value_net(eye).squeeze(1).numpy().astype(np.float64)
    # no Q for a policy-gradient method: store log pi as the "Q" so heatmaps / argmax still work
    return {"Q": np.log(probs + 1e-12), "V": V, "probs": probs, "model": policy_net.state_dict()}
