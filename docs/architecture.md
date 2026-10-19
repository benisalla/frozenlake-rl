# Architecture: every method, its equation, and why it works

The code base is organised so that **methods differ only where they really differ**: what they
update and towards which target. Exploration (epsilon-greedy, `common.py`), step sizes
(`Schedules`), evaluation (`evaluate.py`), history (`metrics.Tracker`), saving and figures
(`train.py`, `viz.py`) are shared, so a comparison measures the methods, not their plumbing.

```
envs.py         make_env(name), the model P, success definition
config.py       Config dataclass + tuned defaults per (env, method)
common.py       epsilon-greedy, schedules, greedy policy
algorithms/
  dp.py           value_iteration, policy_iteration                       (needs P)
  monte_carlo.py  mc_control (first-visit / every-visit, exploring starts)
  td.py           sarsa, expected_sarsa, q_learning, double_q_learning,
                  n_step_sarsa, sarsa_lambda, q_lambda
  planning.py     dyna_q (+ CountModel)
  neural.py       dqn, reinforce (one-hot states, small torch MLPs)
evaluate.py     ExactEvaluator (from P), rollout_evaluate (real episodes), frozenlake-eval
metrics.py      Tracker: scores the greedy policy during training
train.py        train(cfg) -> RunResult, frozenlake-train
compare.py      compare(env, methods), frozenlake-compare
play.py         frozenlake-play (terminal or GIF)
viz.py          heatmaps, arrows, Q tables, curves
```

Every algorithm has the same signature, `fn(env, cfg, rng, tracker) -> {"Q", "V", ...}`, and
calls `tracker.end_episode(...)` after each episode. Shapes: `Q` is `(S, A)`, `V` and the
greedy `policy` are `(S,)`.

## Notation

`s, a, r, s'`: state, action, reward, next state. `gamma`: discount (0.99 on FrozenLake and Taxi,
1.0 on CliffWalking). `alpha`: step size. `epsilon`: exploration rate. `done`: the transition
ended the episode (hole, goal, drop-off), in which case nothing is bootstrapped after it. A
*truncation* by the time limit is not `done`: the state still has a future, so TD methods keep
the bootstrap term there.

The Bellman equations every method is built on:

$$q_\pi(s,a) = \mathbb{E}\big[r + \gamma\, q_\pi(s', a')\big], \quad a' \sim \pi(\cdot|s')$$
$$q_*(s,a) = \mathbb{E}\big[r + \gamma \max_{a'} q_*(s', a')\big]$$

## 1. Dynamic programming (`algorithms/dp.py`)

**Value iteration** applies the optimality equation as an update to all states at once:

$$V_{k+1}(s) = \max_a \sum_{s'} p(s'|s,a)\,[r + \gamma (1-\text{done}) V_k(s')]$$

`model_arrays` turns P into dense arrays `T[s, a, s']` (probability of continuing to s') and
`R[s, a]` (expected reward), so one sweep is `(R + gamma * T @ V).max(1)`. Stop when the largest
change is below `theta = 1e-8`.

**Policy iteration**: evaluate the current policy until convergence
(`V <- R_pi + gamma T_pi V`), then improve it greedily, until the policy is stable. Ties keep the
old action, otherwise the loop can cycle between equally good policies.

*Why it works*: the Bellman backup is a gamma-contraction in max-norm, so the iteration converges
to the unique fixed point from any start. *Why it is not "learning"*: it needs P. Every other
method replaces the sum over s' by samples.

## 2. Monte Carlo control (`algorithms/monte_carlo.py`)

Play one episode with epsilon-greedy(Q), compute the returns backwards
`G_t = r_{t+1} + gamma G_{t+1}`, and move Q towards them:

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha\,[G_t - Q(s_t, a_t)]$$

* `mc-first-visit`: only the first occurrence of (s, a) in the episode is updated (unbiased).
* `mc-every-visit`: every occurrence is updated. On FrozenLake the agent often bumps into a wall
  and stays in place, so the same pair can be updated ten times in one episode with the same
  noisy return: every-visit learns visibly slower and noisier there.
* `--exploring-starts`: start in a random non-terminal state with a random first action
  (Sutton & Barto 5.3), implemented by setting `env.unwrapped.s` after `reset()`.
* `--sample-average`: the textbook `1/N(s, a)` step size. It is the right estimator for a fixed
  policy, but in control the policy keeps improving, and averaging the returns of the early
  random policy forever makes Q lag behind. A constant (slowly decaying) alpha forgets them; on
  the slippery 4x4 lake it reaches 0.74 where the sample average stalls at 0.51.

*Why it is slow on hard problems*: `G_t` is a sum of many random rewards, so its variance grows
with the episode length, and on a sparse-reward problem nothing is learned until the goal is
reached by chance.

## 3. One-step temporal difference (`algorithms/td.py`)

All of them update after every step:

$$Q(s,a) \leftarrow Q(s,a) + \alpha\,[\underbrace{r + \gamma\, X(s')}_{\text{target}} - Q(s,a)]$$

| method | X(s') | policy it learns about |
|---|---|---|
| `sarsa` | `Q(s', a')`, a' the next action actually taken | the epsilon-greedy one it follows (on-policy) |
| `expected-sarsa` | `sum_a' pi(a'|s') Q(s', a')` | the epsilon-greedy one; no noise from sampling a' |
| `q-learning` | `max_a' Q(s', a')` | the greedy one (off-policy) |
| `double-q-learning` | `Q2(s', argmax_a' Q1(s', a'))`, roles swapped at random | the greedy one, without the max bias |

The three first share one loop, `_one_step_td`, with the target as a one-line lambda, so the
code shows exactly how small the difference is.

**Double Q-learning**: `max_a Q(s', a)` over noisy estimates is biased upwards (the max picks
the action whose noise happened to be positive). Using one table to choose the action and the
other to evaluate it removes that bias (van Hasselt 2010). Behaviour is epsilon-greedy on
`Q1 + Q2`. Each table gets half the updates, so it learns a little slower.

### SARSA vs Q-learning on the cliff

With constant epsilon = 0.1, Q-learning's values are those of the greedy policy, which walks
along the cliff edge (-13). But the agent that *collects* the experience still explores, and one
random step down from the edge costs -100. SARSA's values include that risk, so its greedy
policy is the path one or two rows up (-15 to -17), and it collects more reward while learning.
`frozenlake-compare --env cliffwalking --methods sarsa,q-learning` reproduces the textbook figure
(`train_returns.png`).

## 4. Multi-step methods (`algorithms/td.py`)

One-step TD moves a reward back one state per visit. Two ways to move it further at once:

**n-step SARSA** (`--n`, default 4): wait n steps, then use n real rewards and bootstrap:

$$G_{t:t+n} = r_{t+1} + \gamma r_{t+2} + \dots + \gamma^{n-1} r_{t+n} + \gamma^n Q(s_{t+n}, a_{t+n})$$

n = 1 is SARSA, n = infinity is Monte Carlo. The code follows Sutton & Barto's pseudo-code
(section 7.2): `tau = t - n + 1` is the time being updated.

**Eligibility traces** (`sarsa-lambda`, `q-lambda`, `--lam`, default 0.9): every pair keeps a
trace `E(s, a)`, set to 1 when visited (replacing traces) and multiplied by `gamma * lambda`
each step. Each TD error updates every pair in proportion to its trace:

$$\delta = r + \gamma X(s') - Q(s,a), \qquad Q \leftarrow Q + \alpha\,\delta\,E, \qquad E \leftarrow \gamma\lambda E$$

It is the online, incremental form of averaging all n-step returns with weights
`(1 - lambda) lambda^(n-1)`. **Watkins's Q(lambda)** uses `X = max Q` and resets all traces after
an exploratory action, since the rest of the path is no longer what the greedy policy would do.

## 5. Dyna-Q (`algorithms/planning.py`)

Q-learning, plus after each real step:

1. record the transition in a learned model (`CountModel`: counts of `(s', r, done)` per
   `(s, a)`, so sampling it reproduces the slippery probabilities);
2. do `planning_steps` (default 10) more Q-learning updates on transitions sampled from the
   model at previously visited pairs.

Every real step is reused many times, which is why Dyna-Q needs the fewest real episodes in
the results. `--planning-steps 0` is exactly Q-learning. The cost moves from the environment to
the CPU: Dyna-Q has the highest wall time per episode.

## 6. Neural methods on one-hot states (`algorithms/neural.py`)

The state index becomes a one-hot vector `e_s` (size S), so `Linear(S, hidden)` is a table of
`hidden`-dimensional embeddings, one per state. Networks: `S -> 64 -> A` (128 hidden on Taxi).

**DQN**: epsilon-greedy on `Q_theta`, transitions stored in a `ReplayBuffer` (20 000), one Adam
step every 4 env steps on a batch of 64, target network synced every 100 updates:

$$L(\theta) = \text{Huber}\big(Q_\theta(s,a),\; r + \gamma (1-\text{done}) \max_{a'} Q_{\bar\theta}(s',a')\big)$$

The replay buffer breaks the correlation between consecutive samples; the target network keeps
the regression target still for a while, which stops the "chasing its own tail" divergence.

**REINFORCE with a value baseline**: sample actions from `softmax(policy_net(e_s))` (no epsilon:
the policy's own randomness explores), collect `episodes_per_update` (10) episodes, then minimise
the mean over all their steps of

$$-\log \pi_\theta(a_t|s_t)\,(G_t - V_\phi(s_t)) + \tfrac12 (G_t - V_\phi(s_t))^2 - \beta\, H(\pi_\theta(\cdot|s_t))$$

The baseline does not change the expected gradient but removes most of its variance. Averaging
ten episodes per update matters a lot on FrozenLake: with one episode per update, most gradients
come from failed episodes, and the policy settles in a 60 % local optimum on the slippery 4x4
lake; with ten it reaches 73 %, and it solves the 8x8 lake that it could not solve before.

For uniformity a REINFORCE run stores `log pi` as its "Q" (so `argmax` gives the greedy action)
and the critic `V_phi` as its V.

## Shared pieces

* **Exploration** (`common.epsilon_greedy`): random action with probability epsilon, otherwise a
  greedy one with *random tie-breaking*. Without it, an all-zero Q table always picks action 0
  (left), and on FrozenLake the agent would bump into the wall forever.
* **Schedules** (`common.Schedules`): epsilon decays linearly from 1.0 to 0.05 over the first half
  of the episodes (70 % on the 8x8 slippery lake; constant 0.1 on the cliff), alpha decays
  linearly to `alpha_end` when set. A decaying alpha matters on stochastic problems: a constant
  alpha keeps Q jumping around with every slip, and the greedy policy flips between near-equal
  actions.
* **Seeds**: `cfg.seed` seeds the numpy Generator used by the agent, the first `env.reset`, the
  action space and torch. Two runs with the same config give the same Q table (tested).
