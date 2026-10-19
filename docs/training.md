# Training

## One method, one environment

```bash
frozenlake-train --env frozenlake-4x4-slippery --method q-learning
frozenlake-train --env taxi --method expected-sarsa --episodes 2000 --alpha 0.3
frozenlake-train --env frozenlake-8x8-slippery --method value-iteration
frozenlake-train --env cliffwalking --method sarsa --epsilon-start 0.1 --epsilon-end 0.1
```

or from Python:

```python
from frozenlake_rl.config import default_config
from frozenlake_rl.train import train

result = train(default_config("frozenlake-4x4-slippery", "sarsa-lambda", lam=0.8), out_dir="runs/my_run")
result.Q, result.V, result.policy, result.history, result.summary
```

`default_config(env, method)` holds settings that work out of the box; every field of
`config.Config` is also a CLI flag (`--alpha-end`, `--planning-steps`, `--exploring-starts`, ...).

A run writes `runs/<env>/<method>/` (or `--out`):

| file | content |
|---|---|
| `agent.npz` | `Q` (S x A), `V` (S), `policy` (S), and `probs` for REINFORCE |
| `summary.json` | config, train time, episodes / env steps, episodes to reach the target, exact and rollout scores |
| `history.json` | exact greedy score at each evaluation point, online training returns |
| `run.png` | V heatmap + policy arrows, Q table, learning curve |
| `model.pt` | network weights (DQN, REINFORCE) |

## Default budgets

| env | tabular TD | Monte Carlo | Dyna-Q | DQN | REINFORCE |
|---|---|---|---|---|---|
| `frozenlake-4x4` | 1 000 episodes, alpha 0.5 -> 0.01 | 1 000 | 1 000 | 1 000 | 1 000 |
| `frozenlake-4x4-slippery` | 10 000, alpha 0.1 -> 0.01 | 20 000, alpha 0.05 -> 0.005 | 2 000 | 1 500 | 8 000 |
| `frozenlake-8x8` | 3 000, alpha 0.5 -> 0.01 | 3 000 | 3 000 | 2 000 | 4 000 |
| `frozenlake-8x8-slippery` | 20 000, alpha 0.1 -> 0.01 | 40 000, alpha 0.05 -> 0.005 | 2 500 | 2 000 | 15 000 |
| `taxi` | 3 000, alpha 0.5 -> 0.05 (n-step: 5 000, alpha 0.1) | | 1 000 | 1 500 | 5 000 |
| `cliffwalking` | 500, alpha 0.5, epsilon fixed at 0.1 | | 500 | 500 | 2 000 |

Exploration: epsilon goes linearly from 1.0 to 0.05 over the first 50 % of the episodes (70 %
on the 8x8 slippery lake, to 0.01 on Taxi), then stays there. CliffWalking keeps epsilon = 0.1
for the whole run on purpose: that is the setting in which SARSA and Q-learning differ.

### Why these numbers

* **The deterministic lakes** need only enough exploration to find the goal once; after that a
  large alpha (0.5) propagates the reward fast. The decay to 0.01 removes the jitter between
  near-equal actions (with gamma = 0.99, "bump into the wall and try again" is only 1 % worse
  than the right move, and a noisy Q can rank them wrongly).
* **The slippery lakes** are stochastic: a single transition says little, so alpha must be small
  enough to average over slips (0.1, decayed to 0.01), and many more episodes are needed.
* **Monte Carlo** targets are whole-episode returns, much noisier than TD targets, so it gets a
  smaller alpha and twice the episodes.
* **Dyna-Q** replays every real transition 10 times, so it needs a fraction of the episodes; its
  cost is CPU time per episode.
* **n-step SARSA on Taxi**: the larger n, the noisier the target, the smaller the step size must
  be (Sutton & Barto figure 7.2); with alpha 0.5 and n = 4 it does not converge on Taxi.
* **DQN** does one Adam step every 4 environment steps on a batch of 64 from a 20 000-transition
  buffer, target network synced every 100 updates. **REINFORCE** averages 10 episodes per update,
  lr 0.01, entropy bonus 0.01: with single-episode updates it gets stuck at 60 % on the
  slippery 4x4 lake and never finds the goal of the 8x8 lake.

## Comparing methods

```bash
frozenlake-compare --env frozenlake-4x4-slippery                     # every method
frozenlake-compare --env taxi --methods td                           # DP + TD family + Dyna-Q
frozenlake-compare --env cliffwalking --methods sarsa,q-learning --episodes 1000
frozenlake-compare --env frozenlake-8x8-slippery --episodes-scale 2  # double every budget
python scripts/compare_all.py                                        # the README results
```

Groups for `--methods`: `all`, `tabular`, `td`, `dp`, `mc`, `neural`, or a comma list.

## Watching a policy

```bash
frozenlake-play runs/frozenlake-4x4-slippery/q-learning --episodes 3
frozenlake-play runs/taxi/q-learning --gif examples/output/taxi.gif
```

## Compute

Everything is single-threaded (`--threads 1` sets torch and BLAS threads). One environment step
of a toy-text env costs about 8 microseconds, so a tabular run of 10 000 episodes takes 10 to 30
seconds in pure Python, including the evaluation. The neural methods cost 10 to 100 times more
per step because of the forward/backward passes.
