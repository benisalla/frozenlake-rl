# Evaluation

Every method is scored the same way, by `evaluate.py`, on its final **greedy** policy (argmax of
Q; argmax of pi for REINFORCE).

## Two measurements

**Exact** (`ExactEvaluator`). For a policy pi and the time limit H of the environment:

    v_0 = 0,   v_k(s) = sum_a pi(a|s) sum_s' p(s'|s,a) [ x(s,a,s') + (1 - done) v_{k-1}(s') ]

with `x` = the reward (expected return) or the indicator "this transition is a success" (success
probability), and the result averaged over the start distribution. It is exact, noise-free and
costs a few hundred matrix-vector products, so the tracker can score the policy 100 times per
run. Only the experimenter uses P here; agents (except DP) never see it.

**Rollouts** (`rollout_evaluate`, `frozenlake-eval`): 1 000 real gymnasium episodes with the
greedy policy, seeds 10 000+. This is what you would measure without knowing the model. The two
agree within the Monte Carlo error (about +-0.014 for p = 0.74 and N = 1000; tested in
`test_exact_and_rollout_evaluation_agree`).

**Why both?** A learning curve from rollouts of a slippery lake is so noisy that "episodes to
reach 0.70" would mostly measure luck; the exact score removes that noise. The rollout number
keeps everybody honest: it is the plain "play 1 000 games and count" figure.

## Success, target and ceiling

| env | success | target ("solved") | ceiling within the time limit |
|---|---|---|---|
| FrozenLake | reached G (reward 1) | success >= 1.0 / 0.70 / 1.0 / 0.80 | 1.0 / 0.744 / 1.0 / 0.913 |
| Taxi | passenger delivered | mean return >= 7 | 7.93 |
| CliffWalking | reached G | mean return >= -13 | -13 |

The ceiling comes from finite-horizon DP that maximises the metric itself (`ExactEvaluator.best()`).
Value iteration with gamma = 0.99 optimises the discounted return, a slightly different
objective: on the 8x8 slippery lake its policy scores 0.863 while the success-maximising
time-dependent policy reaches 0.913, which is why some learned policies (n-step SARSA 0.879,
Dyna-Q 0.875) score above value iteration there.

## Episodes to target

`metrics.Tracker` scores the greedy policy at 100 evenly spaced moments of the run (every sweep
for DP); "episodes to target" is the first of them where the exact score met the target, so its
resolution is 1 % of the budget. Greedy policies can dip after reaching the target (a slip
changes a near-tie), which the curves show.

## The runs behind the README

`python scripts/compare_all.py` (one CPU thread, seed 0, default configs, 1 000 rollout episodes):

| suite | methods | wall time |
|---|---|---|
| frozenlake-4x4 | all 14 | 0.3 min |
| frozenlake-4x4-slippery | all 14 | 1.5 min |
| frozenlake-8x8 | all 14 | 1.0 min |
| frozenlake-8x8-slippery | all 14 | 8.3 min |
| taxi | DP + TD + Dyna-Q (10) | 1.0 min |
| cliffwalking | DP + TD + Dyna-Q (10) | 0.2 min |

The full tables are in [assets/results.md](../assets/results.md) and in the README; per-run
files (`agent.npz`, `summary.json`, `history.json`, `run.png`) are written to `runs/compare/`.

## What the results say

* **DP** solves every problem in under a second, but only because it reads P.
* **Monte Carlo** is fine on short problems (4x4, 8x8 deterministic) and on the slippery 4x4 lake
  with a small step size, but on the 8x8 slippery lake its returns are too noisy: first-visit MC
  reaches 0.62 and every-visit MC fails (0.03). Its `--sample-average` mode (1/N steps) is worse
  than a constant alpha in control, because it never forgets the returns of the early policy.
* **One-step TD** methods all solve every FrozenLake and Taxi; Q-learning is the fastest of them.
* **SARSA vs Q-learning** on the cliff reproduces Sutton & Barto's example 6.6.
* **Multi-step** methods propagate sparse rewards faster (8x8 deterministic: 1 640-1 680 episodes
  vs 1 980), but larger n needs a smaller step size: n-step SARSA with n = 4 and alpha = 0.5 does
  not converge on Taxi; alpha = 0.1 fixes it.
* **Dyna-Q** needs the fewest real episodes everywhere (3 to 10 times fewer than Q-learning),
  paying in CPU time per episode.
* **DQN and REINFORCE** reach the tabular solutions on one-hot states, with more wall time per
  episode; REINFORCE needs batched updates (10 episodes) to escape a 60 % local optimum.

## Limitations

One seed per run: seed-to-seed variation was not measured, so on the slippery lakes differences of
a few hundredths between methods in the tables should not be read as meaningful. The notebook runs
the same comparison with larger budgets (`EPISODES_SCALE`).
