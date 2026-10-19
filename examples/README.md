# Examples

Short scripts that use the `frozenlake_rl` package from Python, in the order the methods build
on each other. All run on one CPU thread in seconds to a minute; run them from the repository
root after `pip install -e ".[dev]"`. Figures go to `examples/output/`.

| script | what it shows | time |
|---|---|---|
| [01_value_iteration.py](01_value_iteration.py) | dynamic programming with the model P: value iteration and policy iteration give the same V*, the optimal slippery-lake policy succeeds 74 % of the time | < 1 s |
| [02_sarsa_vs_q_learning_cliff.py](02_sarsa_vs_q_learning_cliff.py) | on-policy vs off-policy on CliffWalking: SARSA takes the safe path, Q-learning the optimal edge and falls while exploring | ~5 s |
| [03_monte_carlo_vs_td.py](03_monte_carlo_vs_td.py) | complete returns (first/every-visit MC) vs one-step bootstrapping (SARSA, Q-learning), same budget, same evaluation | ~1 min |
| [04_traces_and_dyna.py](04_traces_and_dyna.py) | credit assignment on the 8x8 lake: n-step SARSA, SARSA(lambda), Q(lambda) and Dyna-Q need far fewer episodes than one-step TD | ~1 min |
| [05_dqn_and_reinforce.py](05_dqn_and_reinforce.py) | DQN and REINFORCE on one-hot states reach the tabular solution; plays one episode | ~1 min |
| [06_compare_everything.py](06_compare_everything.py) | `compare()`: every method, one table + curves + policies; takes an env name and a method group | ~1 min (4x4) |

```bash
python examples/01_value_iteration.py
python examples/02_sarsa_vs_q_learning_cliff.py
python examples/03_monte_carlo_vs_td.py
python examples/04_traces_and_dyna.py
python examples/05_dqn_and_reinforce.py
python examples/06_compare_everything.py
```

Any single method is also one command from the CLI, for example
`frozenlake-train --env frozenlake-4x4-slippery --method expected-sarsa`, then
`frozenlake-play runs/frozenlake-4x4-slippery/expected-sarsa`.
