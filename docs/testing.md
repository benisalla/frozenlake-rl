# Testing

```bash
OMP_NUM_THREADS=1 pytest -q          # 40 tests, about 45 s on one CPU
```

| file | what it checks |
|---|---|
| `test_envs.py` | every env builds with its time limit and model P; sizes; aliases; holes and goal are detected as terminal; success definition; one-hot encoding |
| `test_common.py` | linear schedules, alpha / epsilon schedules from the config, epsilon-greedy probabilities and random tie-breaking, default configs for every (env, method) |
| `test_dp_and_evaluation.py` | value and policy iteration agree; the slippery 4x4 optimum is 0.74; V* satisfies the Bellman optimality equation; the exact evaluator matches 2 000 real episodes within Monte Carlo error; stochastic policies; the Taxi ceiling |
| `test_learning.py` | **every tabular method solves the 4x4 lake** in 600 episodes; MC with exploring starts; DQN and REINFORCE learn it too; SARSA earns more than Q-learning online on the cliff while Q-learning's greedy policy is optimal; Dyna-Q with 400 episodes beats Q-learning with the same 400 episodes and is 3x closer to V*; runs are reproducible from the seed |
| `test_cli_and_outputs.py` | `frozenlake-train` / `frozenlake-eval` / `play` / `compare` write and read their files; method groups; the notebook builder produces a valid notebook |

The learning tests are the important ones: they do not only check shapes, they check that each
algorithm reaches the known optimum on a problem small enough to run in about a second.
