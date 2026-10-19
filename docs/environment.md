# Environments

Every environment is a gymnasium *toy-text* environment: a finite set of states, a finite set
of actions, and the complete model available as `env.unwrapped.P`. That is what makes them the
right playground for classical RL: the exact optimum can be computed, so every learning method
can be measured against it instead of against a guess.

Pick one with `--env` (or the first argument of `make_env`):

| `--env` | alias | gymnasium | states x actions | time limit | reward | "solved" (target) |
|---|---|---|---|---|---|---|
| `frozenlake-4x4` | `fl4` | `FrozenLake-v1`, `is_slippery=False` | 16 x 4 | 100 | +1 at the goal | success = 1.0 |
| `frozenlake-4x4-slippery` | `fl4s` | `FrozenLake-v1` | 16 x 4 | 100 | +1 at the goal | success >= 0.70 (ceiling 0.744) |
| `frozenlake-8x8` | `fl8` | `FrozenLake-v1`, `map_name="8x8"`, not slippery | 64 x 4 | 200 | +1 at the goal | success = 1.0 |
| `frozenlake-8x8-slippery` | `fl8s` | `FrozenLake-v1`, `map_name="8x8"` | 64 x 4 | 200 | +1 at the goal | success >= 0.80 (ceiling 0.913) |
| `taxi` | | `Taxi-v4` (or `Taxi-v3` on older gymnasium) | 500 x 6 | 200 | -1 per step, +20 drop-off, -10 illegal pick-up/drop-off | mean return >= 7 (ceiling 7.93) |
| `cliffwalking` | `cliff` | `CliffWalking-v1` (or `-v0`) | 48 x 4 | 200 (added) | -1 per step, -100 and back to start on the cliff | mean return >= -13 (the optimum) |

The "ceiling" is the best value of the metric any policy can reach within the time limit,
computed by finite-horizon dynamic programming (`ExactEvaluator.best()`, see
[evaluation.md](evaluation.md)).

## FrozenLake

```
SFFF      S start, F frozen (safe), H hole (episode ends, reward 0), G goal (episode ends, reward 1)
FHFH
FFFH
HFFG
```

Actions: 0 = left, 1 = down, 2 = right, 3 = up. On **slippery** ice the agent moves in the
intended direction with probability 1/3 and to each of the two perpendicular directions with
probability 1/3. That is why the optimal policy often points *away* from a hole: pressing "left"
next to a hole on your right can only slip up or down, never right.

**Why it is a good teaching problem.** The reward is sparse (one +1 at the very end), so it
shows how each method spreads credit back along the path. The slippery version is stochastic,
so it shows the difference between a policy that is fast and one that is safe. Even the best
policy fails 26 % of the time on the 4x4 slippery lake, which is a good reminder that "solved"
means "as good as the optimum", not "always succeeds".

## Taxi

A 5x5 grid with four pick-up/drop-off locations R, G, Y, B. State = (taxi row, taxi column,
passenger location in R/G/Y/B/in-taxi, destination): 25 x 5 x 4 = 500 states. Actions: south,
north, east, west, pick-up, drop-off. Every step costs -1, so the return directly measures how
short the route is; the start state is random, so the optimal mean return is 7.93, not a single
number per episode. The figures draw two 5x5 slices of the 500 states (passenger waiting at R,
and passenger in the taxi; destination G).

## CliffWalking

A 4x12 grid; start bottom-left, goal bottom-right, the cells between them are the cliff. Each
step costs -1; stepping into the cliff costs -100 and sends the agent back to the start (the
episode does not end). The shortest path walks along the cliff edge (return -13). It is the
classic example of on-policy vs off-policy learning: see [architecture.md](architecture.md#sarsa-vs-q-learning-on-the-cliff).
gymnasium's newer `CliffWalking-v1` has no time limit, so `make_env` adds one of 200 steps.

## The model P

```python
from frozenlake_rl.envs import make_env, transition_model
P = transition_model(make_env("fl4s"))
P[14][2]   # [(0.333, 14, 0, False), (0.333, 15, 1, True), (0.333, 10, 0, False)]: slide down (wall), goal, slide up
```

`P[s][a]` lists the possible outcomes of action `a` in state `s` as
`(probability, next_state, reward, terminated)`. Only dynamic programming uses it to learn, and
the evaluator uses it to score policies exactly. The learning agents only see `reset()` /
`step()`.

## Gymnasium versions

gymnasium 1.x renamed `Taxi-v3` to `Taxi-v4` and `CliffWalking-v0` to `CliffWalking-v1` and
refuses the old ids. `make_env` tries the newest id first and falls back to the old one, so the
code runs on gymnasium 1.0 up to the current release (1.4 was used for the results).
