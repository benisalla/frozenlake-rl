"""04: faster credit assignment on the 8x8 lake: n-step, eligibility traces and Dyna-Q (~1 min).

The goal is 14 steps away and the only reward is there. One-step Q-learning moves that reward
back one state per visit; n-step SARSA and SARSA(lambda) move it along the whole path at once;
Dyna-Q replays each real transition 10 more times from its learned model. The printout gives the
number of real episodes each needs before its greedy policy always reaches the goal.

    python examples/04_traces_and_dyna.py
"""

from pathlib import Path

import matplotlib.pyplot as plt

from frozenlake_rl import viz
from frozenlake_rl.config import default_config
from frozenlake_rl.train import train

OUT = Path("examples/output")
ENV = "frozenlake-8x8"

methods = ["q-learning", "sarsa", "n-step-sarsa", "sarsa-lambda", "q-lambda", "dyna-q"]
runs = {m: train(default_config(ENV, m, eval_episodes=100, eval_points=150)) for m in methods}
print()
for m, r in runs.items():
    print(f"{m:>13}: solved after {r.summary['solved_at']} episodes ({r.summary['solved_steps']} env steps), "
          f"{r.summary['train_time_s']:.1f} s")

fig, ax = plt.subplots(figsize=(8, 4.5))
viz.plot_curves({m: r.history for m, r in runs.items()}, ENV, ax)
print("wrote", viz.save(fig, OUT / "04_traces_and_dyna.png"))
