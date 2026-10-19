"""02: on-policy vs off-policy on CliffWalking (Sutton & Barto example 6.6), about 5 seconds.

Both agents explore with a constant epsilon = 0.1. Q-learning learns the values of the greedy
policy, so it learns the shortest path right along the cliff, and its exploring agent falls in
regularly. SARSA learns the values of the policy it actually follows, exploration included, so it
prefers the safer path one row up and collects more reward while learning.

    python examples/02_sarsa_vs_q_learning_cliff.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from frozenlake_rl import viz
from frozenlake_rl.config import default_config
from frozenlake_rl.train import train

OUT = Path("examples/output")
ENV = "cliffwalking"

runs = {m: train(default_config(ENV, m)) for m in ["sarsa", "expected-sarsa", "q-learning"]}
print()
for m, r in runs.items():
    print(f"{m:>15}: mean online return over the last 200 episodes {np.mean(r.train_returns[-200:]):7.1f} | "
          f"greedy policy return {r.summary['exact']['return']:.0f} (optimal -13)")

fig, axes = plt.subplots(2, 1, figsize=(8, 7), gridspec_kw={"height_ratios": [2, 1]})
viz.plot_train_returns({m: r.train_returns for m, r in runs.items()}, ENV, window=20, ax=axes[0], ylim=(-150, 0))
viz.plot_value_policy(runs["sarsa"].V, runs["sarsa"].policy, ENV, axes[1], "SARSA: greedy policy")
print("wrote", viz.save(fig, OUT / "02_cliff.png"))
