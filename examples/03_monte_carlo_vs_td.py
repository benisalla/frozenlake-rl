"""03: Monte Carlo vs TD on the slippery 4x4 lake, about 1 minute.

The same epsilon schedule, the same 10 000 episodes, the same exact evaluation; only the update
differs: complete returns (first-visit MC, every-visit MC) or one-step bootstrapped targets
(SARSA, Q-learning). The curve shows the success probability of each greedy policy during
training against the 74 % optimum computed by value iteration.

    python examples/03_monte_carlo_vs_td.py
"""

from pathlib import Path

import matplotlib.pyplot as plt

from frozenlake_rl import viz
from frozenlake_rl.config import default_config
from frozenlake_rl.train import optimal_score, train

OUT = Path("examples/output")
ENV = "frozenlake-4x4-slippery"

methods = ["mc-first-visit", "mc-every-visit", "sarsa", "q-learning"]
runs = {m: train(default_config(ENV, m, eval_episodes=500)) for m in methods}

fig, ax = plt.subplots(figsize=(8, 4.5))
viz.plot_curves({m: r.history for m, r in runs.items()}, ENV, ax, optimum=optimal_score(ENV)["success"])
print("wrote", viz.save(fig, OUT / "03_mc_vs_td.png"))
