"""01: solve the slippery 4x4 lake exactly with dynamic programming (under a second).

Value iteration and policy iteration read the model P of the environment and compute the
optimal values and policy. The script prints V as a grid, the greedy arrows, the success
probability of that policy (about 74 %, the best anyone can do on this ice), and saves a figure.

    python examples/01_value_iteration.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from frozenlake_rl import viz
from frozenlake_rl.config import default_config
from frozenlake_rl.envs import make_env
from frozenlake_rl.train import train

OUT = Path("examples/output")
ENV = "frozenlake-4x4-slippery"
ARROWS = np.array(["<", "v", ">", "^"])

vi = train(default_config(ENV, "value-iteration"))
pi = train(default_config(ENV, "policy-iteration"))

print("\nV* (value iteration):")
print(np.round(vi.V.reshape(4, 4), 3))
print("\ngreedy policy (H hole, G goal):")
desc = make_env(ENV).unwrapped.desc.astype(str)
print(np.where(np.isin(desc, ["H", "G"]), desc, ARROWS[vi.policy].reshape(4, 4)))
print(f"\nvalue iteration: {vi.summary['sweeps']} sweeps | policy iteration: {pi.extras['improvements']} "
      f"improvement steps ({pi.summary['sweeps']} evaluation sweeps in total)")
print(f"same values: {np.allclose(vi.V, pi.V, atol=1e-6)}")
print(f"success probability of the optimal policy: {vi.summary['exact']['success']:.3f} "
      f"(1000 real episodes: {vi.summary['rollout']['success_rate']:.3f})")

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
viz.plot_value_policy(vi.V, vi.policy, ENV, axes[0], "value iteration")
viz.plot_value_policy(pi.V, pi.policy, ENV, axes[1], "policy iteration")
print("wrote", viz.save(fig, OUT / "01_value_iteration.png"))
