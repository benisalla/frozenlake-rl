"""05: neural networks on one-hot states: DQN and REINFORCE on the 4x4 lake (~1 minute).

A one-hot state makes the first linear layer a lookup table, so the networks can represent the
exact solution; the example checks that the deep-RL training loops (replay buffer + target
network, policy gradient + value baseline) find it, and compares their learned Q / policy with
the tabular one. Then it plays one episode of the DQN policy in the terminal.

    python examples/05_dqn_and_reinforce.py
"""

from pathlib import Path

import matplotlib.pyplot as plt

from frozenlake_rl import viz
from frozenlake_rl.config import default_config
from frozenlake_rl.play import play
from frozenlake_rl.train import train

OUT = Path("examples/output")
ENV = "frozenlake-4x4"

runs = {m: train(default_config(ENV, m, eval_episodes=200)) for m in ["q-learning", "dqn", "reinforce"]}

fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
for ax, (m, r) in zip(axes, runs.items(), strict=False):
    viz.plot_value_policy(r.V, r.policy, ENV, ax, f"{m}: V(s) and policy")
viz.plot_curves({m: r.history for m, r in runs.items()}, ENV, axes[3])
print("wrote", viz.save(fig, OUT / "05_neural.png"))

print("\nDQN policy, one episode:")
play(ENV, runs["dqn"].policy, episodes=1, delay=0.0)
