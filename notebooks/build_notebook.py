"""Build notebooks/train_colab_kaggle.ipynb from the package source.

The notebook installs the package from GitHub, then walks through every method: markdown
explains the idea, the method's real code is shown (copied from src/ at build time, so it is
always in sync), and a code cell runs it from the package. The last part runs the full
comparison on every environment with larger budgets. Regenerate after changing src/:

    python notebooks/build_notebook.py . notebooks/train_colab_kaggle.ipynb
"""

import ast
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "notebooks" / "train_colab_kaggle.ipynb"
SRC = ROOT / "src" / "frozenlake_rl"

cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


def source(module: str, name: str) -> str:
    """Source of one top-level function or class of src/frozenlake_rl/<module>.py."""
    text = (SRC / f"{module}.py").read_text()
    for node in ast.parse(text).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            return ast.get_source_segment(text, node)
    raise KeyError(f"{name} not in {module}")


def show(module: str, *names: str) -> None:
    """A markdown cell with the package code of these functions (read-only view)."""
    body = "\n\n".join(source(module, n) for n in names)
    md(f"The package code (`src/frozenlake_rl/{module}.py`):\n\n```python\n{body}\n```")


# ---------------------------------------------------------------------------------------------
md("""
# frozenlake-rl: every classical RL method on FrozenLake, Taxi and CliffWalking

This notebook goes through the methods of the `frozenlake-rl` repository one by one, in the
order they build on each other, and then runs the full comparison:

| part | methods | needs the model P? | learns from |
|---|---|---|---|
| 1. Dynamic programming | value iteration, policy iteration | yes | nothing, it computes |
| 2. Monte Carlo | first-visit / every-visit MC control | no | complete episodes |
| 3. Temporal difference | SARSA, expected SARSA, Q-learning, double Q-learning | no | single steps |
| 4. Multi-step | n-step SARSA, SARSA(lambda), Q(lambda) | no | several steps / traces |
| 5. Planning | Dyna-Q | learns its own | real + simulated steps |
| 6. Neural | DQN, REINFORCE on one-hot states | no | replay / whole episodes |

Everything runs on a CPU (Colab or Kaggle without GPU is fine). The small problems take
seconds; the full comparison at the end takes 20 to 40 minutes.
""")
code("""
# Install the package (Kaggle: turn Internet on in the settings panel).
# The repo is private: on Colab/Kaggle, add a GitHub token as the secret GITHUB_TOKEN, or
# upload the repo folder and run `pip install -e <folder>` instead.
import os, subprocess, sys
token = os.environ.get("GITHUB_TOKEN", "")
try:
    from google.colab import userdata  # Colab secrets
    token = token or userdata.get("GITHUB_TOKEN")
except Exception:
    pass
try:
    from kaggle_secrets import UserSecretsClient  # Kaggle secrets
    token = token or UserSecretsClient().get_secret("GITHUB_TOKEN")
except Exception:
    pass
url = f"git+https://{token + '@' if token else ''}github.com/benisalla/frozenlake-rl"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", url], check=True)
""")
code("""
import numpy as np
import matplotlib.pyplot as plt

from frozenlake_rl.common import set_threads
from frozenlake_rl.config import default_config
from frozenlake_rl.envs import make_env, transition_model
from frozenlake_rl.train import train, optimal_score
from frozenlake_rl.evaluate import ExactEvaluator, rollout_evaluate
from frozenlake_rl.compare import compare
from frozenlake_rl import viz

set_threads(1)
%matplotlib inline
""")

# ---------------------------------------------------------------------------------------------
md("""
## 0. The problem: a finite Markov decision process

FrozenLake is a grid: start `S`, frozen cells `F`, holes `H`, goal `G`. Reaching the goal gives
reward 1, everything else 0; falling in a hole ends the episode. On *slippery* ice the agent
moves in the intended direction only 1/3 of the time and slides to one of the two
perpendicular directions otherwise.

An MDP is: states S, actions A, and the dynamics p(s', r | s, a). Toy-text environments give
the dynamics explicitly as `env.unwrapped.P[s][a] = [(prob, s', r, terminated), ...]`. Only
dynamic programming reads it; every other method only calls `reset()` and `step()`.

**Why does it matter?** With P you can *compute* the best policy; without it you must *try*
actions and learn from the results, which is the whole difficulty of reinforcement learning.
""")
code("""
env = make_env("frozenlake-4x4-slippery", render_mode="ansi")
env.reset(seed=0)
print(env.render())
P = transition_model(env)
print("from state 14, action RIGHT (2):", P[14][2])   # 3 outcomes, 1/3 each, one reaches the goal
""")
md("""
The value of a policy pi is the expected discounted return from each state, and it obeys the
**Bellman equation** (one step of the dynamics, then the value of where you land):

$$v_\\pi(s) = \\sum_a \\pi(a|s) \\sum_{s', r} p(s', r|s, a)\\,[r + \\gamma v_\\pi(s')]$$

The optimal value obeys the **Bellman optimality equation**, with a max instead of the average
over actions:

$$v_*(s) = \\max_a \\sum_{s', r} p(s', r|s, a)\\,[r + \\gamma v_*(s')], \\qquad
q_*(s, a) = \\sum_{s', r} p(s', r|s, a)\\,[r + \\gamma \\max_{a'} q_*(s', a')]$$

Every method below is a different way of solving (or approximately solving) one of these
equations: DP iterates them with the true p, MC replaces the expectation by full sampled
returns, TD replaces it by one sampled step plus the current estimate.
""")

# ---------------------------------------------------------------------------------------------
md("""
## 1. Dynamic programming: value iteration and policy iteration

**Value iteration** turns the optimality equation into an update and applies it to every state
until nothing changes: `V(s) <- max_a sum p [r + gamma V(s')]`. Each sweep propagates the
goal's reward one step further back.

**Policy iteration** alternates two steps: *evaluate* the current policy completely (solve the
Bellman equation for v_pi), then *improve* it greedily. It needs very few improvement steps
(about 7 here, against hundreds of value-iteration sweeps) because each one is a full, exact evaluation.

Why does it work? The Bellman backup is a gamma-contraction: every sweep shrinks the error by
at least a factor gamma, so it converges to the unique fixed point v_*.
""")
show("algorithms/dp", "value_iteration", "policy_iteration")
code("""
vi = train(default_config("frozenlake-4x4-slippery", "value-iteration"))
pi = train(default_config("frozenlake-4x4-slippery", "policy-iteration"))
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
viz.plot_value_policy(vi.V, vi.policy, "frozenlake-4x4-slippery", axes[0], "value iteration")
viz.plot_value_policy(pi.V, pi.policy, "frozenlake-4x4-slippery", axes[1], "policy iteration")
plt.show()
print("same values:", np.allclose(vi.V, pi.V, atol=1e-6), "| sweeps VI:", vi.summary["sweeps"],
      "| improvements PI:", pi.extras["improvements"])
""")
md("""
Look at the arrows: on slippery ice the optimal policy often points *away* from the hole
(left in the first column, up on the top row), because moving "left" can only slip up or
down, never into the hole on the right. Even the best policy only succeeds about 74 % of the
time: that number is the ceiling every learning method below is measured against.
""")

md("""
### How every method is scored

The same two measurements for every method:

* **exact**: the success probability (or expected return) of the greedy policy computed from
  P with the 100/200-step time limit. No sampling noise, so the learning curves are smooth.
  Only the experimenter uses P here; the agents never see it.
* **rollouts**: N real gymnasium episodes with the greedy policy, the number you would get
  without knowing the model. Both agree within the Monte Carlo error.
""")
show("evaluate", "ExactEvaluator")
code("""
print("exact   :", ExactEvaluator("frozenlake-4x4-slippery")(vi.policy))
print("rollouts:", rollout_evaluate("frozenlake-4x4-slippery", vi.policy, episodes=2000))
print("uniform random policy:", ExactEvaluator("frozenlake-4x4-slippery")(np.full((16, 4), 0.25)))
""")

# ---------------------------------------------------------------------------------------------
md("""
## 2. Monte Carlo control: learn from complete episodes

No model now. Play a whole episode with an epsilon-greedy policy, compute the actual return
G_t that followed each (s, a), and average it into Q(s, a). Then act greedily w.r.t. the new
Q (with epsilon exploration). This is *generalised policy iteration* with sampled evaluation.

* **first-visit** averages only the first occurrence of (s, a) per episode (unbiased);
  **every-visit** averages all of them (more updates, slightly biased, converges too).
* **exploring starts** begin each episode in a random state with a random action, so every
  pair is tried even if the policy never goes there.

Why is it slow on big problems? G_t sums many random rewards, so its variance grows with the
episode length, and on FrozenLake most episodes end in a hole with G = 0: nothing is learned
until the goal is hit by chance.
""")
show("algorithms/monte_carlo", "mc_control")
code("""
mc = train(default_config("frozenlake-4x4-slippery", "mc-first-visit"))
mc_es = train(default_config("frozenlake-4x4-slippery", "mc-first-visit", exploring_starts=True))
viz.plot_curves({"mc first-visit": mc.history, "mc + exploring starts": mc_es.history},
                "frozenlake-4x4-slippery", optimum=vi.summary["optimum"]); plt.show()
""")

# ---------------------------------------------------------------------------------------------
md("""
## 3. Temporal-difference learning: learn from every single step

TD methods update after each step towards a target that uses the *current estimate* of the
next state (bootstrapping): `Q(s,a) <- Q(s,a) + alpha [target - Q(s,a)]`.

| method | target | on/off policy |
|---|---|---|
| SARSA | r + gamma Q(s', a') with the a' actually taken next | on-policy: learns the value of the exploring policy |
| expected SARSA | r + gamma sum_a' pi(a'|s') Q(s', a') | on-policy, no sampling noise from a' |
| Q-learning | r + gamma max_a' Q(s', a') | off-policy: learns the greedy policy while exploring |
| double Q-learning | r + gamma Q2(s', argmax Q1(s', .)) | off-policy, removes the max over noisy estimates bias |

**On-policy vs off-policy** in one picture: CliffWalking. With epsilon = 0.1 fixed, Q-learning
learns the optimal path right along the cliff, but its exploring agent keeps falling in
(-100). SARSA accounts for its own exploration and learns the safer path one row up: worse
greedy policy, better returns while learning (Sutton & Barto, example 6.6).
""")
show("algorithms/td", "_one_step_td", "sarsa", "expected_sarsa", "q_learning")
code("""
cliff = {m: train(default_config("cliffwalking", m)) for m in ["sarsa", "expected-sarsa", "q-learning"]}
viz.plot_train_returns({m: r.train_returns for m, r in cliff.items()}, "cliffwalking", window=20, ylim=(-150, 0))
plt.show()
fig, axes = plt.subplots(1, 2, figsize=(12, 2.4))
viz.plot_value_policy(cliff["sarsa"].V, cliff["sarsa"].policy, "cliffwalking", axes[0], "SARSA")
viz.plot_value_policy(cliff["q-learning"].V, cliff["q-learning"].policy, "cliffwalking", axes[1], "Q-learning")
plt.show()
""")
show("algorithms/td", "double_q_learning")
code("""
td = {m: train(default_config("frozenlake-4x4-slippery", m))
      for m in ["sarsa", "expected-sarsa", "q-learning", "double-q-learning"]}
viz.plot_curves({m: r.history for m, r in td.items()}, "frozenlake-4x4-slippery",
                optimum=vi.summary["optimum"]); plt.show()
""")

# ---------------------------------------------------------------------------------------------
md("""
## 4. Between MC and TD: n-step returns and eligibility traces

One-step TD propagates a reward back by one state per visit; MC propagates it along the whole
episode but with high variance. **n-step SARSA** uses n real rewards then bootstraps:

$$G_{t:t+n} = r_{t+1} + \\gamma r_{t+2} + \\dots + \\gamma^{n-1} r_{t+n} + \\gamma^n Q(s_{t+n}, a_{t+n})$$

**Eligibility traces** do the same online and for all n at once (the lambda-return): each
visited pair keeps a trace E(s, a) that fades by gamma*lambda per step, and every TD error
updates all pairs in proportion to their trace. One reward at the goal immediately updates
the whole path that led to it. **Q(lambda)** (Watkins) is the off-policy version: it cuts the
traces after an exploratory action, because what followed is not the greedy path.
""")
show("algorithms/td", "n_step_sarsa", "_lambda_control")
code("""
multi = {m: train(default_config("frozenlake-8x8", m)) for m in ["sarsa", "n-step-sarsa", "sarsa-lambda", "q-lambda"]}
viz.plot_curves({m: r.history for m, r in multi.items()}, "frozenlake-8x8"); plt.show()
""")

# ---------------------------------------------------------------------------------------------
md("""
## 5. Dyna-Q: learn a model, then plan with it

Dyna-Q is Q-learning plus a learned model. After every real step it stores what happened
(counts of (s', r) per (s, a), so slippery outcomes keep their probabilities) and then performs
`planning_steps` extra Q-learning updates on transitions *replayed from that model*. Each real
experience is used many times, so far fewer real episodes are needed: it sits between DP
(full model, no experience) and Q-learning (no model, only experience).
""")
show("algorithms/planning", "CountModel", "dyna_q")
code("""
plan = {f"dyna-q (n={k})": train(default_config("frozenlake-8x8", "dyna-q", planning_steps=k)) for k in [0, 5, 20]}
viz.plot_curves({m: r.history for m, r in plan.items()}, "frozenlake-8x8"); plt.show()
""")

# ---------------------------------------------------------------------------------------------
md("""
## 6. Neural networks on one-hot states: DQN and REINFORCE

The deep-RL machinery on a problem where the answer is known. A state becomes a one-hot vector,
so the first linear layer is literally a table; the network adds nothing a table cannot do,
which makes it a clean test of the training loops.

* **DQN**: Q-learning with an MLP, a replay buffer (break correlations between consecutive
  samples) and a target network (a frozen copy for the target, so the regression target does
  not move at every step).
* **REINFORCE**: learn the policy directly, `grad = E[grad log pi(a|s) (G_t - V(s))]`, with a
  learned value baseline to cut the variance and an entropy bonus to keep exploring.
""")
show("algorithms/neural", "dqn", "reinforce")
code("""
neural = {m: train(default_config("frozenlake-4x4-slippery", m)) for m in ["dqn", "reinforce"]}
viz.plot_curves({m: r.history for m, r in neural.items()}, "frozenlake-4x4-slippery",
                optimum=vi.summary["optimum"]); plt.show()
""")

# ---------------------------------------------------------------------------------------------
md("""
## 7. The full comparison

Every method with the same evaluation on every environment. `EPISODES_SCALE` multiplies each
method's default episode budget (1 = the short CPU runs of the README; 2 or more gives the slow
learners, Monte Carlo and REINFORCE on the 8x8 slippery lake, the time they need). Taxi and
CliffWalking run DP (the reference), the TD family and Dyna-Q.
""")
code("""
EPISODES_SCALE = 2          # 1 = the budgets of the README tables
SUITES = {
    "frozenlake-4x4": "all",
    "frozenlake-4x4-slippery": "all",
    "frozenlake-8x8": "all",
    "frozenlake-8x8-slippery": "all",
    "taxi": "td",
    "cliffwalking": "td",
}
from frozenlake_rl.compare import parse_methods
from IPython.display import Markdown, display, Image

for env_name, group in SUITES.items():
    compare(env_name, parse_methods(group), f"runs/compare/{env_name}", verbose=False, episodes_scale=EPISODES_SCALE)
    display(Markdown(open(f"runs/compare/{env_name}/results.md").read()))
""")
code("""
for env_name in SUITES:
    for fig_name in ["curves.png", "policies.png"]:
        path = f"runs/compare/{env_name}/{fig_name}"
        if os.path.exists(path):
            display(Image(path))
""")
md("""
## What to take away

* **DP** gives the exact optimum in milliseconds, but only because it reads P.
* **Monte Carlo** works on short episodic problems; its variance makes it the slowest learner on
  long or sparse ones (8x8 slippery).
* **TD** learns from every step; **Q-learning** finds the optimal greedy policy, **SARSA**
  the best policy *given that it keeps exploring* (the cliff).
* **n-step / lambda** methods spread sparse rewards faster along the path: fewer episodes.
* **Dyna-Q** reuses experience through a learned model: the fewest real episodes of all.
* **DQN / REINFORCE** reach the same policies with much more compute: the network brings
  nothing on a table, but the training loops are exactly those of the Atari-scale versions.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT} ({len(cells)} cells)")
