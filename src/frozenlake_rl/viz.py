"""Figures, all saved as PNG: value heatmaps with policy arrows, Q tables, learning curves.

* `plot_value_policy`: V(s) as a heatmap on the grid, the greedy action as an arrow in every
  cell, holes / goal / cliff marked. Taxi has 500 states, so it is drawn as two 5x5 slices
  (passenger waiting at R, passenger in the taxi; destination G).
* `plot_q_table`: the raw Q(s, a) table, states x actions.
* `plot_curves`: exact success probability (or expected return) of the greedy policy vs
  episodes, one line per method, with the optimum from value iteration as a dashed line.
* `plot_train_returns`: moving average of the returns the agent got while learning (with its
  exploration), the classic SARSA vs Q-learning cliff figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from frozenlake_rl.envs import ACTION_ARROWS, env_family, get_spec, make_env
from frozenlake_rl.metrics import moving_average

TAXI_LOCS = "RGYB"


def _grid_labels(env_name: str) -> np.ndarray | None:
    """Letters drawn on special cells: S / H / G for FrozenLake, C (cliff) / S / G for CliffWalking."""
    spec = get_spec(env_name)
    rows, cols = spec.grid
    if env_family(env_name) == "frozenlake":
        desc = make_env(env_name).unwrapped.desc.astype(str)
        return np.where(np.isin(desc, ["S", "H", "G"]), desc, "")
    labels = np.full((rows, cols), "", dtype=object)
    labels[3, 1:-1] = "C"
    labels[3, 0], labels[3, -1] = "S", "G"
    return labels


def plot_value_policy(V, policy, env_name: str, ax=None, title: str = "", cmap: str = "viridis"):
    """Heatmap of V with the greedy action of each state drawn on top."""
    spec = get_spec(env_name)
    family = env_family(env_name)
    V, policy = np.asarray(V, dtype=float), np.asarray(policy)
    if policy.ndim == 2:
        policy = policy.argmax(1)
    if family == "taxi":
        return _plot_taxi(V, policy, ax, title, cmap)
    rows, cols = spec.grid
    labels = _grid_labels(env_name)
    if ax is None:
        _, ax = plt.subplots(figsize=(cols * 0.6 + 1.5, rows * 0.6 + 0.8))
    grid = V.reshape(rows, cols)
    mask = labels != ""
    shown = np.ma.masked_where(mask & (labels != "S"), grid)
    im = ax.imshow(shown, cmap=cmap)
    arrows = ACTION_ARROWS[family]
    vmin, vmax = float(np.min(grid)), float(np.max(grid))
    for r in range(rows):
        for c in range(cols):
            s = r * cols + c
            label = labels[r, c]
            if label in ("H", "C"):
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="#1f3b73" if label == "H" else "#7a1f1f"))
                ax.text(c, r, label, ha="center", va="center", color="white", fontsize=9, weight="bold")
            elif label == "G":
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="#f2c14e"))
                ax.text(c, r, "G", ha="center", va="center", color="black", fontsize=9, weight="bold")
            else:
                bright = (grid[r, c] - vmin) / (vmax - vmin + 1e-12) > 0.6
                ax.text(c, r, arrows[int(policy[s])], ha="center", va="center", fontsize=12,
                        color="black" if bright else "white")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def _taxi_state(row: int, col: int, passenger: int, dest: int) -> int:
    return ((row * 5 + col) * 5 + passenger) * 4 + dest


def _plot_taxi(V, policy, ax, title, cmap):
    """Two 5x5 slices of the 500 Taxi states, destination G."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.2))
    names = ["S", "N", "E", "W", "P", "D"]
    arrows = {"S": "↓", "N": "↑", "E": "→", "W": "←", "P": "P", "D": "D"}
    dest = 1
    panels = []
    for passenger in (0, 4):           # waiting at R, or already in the taxi
        g = np.array([[V[_taxi_state(r, c, passenger, dest)] for c in range(5)] for r in range(5)])
        a = np.array([[policy[_taxi_state(r, c, passenger, dest)] for c in range(5)] for r in range(5)])
        panels.append((g, a))
    full = np.concatenate([panels[0][0], np.full((5, 1), np.nan), panels[1][0]], axis=1)
    im = ax.imshow(np.ma.masked_invalid(full), cmap=cmap)
    locs = {(0, 0): "R", (0, 4): "G", (4, 0): "Y", (4, 3): "B"}
    for k, (_values, a) in enumerate(panels):
        off = k * 6
        for r in range(5):
            for c in range(5):
                txt = arrows[names[int(a[r, c])]]
                ax.text(c + off, r, txt, ha="center", va="center", color="white", fontsize=11)
                if (r, c) in locs:
                    ax.text(c + off - 0.35, r - 0.3, locs[(r, c)], color="#f2c14e", fontsize=7, weight="bold")
    ax.set_xticks([2, 8])
    ax.set_xticklabels(["passenger at R, dest G", "passenger in taxi, dest G"], fontsize=8)
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    return ax


def plot_q_table(Q, env_name: str, ax=None, title: str = "Q(s, a)"):
    Q = np.asarray(Q)
    family = env_family(env_name)
    if ax is None:
        _, ax = plt.subplots(figsize=(3.2, min(10, 1 + Q.shape[0] * 0.18)))
    im = ax.imshow(Q, aspect="auto", cmap="magma")
    ax.set_xticks(range(Q.shape[1]))
    ax.set_xticklabels(ACTION_ARROWS[family][: Q.shape[1]])
    ax.set_ylabel("state")
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_curves(histories: dict, env_name: str, ax=None, optimum: float | None = None, x_key: str = "x",
                title: str | None = None):
    """Exact score of the greedy policy during training, one line per method."""
    spec = get_spec(env_name)
    metric = spec.target_metric
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    cmap = plt.get_cmap("tab10" if len(histories) <= 10 else "tab20")
    for i, (name, hist) in enumerate(histories.items()):
        ax.plot(hist[x_key], hist[metric], label=name, color=cmap(i % cmap.N), lw=1.6)
    if optimum is not None:
        ax.axhline(optimum, ls="--", color="black", lw=1, label="best possible (finite-horizon DP)")
    ax.axhline(spec.target, ls=":", color="gray", lw=1, label=f"target {spec.target:g}")
    ax.set_xlabel("episodes" if x_key == "x" else "environment steps")
    ax.set_ylabel("success probability (exact)" if metric == "success" else "expected return (exact)")
    ax.set_title(title or f"{spec.name}: greedy policy during training", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    return ax


CURVE_GROUPS = {
    "Monte Carlo vs one-step TD": ["mc-first-visit", "mc-every-visit", "sarsa", "q-learning"],
    "one-step TD variants": ["sarsa", "expected-sarsa", "q-learning", "double-q-learning"],
    "multi-step and planning": ["q-learning", "n-step-sarsa", "sarsa-lambda", "q-lambda", "dyna-q"],
    "neural (one-hot states) vs tabular": ["q-learning", "dqn", "reinforce"],
}


def plot_curves_by_family(histories: dict, env_name: str, optimum: float | None = None):
    """The learning curves split in 2x2 panels of 3-5 methods each, so each comparison stays readable."""
    groups = {title: [m for m in methods if m in histories] for title, methods in CURVE_GROUPS.items()}
    groups = {t: ms for t, ms in groups.items() if len(ms) > 1} or {"all methods": list(histories)}
    n = len(groups)
    cols = 2 if n > 1 else 1
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6.5 * cols, 3.8 * rows), squeeze=False)
    for ax, (title, methods) in zip(axes.flat, groups.items(), strict=False):
        plot_curves({m: histories[m] for m in methods}, env_name, ax, optimum=optimum,
                    title=f"{get_spec(env_name).name}: {title}")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    return fig


def plot_train_returns(train_returns: dict, env_name: str, window: int = 20, ax=None, ylim=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    for name, returns in train_returns.items():
        ax.plot(moving_average(returns, window), label=name, lw=1.4)
    ax.set_xlabel("episode")
    ax.set_ylabel(f"return during training (moving average {window})")
    ax.set_title(f"{get_spec(env_name).name}: online returns with exploration", fontsize=10)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return ax


def save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def run_figure(arrays: dict, summary: dict, history: dict, path: str | Path) -> Path:
    """One figure per run: values + policy, Q table, learning curve."""
    env_name, method = summary["config"]["env"], summary["config"]["method"]
    taxi = env_family(env_name) == "taxi"
    ratios = [1.6 if taxi else 1, 0.6, 1.4]
    fig, axes = plt.subplots(1, 3, figsize=(15 if taxi else 13, 4), gridspec_kw={"width_ratios": ratios})
    plot_value_policy(arrays["V"], arrays["policy"], env_name, axes[0], f"{method}: V(s) and greedy policy")
    plot_q_table(arrays["Q"], env_name, axes[1])
    plot_curves({method: history}, env_name, axes[2], optimum=summary.get("optimum"))
    return save(fig, path)
