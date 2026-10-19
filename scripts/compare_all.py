"""Run the full comparison behind the README results and collect the figures in assets/.

    python scripts/compare_all.py                              # every suite (about 35 min on 1 CPU)
    python scripts/compare_all.py --suites frozenlake-4x4 taxi  # some of them
    python scripts/compare_all.py --plot-only                   # rebuild assets/ from runs/compare

Each suite is `frozenlake-compare --env <env> --methods <group>`; afterwards the script copies
curves / policies / online-return figures to assets/<env>_*.png, concatenates the tables into
assets/results.md and draws assets/scoreboard.png (final score / best possible, every method x env).
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from frozenlake_rl.common import set_threads
from frozenlake_rl.compare import compare, parse_methods
from frozenlake_rl.config import METHODS
from frozenlake_rl.envs import get_spec
from frozenlake_rl.viz import plot_curves_by_family, save

SUITES = {
    "frozenlake-4x4": "all",
    "frozenlake-4x4-slippery": "all",
    "frozenlake-8x8": "all",
    "frozenlake-8x8-slippery": "all",
    "taxi": "td",
    "cliffwalking": "td",
}


def scoreboard(runs: Path, assets: Path) -> None:
    envs = [e for e in SUITES if (runs / e / "results.json").exists()]
    grid = np.full((len(METHODS), len(envs)), np.nan)
    for j, env in enumerate(envs):
        spec = get_spec(env)
        for row in json.loads((runs / env / "results.json").read_text()):
            score = row["exact"][spec.target_metric]
            best = row["optimum"]
            if spec.target_metric == "success":
                grid[METHODS.index(row["method"]), j] = score / best
            else:  # returns: 1 = optimal, minus the relative gap to it
                grid[METHODS.index(row["method"]), j] = 1 - (best - score) / abs(best)
    fig, ax = plt.subplots(figsize=(1.6 * len(envs) + 3, 6.5))
    im = ax.imshow(np.clip(grid, 0, 1), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if np.isnan(grid[i, j]):
                ax.text(j, i, "-", ha="center", va="center", color="gray")
                continue
            env = envs[j]
            spec = get_spec(env)
            row = next(r for r in json.loads((runs / env / "results.json").read_text()) if r["method"] == METHODS[i])
            value = row["exact"][spec.target_metric]
            ax.text(j, i, f"{value:.2f}" if spec.target_metric == "success" else f"{value:.1f}",
                    ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(envs)))
    ax.set_xticklabels([e.replace("frozenlake-", "FL ") for e in envs], rotation=20, fontsize=8)
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels(METHODS, fontsize=8)
    ax.set_title("final greedy policy: success probability (FrozenLake) / expected return (Taxi, Cliff)\n"
                 "colour = fraction of the best possible score", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(assets / "scoreboard.png", dpi=110)
    plt.close(fig)


def redraw_curves(d: Path, env: str) -> None:
    """Rebuild curves.png from the saved histories (so figure changes need no re-training)."""
    rows = json.loads((d / "results.json").read_text())
    histories = {r["method"]: json.loads((d / r["method"] / "history.json").read_text())
                 for r in rows if r["family"] != "DP"}
    if histories:
        save(plot_curves_by_family(histories, env, optimum=rows[0]["optimum"]), d / "curves.png")


def collect(runs: Path, assets: Path, timings: dict) -> None:
    assets.mkdir(parents=True, exist_ok=True)
    parts = []
    for env in SUITES:
        d = runs / env
        if not (d / "results.md").exists():
            continue
        redraw_curves(d, env)
        for name in ("curves", "policies", "train_returns"):
            if (d / f"{name}.png").exists():
                shutil.copy(d / f"{name}.png", assets / f"{env}_{name}.png")
        text = (d / "results.md").read_text()
        if env in timings:
            text += f"\nsuite wall time: {timings[env] / 60:.1f} min\n"
        parts.append(text)
    (assets / "results.md").write_text("\n".join(parts))
    scoreboard(runs, assets)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suites", nargs="*", default=list(SUITES))
    parser.add_argument("--runs", default="runs/compare")
    parser.add_argument("--assets", default="assets")
    parser.add_argument("--eval-episodes", type=int, default=1000)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    set_threads(1)
    runs, assets = Path(args.runs), Path(args.assets)
    timing_file = runs / "timings.json"
    timings = json.loads(timing_file.read_text()) if timing_file.exists() else {}
    if not args.plot_only:
        for env in args.suites:
            t0 = time.perf_counter()
            compare(env, parse_methods(SUITES[env]), runs / env, eval_episodes=args.eval_episodes)
            timings[env] = time.perf_counter() - t0
            timing_file.parent.mkdir(parents=True, exist_ok=True)
            timing_file.write_text(json.dumps(timings, indent=2))
    collect(runs, assets, timings)
    print(f"wrote {assets}/results.md and figures")


if __name__ == "__main__":
    main()
