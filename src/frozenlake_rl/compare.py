"""Run several methods on one environment and compare them: `frozenlake-compare`.

    frozenlake-compare --env frozenlake-4x4-slippery                 # every method
    frozenlake-compare --env taxi --methods td                        # DP + the TD family + Dyna-Q
    frozenlake-compare --env cliffwalking --methods sarsa,q-learning

Writes to `runs/compare/<env>/`: one sub-directory per method (as frozenlake-train), plus
* results.md / results.json: the comparison table,
* curves.png: exact greedy score vs episodes for every model-free method,
* policies.png: V(s) heatmap + greedy arrows of every method side by side,
* train_returns.png: online returns during training (exploration included).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from frozenlake_rl.config import (
    DP_METHODS,
    MC_METHODS,
    METHODS,
    NEURAL_METHODS,
    PLANNING_METHODS,
    TD_METHODS,
    default_config,
)
from frozenlake_rl.envs import ENVS, env_family, get_spec
from frozenlake_rl.train import add_config_args, save_run, train
from frozenlake_rl.viz import plot_curves_by_family, plot_train_returns, plot_value_policy, save

GROUPS = {
    "all": METHODS,
    "td": DP_METHODS + TD_METHODS + PLANNING_METHODS,
    "tabular": DP_METHODS + MC_METHODS + TD_METHODS + PLANNING_METHODS,
    "dp": DP_METHODS,
    "mc": MC_METHODS,
    "neural": NEURAL_METHODS,
}

FAMILY = {**{m: "DP" for m in DP_METHODS}, **{m: "MC" for m in MC_METHODS}, **{m: "TD" for m in TD_METHODS},
          **{m: "planning" for m in PLANNING_METHODS}, **{m: "neural" for m in NEURAL_METHODS}}


def parse_methods(text: str) -> list[str]:
    methods: list[str] = []
    for part in text.split(","):
        part = part.strip()
        for m in GROUPS.get(part, (part,)):
            if m not in METHODS:
                raise ValueError(f"unknown method {m!r}")
            if m not in methods:
                methods.append(m)
    return methods


def results_table(rows: list[dict], env_name: str) -> str:
    spec = get_spec(env_name)
    metric = spec.target_metric
    target = f"{metric} >= {spec.target:g}"
    lines = [
        f"| method | family | episodes | env steps | episodes to {target} | exact {metric} (greedy) | "
        f"rollout success | rollout mean return | train time |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        dp = r["family"] == "DP"
        solved = r["solved_at"]
        if dp:
            to_target = f"0 ({solved} sweeps)" if solved is not None else "-"
            episodes, steps = "model", "-"
        else:
            to_target = f"{solved:,}" if solved is not None else "not reached"
            episodes, steps = f"{r['episodes']:,}", f"{r['env_steps']:,}"
        score = r["exact"][metric]
        score = f"{score:.3f}" if metric == "success" else f"{score:.2f}"
        lines.append(
            f"| `{r['method']}` | {r['family']} | {episodes} | {steps} | {to_target} | {score} | "
            f"{r['rollout']['success_rate']:.3f} | {r['rollout']['mean_return']:.2f} | {r['train_time_s']:.1f} s |"
        )
    return "\n".join(lines)


def compare(env_name: str, methods: list[str], out_dir: str | Path, verbose: bool = True,
            episodes_scale: float = 1.0, **overrides) -> list[dict]:
    """Train every method with its tuned defaults (times `episodes_scale` episodes) and compare."""
    env_name = get_spec(env_name).name
    out_dir = Path(out_dir)
    rows, histories, train_returns, results = [], {}, {}, {}
    for method in methods:
        cfg = default_config(env_name, method, **overrides)
        if episodes_scale != 1.0 and overrides.get("episodes") is None:
            cfg = default_config(env_name, method, **{**overrides, "episodes": int(cfg.episodes * episodes_scale)})
        result = train(cfg, verbose=verbose)
        save_run(result, out_dir / method)
        results[method] = result
        s = result.summary
        rows.append({"method": method, "family": FAMILY[method], **{k: s[k] for k in (
            "episodes", "env_steps", "solved_at", "solved_steps", "train_time_s", "exact", "rollout", "optimum")}})
        if method not in DP_METHODS:
            histories[method] = result.history
            train_returns[method] = result.train_returns

    table = results_table(rows, env_name)
    optimum = rows[0]["optimum"] if rows else None
    (out_dir / "results.md").write_text(
        f"### {env_name}\n\n{table}\n\nbest possible {get_spec(env_name).target_metric} within the time limit "
        f"(finite-horizon DP): {optimum:.3f}\n")
    (out_dir / "results.json").write_text(json.dumps(rows, indent=2))

    if histories:
        save(plot_curves_by_family(histories, env_name, optimum=optimum), out_dir / "curves.png")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ylim = (-150, 0) if env_family(env_name) == "cliffwalking" else None
        plot_train_returns(train_returns, env_name, window=20 if len(next(iter(train_returns.values()))) < 2000
                           else 200, ax=ax, ylim=ylim)
        save(fig, out_dir / "train_returns.png")

    n = len(results)
    cols = min(4, n)
    rows_n = (n + cols - 1) // cols
    taxi = env_family(env_name) == "taxi"
    w, h = (5.5, 2.6) if taxi else ((4.2, 1.8) if env_family(env_name) == "cliffwalking" else (3.2, 2.9))
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * w, rows_n * h), squeeze=False)
    metric = get_spec(env_name).target_metric
    for ax, (method, result) in zip(axes.flat, results.items(), strict=False):
        score = result.summary["exact"][metric]
        plot_value_policy(result.V, result.policy, env_name, ax, f"{method}  ({metric} {score:.2f})")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    save(fig, out_dir / "policies.png")
    if verbose:
        print()
        print(table)
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare RL methods on one environment.")
    parser.add_argument("--env", default="frozenlake-4x4-slippery", help=f"one of {sorted(ENVS)}")
    parser.add_argument("--methods", default="all",
                        help=f"comma list of methods or groups {sorted(GROUPS)} (default: all)")
    parser.add_argument("--out", default=None, help="output dir (default runs/compare/<env>)")
    parser.add_argument("--episodes-scale", type=float, default=1.0,
                        help="multiply every method's default episode budget (e.g. 2 for longer runs)")
    add_config_args(parser)
    args = vars(parser.parse_args(argv))
    env, methods, out = args.pop("env"), parse_methods(args.pop("methods")), args.pop("out")
    scale = args.pop("episodes_scale")
    env = get_spec(env).name
    compare(env, methods, out or Path("runs") / "compare" / env, episodes_scale=scale, **args)


if __name__ == "__main__":
    main()
