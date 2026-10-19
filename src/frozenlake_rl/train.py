"""Train one method on one environment: `train(cfg)` / `frozenlake-train`.

    frozenlake-train --env frozenlake-4x4-slippery --method q-learning
    frozenlake-train --env taxi --method sarsa --episodes 2000 --alpha 0.5
    frozenlake-train --env frozenlake-8x8-slippery --method value-iteration

Every run does the same thing whatever the method:
1. build the env and a seeded numpy Generator (and torch seed for the neural methods);
2. call the algorithm, which reports every episode to a `Tracker` (exact greedy score);
3. evaluate the final greedy policy on `eval_episodes` real episodes;
4. write `runs/<env>/<method>/`: agent.npz (Q, V, policy), summary.json, history.json, run.png.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field, fields
from functools import cache
from pathlib import Path

import numpy as np

from frozenlake_rl.algorithms import ALGORITHMS
from frozenlake_rl.common import set_threads
from frozenlake_rl.config import DP_METHODS, METHODS, Config, default_config
from frozenlake_rl.envs import ENVS, get_spec, make_env
from frozenlake_rl.evaluate import ExactEvaluator, rollout_evaluate
from frozenlake_rl.metrics import Tracker


@dataclass
class RunResult:
    config: Config
    Q: np.ndarray
    V: np.ndarray
    policy: np.ndarray
    history: dict
    train_returns: list
    summary: dict
    extras: dict = field(default_factory=dict)


@cache
def optimal_score(env_name: str) -> dict:
    """Best achievable value of the evaluation metric within the time limit (finite-horizon DP)."""
    return ExactEvaluator(env_name).best()


def train(cfg: Config, out_dir: str | Path | None = None, verbose: bool = True) -> RunResult:
    set_threads(cfg.threads)
    spec = get_spec(cfg.env)
    env = make_env(cfg.env)
    rng = np.random.default_rng(cfg.seed)
    env.action_space.seed(cfg.seed)
    is_dp = cfg.method in DP_METHODS
    tracker = Tracker(cfg.env, total=cfg.max_sweeps if is_dp else cfg.episodes,
                      eval_points=cfg.max_sweeps if is_dp else cfg.eval_points)
    t0 = time.perf_counter()
    out = ALGORITHMS[cfg.method](env, cfg, rng, tracker)
    train_time = time.perf_counter() - t0
    env.close()

    Q = np.asarray(out["Q"], dtype=np.float64)
    policy = np.asarray(out.get("policy", Q.argmax(1)), dtype=np.int64)
    V = np.asarray(out["V"], dtype=np.float64)
    exact = ExactEvaluator(cfg.env)(policy)
    rollout = rollout_evaluate(cfg.env, policy, cfg.eval_episodes, seed=10_000 + cfg.seed)
    optimum = optimal_score(cfg.env)
    summary = {
        "config": cfg.to_dict(),
        "train_time_s": round(train_time, 3),
        "episodes": 0 if is_dp else cfg.episodes,
        "env_steps": int(tracker.env_steps),
        "sweeps": int(out.get("sweeps", 0)),
        "solved_at": tracker.solved_at,          # episodes (sweeps for DP) to reach the target
        "solved_steps": tracker.solved_steps,
        "solved_time_s": None if tracker.solved_time is None else round(tracker.solved_time, 3),
        "target_metric": spec.target_metric,
        "target": spec.target,
        "exact": exact,
        "rollout": rollout,
        "optimum": optimum[spec.target_metric],    # ceiling of the metric within the time limit
    }
    history = dict(tracker.history)
    result = RunResult(cfg, Q, V, policy, history, tracker.train_returns, summary,
                       {k: v for k, v in out.items() if k not in ("Q", "V", "policy")})
    if verbose:
        score = exact[spec.target_metric]
        solved = summary["solved_at"]
        unit = "sweeps" if is_dp else "episodes"
        print(f"{cfg.method:>18} | {cfg.env:<24} | exact {spec.target_metric} {score:7.3f} "
              f"(best possible {summary['optimum']:.3f}) | rollouts: success {rollout['success_rate']:.3f} "
              f"return {rollout['mean_return']:7.2f} | solved at {solved if solved is not None else '-'} {unit} "
              f"| {train_time:6.1f} s")
    if out_dir is not None:
        save_run(result, out_dir)
    return result


def save_run(result: RunResult, out_dir: str | Path, figure: bool = True) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {"Q": result.Q, "V": result.V, "policy": result.policy}
    if "probs" in result.extras:
        arrays["probs"] = result.extras["probs"]
    np.savez(out_dir / "agent.npz", **arrays)
    (out_dir / "summary.json").write_text(json.dumps(result.summary, indent=2))
    (out_dir / "history.json").write_text(json.dumps({**result.history, "train_returns": result.train_returns}))
    if "model" in result.extras:
        import torch

        torch.save(result.extras["model"], out_dir / "model.pt")
    if figure:
        from frozenlake_rl.viz import run_figure

        run_figure(arrays, result.summary, result.history, out_dir / "run.png")
    return out_dir


def add_config_args(parser: argparse.ArgumentParser) -> None:
    """Every Config field as an optional --flag (None = keep the tuned default)."""
    for f in fields(Config):
        if f.name in ("env", "method"):
            continue
        flag = "--" + f.name.replace("_", "-")
        if f.type in ("bool", bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=None)
        else:
            kind = {"int": int, "float": float, "float | None": float}.get(str(f.type), str)
            parser.add_argument(flag, type=kind, default=None)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train one classical RL method on a discrete environment.")
    parser.add_argument("--env", default="frozenlake-4x4-slippery", help=f"one of {sorted(ENVS)}")
    parser.add_argument("--method", default="q-learning", choices=METHODS)
    parser.add_argument("--out", default=None, help="output dir (default runs/<env>/<method>)")
    add_config_args(parser)
    args = vars(parser.parse_args(argv))
    env, method, out = args.pop("env"), args.pop("method"), args.pop("out")
    cfg = default_config(env, method, **args)
    out = out or Path("runs") / cfg.env / cfg.method
    train(cfg, out_dir=out)
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
