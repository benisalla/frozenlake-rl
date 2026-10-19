"""Watch a trained policy play: `frozenlake-play`.

    frozenlake-play runs/frozenlake-4x4-slippery/q-learning              # text render in the terminal
    frozenlake-play runs/taxi/q-learning --episodes 3 --delay 0.2
    frozenlake-play runs/frozenlake-8x8/sarsa --gif examples/output/sarsa.gif   # rgb frames -> GIF

The policy is the saved greedy action table, so every method plays the same way.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from frozenlake_rl.envs import get_spec, is_success, make_env
from frozenlake_rl.evaluate import load_run


def play(env_name: str, policy: np.ndarray, episodes: int = 1, delay: float = 0.0, seed: int = 0,
         render: bool = True, gif: str | None = None) -> list[dict]:
    spec = get_spec(env_name)
    mode = "rgb_array" if gif else ("ansi" if render else None)
    env = make_env(env_name, render_mode=mode)
    frames, results = [], []
    for ep in range(episodes):
        state, _ = env.reset(seed=seed + ep)
        total, steps, terminated, truncated = 0.0, 0, False, False
        while True:
            if gif:
                frames.append(env.render())
            elif render:
                print(env.render())
                time.sleep(delay)
            if terminated or truncated:
                break
            state, reward, terminated, truncated, _ = env.step(int(policy[state]))
            total += reward
            steps += 1
        ok = is_success(spec, terminated, total)
        results.append({"return": total, "steps": steps, "success": ok})
        if render or gif:
            print(f"episode {ep}: return {total:.1f} in {steps} steps, {'success' if ok else 'failure'}")
    env.close()
    if gif and frames:
        from PIL import Image

        images = [Image.fromarray(f) for f in frames]
        images[0].save(gif, save_all=True, append_images=images[1:], duration=max(int(delay * 1000), 150), loop=0)
        print(f"wrote {gif} ({len(frames)} frames)")
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Play a trained greedy policy.")
    parser.add_argument("run_dir")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--delay", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gif", default=None, help="write an animated GIF instead of printing")
    args = parser.parse_args(argv)
    arrays, summary = load_run(args.run_dir)
    play(summary["config"]["env"], arrays["policy"], args.episodes, args.delay, args.seed, gif=args.gif)


if __name__ == "__main__":
    main()
