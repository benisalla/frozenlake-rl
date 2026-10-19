"""Render the README's before/after GIFs and the value-iteration animation into assets/gifs/.

    python scripts/make_gifs.py            # about 2 minutes on one CPU

What it writes:
* frozenlake-8x8-slippery.gif  random agent | trained agent (Q-learning), same seed, visited cells dotted
* taxi.gif                     random agent | trained agent (Q-learning), same seed
* cliffwalking.gif             SARSA (safe path) | Q-learning (edge path), both trained, same seed
* value_iteration.gif          V(s) heatmap + greedy arrows on the 8x8 slippery lake, sweep by sweep

Frames come from gymnasium's `render_mode="rgb_array"`; each panel gets a caption above it and a
status line under it (step, return, outcome). A panel whose episode ends first holds its last frame.
"""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from frozenlake_rl.algorithms.dp import model_arrays, q_from_v  # noqa: E402
from frozenlake_rl.common import set_threads  # noqa: E402
from frozenlake_rl.config import default_config  # noqa: E402
from frozenlake_rl.envs import get_spec, is_success, make_env  # noqa: E402
from frozenlake_rl.train import train  # noqa: E402
from frozenlake_rl.viz import plot_value_policy  # noqa: E402

OUT = Path("assets/gifs")
FONT = font_manager.findfont("DejaVu Sans")


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT, size)


def rollout_frames(env_name: str, policy, seed: int, rng: np.random.Generator | None = None) -> list[dict]:
    """One episode: a list of {frame, step, ret, done_text}. policy=None means uniformly random actions."""
    spec = get_spec(env_name)
    env = make_env(env_name, render_mode="rgb_array")
    state, _ = env.reset(seed=seed)
    out, total, steps = [], 0.0, 0
    terminated = truncated = False
    visited = [int(state)]
    out.append({"frame": env.render(), "step": 0, "ret": 0.0, "end": "", "trail": list(visited)})
    while not (terminated or truncated):
        action = int(rng.integers(env.action_space.n)) if policy is None else int(policy[state])
        state, reward, terminated, truncated, _ = env.step(action)
        total += reward
        steps += 1
        end = ""
        if terminated or truncated:
            if is_success(spec, terminated, total):
                end = "SUCCESS"
            else:
                end = "time limit" if truncated else "fell in a hole"
        visited.append(int(state))
        out.append({"frame": env.render(), "step": steps, "ret": total, "end": end, "trail": list(visited)})
    env.close()
    return out


def draw_trail(img: Image.Image, trail: list[int], grid: tuple[int, int]) -> Image.Image:
    """Orange dots on the cells visited so far, so the path stays visible after the episode."""
    rows, cols = grid
    cw, ch = img.width / cols, img.height / rows
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    r = min(cw, ch) * 0.16
    for s in set(trail):
        y, x = divmod(s, cols)
        cx, cy = (x + 0.5) * cw, (y + 0.5) * ch
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(255, 140, 0, 200))
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def panel(item: dict, caption: str, width: int, grid: tuple[int, int] | None = None) -> Image.Image:
    img = Image.fromarray(item["frame"])
    if grid is not None:
        img = draw_trail(img, item["trail"], grid)
    scale = width / img.width
    img = img.resize((width, int(img.height * scale)), Image.LANCZOS)
    top, bottom = 34, 28
    canvas = Image.new("RGB", (width, img.height + top + bottom), "white")
    canvas.paste(img, (0, top))
    d = ImageDraw.Draw(canvas)
    d.text((width // 2, top // 2), caption, fill="black", anchor="mm", font=font(17))
    status = f"step {item['step']}   return {item['ret']:g}"
    if item["end"]:
        status += f"   {item['end']}"
    color = (0, 120, 0) if item["end"] == "SUCCESS" else ((170, 0, 0) if item["end"] else (60, 60, 60))
    d.text((width // 2, img.height + top + bottom // 2), status, fill=color, anchor="mm", font=font(14))
    return canvas


def side_by_side(episodes: list[list[dict]], captions: list[str], path: Path, width: int,
                 hold: int = 8, duration: int = 160, grid: tuple[int, int] | None = None) -> Path:
    n = max(len(e) for e in episodes) + hold
    frames = []
    for t in range(n):
        panels = [panel(e[min(t, len(e) - 1)], c, width, grid) for e, c in zip(episodes, captions, strict=True)]
        h = max(p.height for p in panels)
        canvas = Image.new("RGB", (sum(p.width for p in panels) + 12 * (len(panels) - 1), h), "white")
        x = 0
        for p in panels:
            canvas.paste(p, (x, 0))
            x += p.width + 12
        frames.append(canvas.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
    return save_gif(frames, path, duration)


def save_gif(frames: list[Image.Image], path: Path, duration: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration, loop=0, optimize=True)
    print(f"wrote {path} ({len(frames)} frames, {path.stat().st_size / 1e6:.2f} MB)")
    return path


def first_success_seed(env_name: str, policy, seeds=range(100)) -> int:
    for seed in seeds:
        if rollout_frames(env_name, policy, seed)[-1]["end"] == "SUCCESS":
            return seed
    return 0


def before_after(env_name: str, method: str, width: int, duration: int = 160, **overrides) -> None:
    trained = train(default_config(env_name, method, eval_episodes=200, **overrides), verbose=True)
    seed = first_success_seed(env_name, trained.policy)
    rng = np.random.default_rng(seed)
    random_ep = rollout_frames(env_name, None, seed, rng)
    trained_ep = rollout_frames(env_name, trained.policy, seed)
    success = trained.summary["exact"]["success"]
    side_by_side([random_ep, trained_ep],
                 ["before: random actions", f"after: {method} (success {success:.0%})"],
                 OUT / f"{get_spec(env_name).name}.gif", width, duration=duration, grid=get_spec(env_name).grid)


def cliff() -> None:
    runs = {m: train(default_config("cliffwalking", m, eval_episodes=50)) for m in ["sarsa", "q-learning"]}
    episodes = [rollout_frames("cliffwalking", runs[m].policy, seed=0) for m in runs]
    side_by_side(episodes, ["SARSA: safe path (on-policy)", "Q-learning: edge path (off-policy)"],
                 OUT / "cliffwalking.gif", width=480, hold=10, duration=220, grid=get_spec("cliffwalking").grid)


def value_iteration_animation(env_name: str = "frozenlake-8x8-slippery", max_frames: int = 45) -> None:
    """V(s) and the greedy arrows after each value-iteration sweep."""
    T, R = model_arrays(make_env(env_name))
    gamma = get_spec(env_name).gamma
    V = np.zeros(T.shape[0])
    snapshots = [(0, V.copy())]
    for sweep in range(1, 2000):
        V_new = q_from_v(T, R, V, gamma).max(1)
        delta = np.abs(V_new - V).max()
        V = V_new
        snapshots.append((sweep, V.copy()))
        if delta < 1e-8:
            break
    # dense at the start (where the policy changes), sparse later
    keep = sorted(set(np.unique(np.geomspace(1, len(snapshots), max_frames).astype(int) - 1).tolist()) | {0})
    frames = []
    for i in keep:
        sweep, V = snapshots[i]
        policy = q_from_v(T, R, V, gamma).argmax(1)
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        plot_value_policy(V, policy, env_name, ax, f"value iteration, sweep {sweep}  (max V = {V.max():.2f})")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=90)
        plt.close(fig)
        frames.append(Image.open(buf).convert("RGB").quantize(colors=128, method=Image.Quantize.MEDIANCUT))
    frames += [frames[-1]] * 8
    save_gif(frames, OUT / "value_iteration.gif", duration=250)


def main() -> None:
    set_threads(1)
    before_after("frozenlake-8x8-slippery", "q-learning", width=320)
    before_after("taxi", "q-learning", width=340, duration=90)
    cliff()
    value_iteration_animation()


if __name__ == "__main__":
    main()
