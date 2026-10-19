import json

import nbformat
import pytest

from frozenlake_rl import compare, evaluate, play, train


def test_train_cli_writes_a_run(tmp_path, capsys):
    out = tmp_path / "run"
    train.main(["--env", "fl4", "--method", "q-learning", "--episodes", "200", "--eval-episodes", "20",
                "--out", str(out)])
    for name in ("agent.npz", "summary.json", "history.json", "run.png"):
        assert (out / name).exists()
    summary = json.loads((out / "summary.json").read_text())
    assert summary["config"]["episodes"] == 200 and summary["rollout"]["episodes"] == 20
    evaluate.main([str(out), "--episodes", "20"])
    assert "rollouts" in capsys.readouterr().out


def test_play_runs_the_saved_policy(tmp_path):
    out = tmp_path / "vi"
    train.main(["--env", "fl4", "--method", "value-iteration", "--eval-episodes", "5", "--out", str(out)])
    arrays, summary = evaluate.load_run(out)
    results = play.play("fl4", arrays["policy"], episodes=2, render=False)
    assert all(r["success"] for r in results) and results[0]["steps"] == 6


def test_compare_writes_table_and_figures(tmp_path):
    rows = compare.compare("fl4", ["value-iteration", "q-learning"], tmp_path, verbose=False,
                           episodes=200, eval_episodes=20)
    assert [r["method"] for r in rows] == ["value-iteration", "q-learning"]
    for name in ("results.md", "results.json", "curves.png", "policies.png", "train_returns.png"):
        assert (tmp_path / name).exists()
    assert "`q-learning`" in (tmp_path / "results.md").read_text()


def test_method_groups():
    assert compare.parse_methods("td")[:2] == ["value-iteration", "policy-iteration"]
    assert compare.parse_methods("sarsa,neural") == ["sarsa", "dqn", "reinforce"]
    with pytest.raises(ValueError):
        compare.parse_methods("ppo")


def test_notebook_builds(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "nb.ipynb"
    subprocess.run([sys.executable, str(root / "notebooks" / "build_notebook.py"), str(root), str(out)], check=True)
    nb = nbformat.read(out, as_version=4)
    nbformat.validate(nb)
    sources = "\n".join(c.source for c in nb.cells)
    assert "frozenlake_rl" in sources and "Bellman" in sources
