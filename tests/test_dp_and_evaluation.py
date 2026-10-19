import numpy as np

from frozenlake_rl.algorithms.dp import model_arrays, policy_evaluation
from frozenlake_rl.config import default_config
from frozenlake_rl.envs import make_env
from frozenlake_rl.evaluate import ExactEvaluator, policy_matrix, rollout_evaluate
from frozenlake_rl.train import optimal_score, train


def test_value_and_policy_iteration_agree_on_slippery_lake():
    vi = train(default_config("fl4s", "value-iteration"), verbose=False)
    pi = train(default_config("fl4s", "policy-iteration"), verbose=False)
    assert np.allclose(vi.V, pi.V, atol=1e-6)
    assert abs(vi.summary["exact"]["success"] - 0.74) < 0.01        # known optimum of the 4x4 slippery lake
    assert vi.summary["solved_at"] is not None


def test_dp_value_satisfies_bellman_optimality():
    res = train(default_config("cliff", "value-iteration"), verbose=False)
    T, R = model_arrays(make_env("cliff"))
    gamma = res.config.gamma
    backup = (R + gamma * T @ res.V).max(1)
    assert np.abs(backup - res.V).max() < 1e-6
    assert res.summary["exact"]["return"] == -13.0                    # shortest path along the cliff


def test_policy_evaluation_of_always_left_is_zero():
    T, R = model_arrays(make_env("fl4"))
    V, _ = policy_evaluation(T, R, np.zeros(16, dtype=int), 0.99, 1e-10, 1000)
    assert np.allclose(V, 0)


def test_exact_and_rollout_evaluation_agree():
    res = train(default_config("fl4s", "value-iteration", eval_episodes=10), verbose=False)
    exact = ExactEvaluator("fl4s")(res.policy)
    rollout = rollout_evaluate("fl4s", res.policy, episodes=2000, seed=1)
    assert abs(exact["success"] - rollout["success_rate"]) < 3 * np.sqrt(0.25 / 2000) + 0.01


def test_exact_evaluator_handles_stochastic_policies():
    uniform = np.full((16, 4), 0.25)
    assert policy_matrix(np.array([1, 2]), 4).tolist() == [[0, 1, 0, 0], [0, 0, 1, 0]]
    random_score = ExactEvaluator("fl4s")(uniform)["success"]
    assert 0 < random_score < 0.05
    assert optimal_score("taxi")["return"] > 7
