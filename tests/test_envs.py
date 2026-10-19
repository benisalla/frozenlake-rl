import numpy as np
import pytest

from frozenlake_rl.envs import (
    ENVS,
    get_spec,
    is_success,
    make_env,
    n_states_actions,
    one_hot,
    terminal_states,
)


@pytest.mark.parametrize("name", sorted(ENVS))
def test_every_env_builds_with_its_time_limit(name):
    env = make_env(name)
    state, _ = env.reset(seed=0)
    assert 0 <= state < env.observation_space.n
    assert env.spec.max_episode_steps == get_spec(name).max_steps
    assert len(env.unwrapped.P) == env.observation_space.n


def test_sizes_and_aliases():
    assert n_states_actions(make_env("fl4")) == (16, 4)
    assert n_states_actions(make_env("frozenlake-8x8-slippery")) == (64, 4)
    assert n_states_actions(make_env("taxi")) == (500, 6)
    assert n_states_actions(make_env("cliff")) == (48, 4)
    with pytest.raises(ValueError):
        get_spec("mountaincar")


def test_terminal_states_are_holes_and_goal():
    assert terminal_states(make_env("fl4")).nonzero()[0].tolist() == [5, 7, 11, 12, 15]
    assert not terminal_states(make_env("taxi")).any()


def test_success_definition():
    fl, taxi = get_spec("fl4"), get_spec("taxi")
    assert is_success(fl, True, 1.0) and not is_success(fl, True, 0.0)
    assert is_success(taxi, True, -3.0) and not is_success(taxi, False, -200.0)


def test_one_hot():
    x = one_hot(3, 5)
    assert x.tolist() == [0, 0, 0, 1, 0] and x.dtype == np.float32
