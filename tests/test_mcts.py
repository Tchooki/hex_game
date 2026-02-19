import numpy as np

from solve.mcts import get_random_transformation


def test_reflexion():
    rng = np.random.default_rng()
    data = rng.integers(-10, 10, (10, 10))
    for _ in range(20):
        trans = get_random_transformation()
        assert (data == trans(trans(data.copy()))).any()


def test_reflexion_with_flatten():
    rng = np.random.default_rng()
    data = rng.integers(-10, 10, (10, 10)).reshape(-1)
    for _ in range(20):
        trans = get_random_transformation()
        data_trans = data.copy().reshape(10, -1)
        data_trans = trans(trans(data_trans)).reshape(-1)
        assert (data == data_trans).any()
