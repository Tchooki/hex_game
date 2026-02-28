from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from hex_game.ai.mcts import generate_data
from hex_game.ai.model import HexNet


class MockModel(HexNet):
    def __init__(self, n: int = 11):
        super().__init__(n=n, n_res_block=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        # Return random probabilities
        #           (not normalized yet as perform_model handles softmax)
        probas = torch.randn(batch_size, self.n * self.n)
        # Return random values between -1 and 1
        values = torch.tanh(torch.randn(batch_size, 1))
        return probas, values


@pytest.fixture
def n():
    return 5  # Smaller board for faster testing


@pytest.fixture
def model(n):
    return MockModel(n=n)


def test_generate_data_shapes(model, n):
    n_games = 2
    boards, policies, values = generate_data(
        model,
        n_games=n_games,
        n_iter=5,  # Small number of iterations for speed
        temperature_threshold=5,
    )

    assert isinstance(boards, np.ndarray)
    assert isinstance(policies, np.ndarray)
    assert isinstance(values, np.ndarray)

    # Number of positions generated should be at least n_games
    #   (if game ends immediately)
    assert len(boards) >= n_games
    assert len(boards) == len(policies)
    assert len(boards) == len(values)

    # Check shapes
    assert boards.shape[1:] == (n, n)
    assert policies.shape[1] == n * n
    assert len(values.shape) == 1


def test_generate_data_save(model, tmp_path):
    n_games = 1
    generate_data(model, n_games=n_games, n_iter=2, save_path=str(tmp_path))

    files = list(Path(tmp_path).iterdir())
    assert len(files) == 1
    assert files[0].name.startswith("selfplay_data_")
    assert files[0].name.endswith(".npz")

    # Load and verify data
    data = np.load(files[0])
    assert "boards" in data
    assert "policies" in data
    assert "values" in data
    assert data["n_games"] == n_games


def test_policy_validity(model):
    n_games = 1
    _, policies, _ = generate_data(model, n_games=n_games, n_iter=10)

    for policy in policies:
        # Policy should sum to 1 (or very close)
        assert np.isclose(np.sum(policy), 1.0, atol=1e-5)
        # All probabilities should be >= 0
        assert np.all(policy >= 0)
