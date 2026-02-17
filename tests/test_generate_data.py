from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from solve.MCTS import generate_data


class MockModel:
    def __init__(self, n: int = 11):
        self.n = n

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        # Return random probabilities (not normalized yet as perform_model handles softmax)
        probas = torch.randn(batch_size, self.n * self.n)
        # Return random values between -1 and 1
        values = torch.tanh(torch.randn(batch_size, 1))
        return probas, values


class TestGenerateData(unittest.TestCase):
    def setUp(self):
        self.n = 5  # Smaller board for faster testing
        self.model = MockModel(n=self.n)
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_generate_data_shapes(self):
        n_games = 2
        boards, policies, values = generate_data(
            self.model,
            n_games=n_games,
            n_iter=5,  # Small number of iterations for speed
            temperature_threshold=5,
        )

        assert isinstance(boards, np.ndarray)
        assert isinstance(policies, np.ndarray)
        assert isinstance(values, np.ndarray)

        # Number of positions generated should be at least n_games (if game ends immediately)
        assert len(boards) >= n_games
        assert len(boards) == len(policies)
        assert len(boards) == len(values)

        # Check shapes
        assert boards.shape[1:] == (self.n, self.n)
        assert policies.shape[1] == self.n * self.n
        assert len(values.shape) == 1

    def test_generate_data_save(self):
        n_games = 1
        generate_data(self.model, n_games=n_games, n_iter=2, save_path=self.test_dir)

        files = list(Path(self.test_dir).iterdir())
        assert len(files) == 1
        assert files[0].name.startswith("selfplay_data_")
        assert files[0].name.endswith(".npz")

        # Load and verify data
        data = np.load(files[0])
        assert "boards" in data
        assert "policies" in data
        assert "values" in data
        assert data["n_games"] == n_games

    def test_policy_validity(self):
        n_games = 1
        _, policies, _ = generate_data(self.model, n_games=n_games, n_iter=10)

        for policy in policies:
            # Policy should sum to 1 (or very close)
            # Use np.isclose for almost equal
            assert np.isclose(np.sum(policy), 1.0, atol=1e-5)
            # All probabilities should be >= 0
            assert np.all(policy >= 0)


if __name__ == "__main__":
    unittest.main()
