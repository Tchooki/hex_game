from hex_game.ai.mcts import generate_data

from .test_generate_data import MockModel


def test_generate_data_parallel():
    n = 5
    model = MockModel(n=n)
    n_games = 4

    # Test with batch_size = 2
    boards, policies, values = generate_data(
        model, n_games=n_games, batch_size=2, n_iter=2
    )

    assert len(boards) >= n_games
    assert len(boards) == len(policies) == len(values)


def test_generate_data_odd_batch():
    n = 5
    model = MockModel(n=n)
    n_games = 3

    # Test with batch_size = 2 (should handle 2 games then 1 game)
    boards, policies, values = generate_data(
        model, n_games=n_games, batch_size=2, n_iter=2
    )

    assert len(boards) >= n_games
