import argparse
from unittest.mock import MagicMock, patch

from hex_game.main import main
from hex_game.ui.players import AIPlayer, HumanPlayer, RandomPlayer


def test_human_player_get_move():
    player = HumanPlayer()
    board = MagicMock()
    # HumanPlayer.get_move should return None as it's handled by GUI
    assert player.get_move(board) is None


def test_random_player_get_move():
    player = RandomPlayer()
    board = MagicMock()
    board.play_random.return_value = (42, None)
    assert player.get_move(board) == 42


@patch("hex_game.ai.mcts.MCTS")
@patch("hex_game.ai.mcts.RootNode")
def test_ai_player_get_move(mock_root_node, mock_mcts):
    model = MagicMock()
    player = AIPlayer(model, time_limit=0.1)
    board = MagicMock()

    # Mock RootNode and its children_N attribute
    mock_root = MagicMock()
    mock_root.children_N = [10, 20, 30]  # Action 2 has most visits
    mock_root_node.return_value = mock_root

    # Temperature 0.1 (greedy)
    assert player.get_move(board, move_count=20) == 2


@patch("hex_game.main.HexBoard")
@patch("argparse.ArgumentParser.parse_args")
def test_main_human_vs_human(mock_parse_args, mock_hex_board):
    # Mock arguments for Human vs Human (default)
    mock_parse_args.return_value = argparse.Namespace(
        ai=False, profile=False, size=11, run="test_run", time=1.0
    )

    main()

    # Verify HexBoard was initialized with two HumanPlayers
    args, kwargs = mock_hex_board.call_args
    assert isinstance(kwargs["player_white"], HumanPlayer)
    assert isinstance(kwargs["player_black"], HumanPlayer)
    mock_hex_board.return_value.run.assert_called_once()


@patch("hex_game.main.HexNet")
@patch("hex_game.main.torch.load")
@patch("hex_game.main.HexBoard")
@patch("argparse.ArgumentParser.parse_args")
@patch("pathlib.Path.exists")
def test_main_human_vs_ai(
    mock_exists, mock_parse_args, mock_hex_board, mock_torch_load, mock_hex_net
):
    # Mock arguments for Human vs AI
    mock_parse_args.return_value = argparse.Namespace(
        ai=True, profile=False, size=11, run="test_run", time=1.0
    )
    mock_exists.return_value = True

    main()

    # Verify HexBoard was initialized with HumanPlayer and AIPlayer
    args, kwargs = mock_hex_board.call_args
    assert isinstance(kwargs["player_white"], HumanPlayer)
    assert isinstance(kwargs["player_black"], AIPlayer)
    mock_hex_board.return_value.run.assert_called_once()


@patch("cProfile.run")
@patch("argparse.ArgumentParser.parse_args")
def test_main_profiling(mock_parse_args, mock_cprofile_run):
    # Mock arguments for Profiling
    mock_parse_args.return_value = argparse.Namespace(
        ai=False, profile=True, size=11, run="test_run", time=1.0
    )

    main()

    # Verify cProfile.run was called
    mock_cprofile_run.assert_called_once()
    args, kwargs = mock_cprofile_run.call_args
    assert "generate_data" in args[0]
