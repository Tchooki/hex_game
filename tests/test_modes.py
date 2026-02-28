from unittest.mock import MagicMock

from hex_game.ui.players import HumanPlayer, RandomPlayer


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
