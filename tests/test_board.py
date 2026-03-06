from hex_game.game.board import BLACK, WHITE, Board


def test_union_find():
    b = Board(11)
    # Test basic union/find
    assert b.uf.find(0) != b.uf.find(1)
    b.uf.union(0, 1)
    assert b.uf.find(0) == b.uf.find(1)


def test_white_win():
    b = Board(3)
    # 0 1 2
    #  3 4 5
    #   6 7 8
    # White needs to connect top (0,1,2) to bottom (6,7,8)

    # Play a vertical line (x=0 to x=2)
    b.play(0 * 3 + 0)  # (0, 0) W
    b.play(0 * 3 + 1)  # (0, 1) B
    b.play(1 * 3 + 0)  # (1, 0) W
    b.play(1 * 3 + 1)  # (1, 1) B
    b.play(2 * 3 + 0)  # (2, 0) W
    # Should be win
    assert b.has_won == WHITE


def test_black_win():
    b = Board(3, turn=BLACK)
    # Black needs to connect left (0,3,6) to right (2,5,8)

    # Play a horizontal line (y=0 to y=2)
    b.play(0 * 3 + 0)  # (0, 0) B
    b.play(1 * 3 + 0)  # (1, 0) W
    b.play(0 * 3 + 1)  # (0, 1) B
    b.play(1 * 3 + 1)  # (1, 1) W
    b.play(0 * 3 + 2)  # (0, 2) B
    assert b.has_won == BLACK


def test_invalid_move():
    b = Board(3)
    b.play(0)
    import pytest

    with pytest.raises(ValueError, match="Can't play at pos"):
        b.play(0)


def test_turn_switching():
    b = Board(3)
    assert b.turn == WHITE
    b.play(0)
    assert b.turn == BLACK
    b.play(1)
    assert b.turn == WHITE


def test_canonical_form():
    # WHITE turn (1)
    b = Board(3)
    b.play(0)  # (0,0) W -> Board: [1, 0, ..., 0]
    # In canonical: current player is 1, board * current_player
    # Since it's BLACK's turn (-1), current_player = -1
    arr = b.to_numpy(canonical=True)
    # W pawn at (0,0) becomes 1 * -1 = -1
    # Since it's canonical and turn=BLACK, it's transposed.
    # (0,0).T is still (0,0)
    assert arr[0, 0] == -1

    # Switch back to WHITE turn for easier comparison
    b.play(1)  # (0,1) B
    arr_white = b.to_numpy(canonical=True)
    # Now it's WHITE turn, current_player = 1
    # W at (0,0) -> 1 * 1 = 1
    # B at (0,1) -> -1 * 1 = -1
    assert arr_white[0, 0] == 1
    assert arr_white[0, 1] == -1


def test_full_board_no_draw():
    # On a hex board, if all cells are filled, one player MUST have won.
    n = 2
    b = Board(n)
    # 0 1
    #  2 3
    # White needs 0->2 or 1->3 (top to bottom)
    # Black needs 0->1 or 2->3 (left to right)

    # Let's fill the board:
    # W B
    #  W B
    b.play(0)  # (0,0) W
    b.play(1)  # (0,1) B
    b.play(2)  # (1,0) W
    b.play(3)  # (1,1) B
    assert b.has_won == WHITE
