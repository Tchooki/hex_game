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

    # Play a horizontal line (y=0 to y=2)
    b.play(0 * 3 + 0)  # (0, 0) W
    b.play(1 * 3 + 0)  # (1, 0) B
    b.play(0 * 3 + 1)  # (0, 1) W
    b.play(1 * 3 + 1)  # (1, 1) B
    b.play(0 * 3 + 2)  # (0, 2) W
    # Should be win
    assert b.has_won == WHITE


def test_black_win():
    b = Board(3, turn=BLACK)
    # Black needs to connect left (0,3,6) to right (2,5,8)

    # Play a vertical line (x=0 to x=2)
    b.play(0 * 3 + 0)  # (0, 0) B
    b.play(0 * 3 + 1)  # (0, 1) W
    b.play(1 * 3 + 0)  # (1, 0) B
    b.play(1 * 3 + 1)  # (1, 1) W
    b.play(2 * 3 + 0)  # (2, 0) B
    assert b.has_won == BLACK


def test_no_win():
    b = Board(3)
    b.play(0 * 3 + 0)
    assert b.has_won == 0
