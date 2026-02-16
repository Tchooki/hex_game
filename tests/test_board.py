import os
import sys
import unittest

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from game.board import BLACK, WHITE, Board, Pos


class TestBoard(unittest.TestCase):
    def test_union_find(self):
        b = Board(11)
        # Test basic union/find
        self.assertNotEqual(b.uf.find(0), b.uf.find(1))
        b.uf.union(0, 1)
        self.assertEqual(b.uf.find(0), b.uf.find(1))

    def test_white_win(self):
        b = Board(3)
        # 0 1 2
        #  3 4 5
        #   6 7 8
        # White needs to connect top (0,1,2) to bottom (6,7,8)

        # Play a vertical line
        b.play(Pos((0, 0), 3))  # Top left
        b.play(Pos((1, 0), 3))  # Middle left
        b.play(Pos((2, 0), 3))  # Bottom left
        # Should be win
        self.assertEqual(b.has_won, WHITE)

    def test_black_win(self):
        b = Board(3, turn=BLACK)
        # Black needs to connect left (0,3,6) to right (2,5,8)

        # Play a horizontal line
        b.play(Pos((0, 0), 3))
        b.play(Pos((0, 1), 3))
        b.play(Pos((0, 2), 3))
        self.assertEqual(b.has_won, BLACK)

    def test_no_win(self):
        b = Board(3)
        b.play(Pos((0, 0), 3))
        self.assertEqual(b.has_won, 0)


if __name__ == "__main__":
    unittest.main()
