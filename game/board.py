"""Board of hex in n x n grid slice."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

WHITE: int = 1
BLACK: int = -1

RNG = np.random.default_rng()


class UnionFind:
    """Disjoint-set data structure."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, i: int) -> int:
        """Find the representative of the set containing i."""
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i: int, j: int) -> bool:
        """Union the sets containing i and j."""
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            if self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            elif self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
            else:
                self.parent[root_j] = root_i
                self.rank[root_i] += 1
            return True
        return False


class Board:
    """Representing Board of hex in n x n grid."""

    def __init__(self, n: int, turn: int = WHITE) -> None:
        self.n = n
        self.action_space = n**2
        self._board = [0 for _ in range(self.n**2)]
        self.turn = turn
        self.has_won = 0

        # Union Find initialization
        # Nodes 0 to n^2-1 are board positions
        # Super nodes for borders:
        # WHITE_TOP = n^2
        # WHITE_BOTTOM = n^2 + 1
        # BLACK_LEFT = n^2 + 2
        # BLACK_RIGHT = n^2 + 3
        self.uf = UnionFind(n**2 + 4)
        self.white_top = n**2
        self.white_bottom = n**2 + 1
        self.black_left = n**2 + 2
        self.black_right = n**2 + 3

    def __repr__(self) -> str:
        chain = ""
        for i in range(self.n):
            chain += "|"
            for j in range(self.n):
                idx = i * self.n + j
                car = self._board[idx]
                if car == WHITE:
                    chain += "O"
                elif car == BLACK:
                    chain += "X"
                else:
                    chain += " "
                chain += "|"
            chain += "\n" + " " * (i + 1)
        return chain

    def decode_coord(self, key: int) -> tuple[int, int]:
        """Decode index to (x, y)."""
        return key // self.n, key % self.n

    def get_neighbours(self, index: int) -> list[int]:
        """Get valid neighbours for a given index."""
        x, y = self.decode_coord(index)
        rslt = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                nx, ny = x + i, y + j
                if (0 <= nx < self.n) and (0 <= ny < self.n) and (i != j):
                    rslt.append(nx * self.n + ny)
        return rslt

    def __getitem__(self, index: int) -> int:
        return self._board[index]

    def __setitem__(self, index: int, new_value: int) -> None:
        self._board[index] = new_value

    def reset(self) -> None:
        self.has_won = 0
        self.turn = WHITE
        self._board = [0 for _ in range(self.n**2)]
        self.uf = UnionFind(self.n**2 + 4)

    def light_copy(self) -> Board:
        new_board = Board(self.n, self.turn)
        new_board._board = self._board[:]  # faster copy
        # Deep copy of UF state
        new_board.uf.parent = self.uf.parent[:]
        new_board.uf.rank = self.uf.rank[:]
        return new_board

    def can_play(self, index: int) -> bool:
        """
        If we can play at index.

        Args:
            index (int): index

        Returns:
            bool: if we can play

        """
        return self._board[index] == 0

    def actions(self) -> list[int]:
        """
        Create generator that yield all possible actions (i.e. empty spaces).

        Yields:
            int: coord

        """
        rslt = []
        for i, val in enumerate(self._board):
            if val == 0:
                rslt.append(i)
        return rslt

    def to_numpy(
        self, transform: Callable[[np.ndarray], np.ndarray] | None = None
    ) -> np.ndarray:
        """
        Convert board to numpy array.

        Args:
            transform (Callable[[np.ndarray], np.ndarray] | None): transform to apply to the board

        Returns:
            np.ndarray: numpy array representation of the board

        """
        b = np.array(self._board).reshape(self.n, -1) * self.turn
        if transform:
            b = transform(b)
        return b

    def to_tensor(
        self, transform: Callable[[np.ndarray], np.ndarray] | None = None
    ) -> torch.Tensor:
        """
        Convert board to tensor.

        Args:
            transform (Callable[[np.ndarray], np.ndarray] | None): transform to apply to the board

        Returns:
            torch.Tensor: tensor representation of the board

        """
        b = np.array(self._board).reshape(self.n, -1) * self.turn
        if transform:
            b = transform(b)
        t = torch.from_numpy(b.copy()).unsqueeze(0).unsqueeze(0).float()
        if torch.cuda.is_available():
            t = t.cuda()
        return t

    def play_random(self) -> tuple[int, int]:
        actions = self.actions()
        if not actions:
            raise ValueError("Board is full")
        index = RNG.choice(actions).item()
        return index, self.play(index)

    def play(self, index: int, verbose: bool = False) -> int:
        """
        Play a move a tell if it's a win or not.

        Args:
            index (int): index
            verbose (bool): verbose mode

        Returns:
            int: 0 if nobody win else, -1 Black or 1 White

        """
        if not self.can_play(index):
            x, y = self.decode_coord(index)
            msg = f"Can't play at pos ({x}, {y}), {'white' if self._board[index] == WHITE else 'black'} has a pawn here."
            raise ValueError(msg)

        self._board[index] = self.turn
        x, y = self.decode_coord(index)

        # Connect to borders
        if self.turn == WHITE:
            if y == 0:
                self.uf.union(index, self.white_top)
            if y == self.n - 1:
                self.uf.union(index, self.white_bottom)
        else:  # BLACK
            if x == 0:
                self.uf.union(index, self.black_left)
            if x == self.n - 1:
                self.uf.union(index, self.black_right)

        # Union with neighbors of same color
        for neighbour in self.get_neighbours(index):
            if self._board[neighbour] == self.turn:
                self.uf.union(index, neighbour)

        # Check win
        if self.turn == WHITE:
            if self.uf.find(self.white_top) == self.uf.find(self.white_bottom):
                if verbose:
                    print(f"Gagné : {self.turn}")
                self.has_won = self.turn
                return self.turn
        elif self.uf.find(self.black_left) == self.uf.find(self.black_right):
            if verbose:
                print(f"Gagné : {self.turn}")
            self.has_won = self.turn
            return self.turn

        # Change turn
        self.turn *= -1
        return 0
