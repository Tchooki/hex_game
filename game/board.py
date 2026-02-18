"""Board of hex in n x n grid slice"""

from __future__ import annotations

import numpy as np
import torch
from numpy.random import choice

WHITE: int = 1
BLACK: int = -1


class UnionFind:
    """disjoint-set data structure"""

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, i):
        """Find the representative of the set containing i"""
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        """Union the sets containing i and j"""
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


class Pos:
    __slots__ = ("n", "pos", "x", "y")

    def __init__(self, pos: tuple[int, int] | int, n: int) -> None:
        self.n = n
        if isinstance(pos, int):
            self.pos = pos
            self.x, self.y = self.decode_coord(pos)
        elif isinstance(pos, tuple):
            self.x = pos[0]
            self.y = pos[1]
            self.pos = self.encode_coord(pos)
        else:
            raise TypeError

    def __hash__(self) -> int:
        return self.pos

    def __eq__(self, value: object) -> bool:
        if isinstance(value, Pos):
            return self.pos == value.pos
        raise TypeError

    def get(self):
        """
        Getter

        Returns:
            int: encoded pos

        """
        return self.pos

    def __repr__(self) -> str:
        return f"({self.x}, {self.y})"

    def __str__(self) -> str:
        return self.__repr__()

    def encode_coord(self, key: tuple[int, int]) -> int:
        """Encode coord"""
        return key[0] * self.n + key[1]

    def decode_coord(self, key: int) -> tuple[int, int]:
        """Decode coord"""
        return key // self.n, key % self.n

    def get_neighbours(self) -> list[Pos]:
        """
        Get valid neighbours next to key

        Args:
            key (tuple[int, int]): coord

        Yields:
            tuple[int, int]: neighbor coord

        """
        rslt = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if (
                    self.x + i < 0
                    or self.x + i >= self.n
                    or self.y + j < 0
                    or self.y + j >= self.n
                    or i == j
                ):
                    continue
                rslt.append(Pos((self.x + i, self.y + j), self.n))
        return rslt


class Board:
    """Representing Board of hex in n x n grid"""

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
                pos = Pos((i, j), self.n)
                car = self[pos]
                if car == WHITE:
                    chain += "O"
                elif car == BLACK:
                    chain += "X"
                else:
                    chain += " "
                chain += "|"
            chain += "\n" + " " * (i + 1)
        return chain

    def __getitem__(self, pos: Pos) -> int:
        return self._board[pos.get()]

    def __setitem__(self, pos: Pos, new_value: int):
        self._board[pos.get()] = new_value

    def reset(self):
        self.has_won = 0
        self.turn = WHITE
        self._board = [0 for _ in range(self.n**2)]
        self.uf = UnionFind(self.n**2 + 4)

    def light_copy(self):
        new_board = Board(self.n, self.turn)
        new_board._board = self._board[:]  # faster copy
        # Deep copy of UF state
        new_board.uf.parent = self.uf.parent[:]
        new_board.uf.rank = self.uf.rank[:]
        return new_board

    def can_play(self, pos: Pos) -> bool:
        """
        If we can play at coord

        Args:
            pos (Pos): coord

        Returns:
            bool: if we can play

        """
        return self[pos] == 0

    def actions(self) -> list[int]:
        """
        Create generator that yield all possible actions (i.e. empty spaces)

        Yields:
            int: coord

        """
        rslt = []
        for i, val in enumerate(self._board):
            if val == 0:
                rslt.append(i)
        return rslt

    def to_numpy(self, transform=None) -> np.ndarray:
        b = np.array(self._board).reshape(self.n, -1) * self.turn
        if transform:
            b = transform(b)
        return b

    def to_tensor(self, transform=None) -> torch.Tensor:
        b = np.array(self._board).reshape(self.n, -1) * self.turn
        if transform:
            b = transform(b)
        t = torch.from_numpy(b.copy()).unsqueeze(0).unsqueeze(0).float()
        if torch.cuda.is_available():
            t = t.cuda()
        return t

    def play_random(self) -> tuple[Pos, int]:
        actions = self.actions()
        if not actions:
            raise ValueError("Board is full")
        pos = Pos(choice(actions).item(), self.n)
        return pos, self.play(pos)

    def play(self, pos: Pos, verbose=False) -> int:
        """
        Play a move a tell if it's a win or not.

        Args:
            pos (Pos): coord
            verbose (bool): verbose mode

        Returns:
            int: 0 if nobody win else, -1 Black or 1 White

        """
        assert self.can_play(pos), (
            f"Can't play at pos {pos}, {'white' if self[pos] == WHITE else 'black'} has a pawn here."
        )

        self[pos] = self.turn
        idx = pos.get()

        # Connect to borders
        if self.turn == WHITE:
            if pos.y == 0:
                self.uf.union(idx, self.white_top)
            if pos.y == self.n - 1:
                self.uf.union(idx, self.white_bottom)
        else:  # BLACK
            if pos.x == 0:
                self.uf.union(idx, self.black_left)
            if pos.x == self.n - 1:
                self.uf.union(idx, self.black_right)

        # Union with neighbors of same color
        for neighbour in pos.get_neighbours():
            if self[neighbour] == self.turn:
                self.uf.union(idx, neighbour.get())

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
        self.turn = self.turn * -1
        return 0
