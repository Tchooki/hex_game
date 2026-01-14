"""Board of hex in n x n grid"""

from numpy.random import choice

from typing import Tuple, Literal, Union, List, Generator
import torch
import numpy as np

WHITE = 1
BLACK = -1

class Pos():
    def __init__(self, pos : Union[Tuple[int, int], int],n: int) -> None:
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
        """Getter

        Returns:
            int: encoded pos
        """
        return self.pos

    def __repr__(self) -> str:
        return f"({self.x}, {self.y})"
    
    def __str__(self) -> str:
        return self.__repr__()

    def encode_coord(self, key : Tuple[int, int]) -> int:
        """Encode coord"""
        return key[0] * self.n + key[1]
    
    def decode_coord(self, key : int) -> Tuple[int, int]:
        """Decode coord"""
        return key // self.n, key % self.n

    def get_neighbours(self) -> List['Pos']:
        """Get valid neighbours next to key

        Args:
            key (Tuple[int, int]): coord

        Yields:
            Tuple[int, int]: neighbor coord
        """
        rslt = []
        for i in range(-1,2):
            for j in range(-1,2):
                if  self.x + i < 0 or self.x + i >= self.n or \
                    self.y + j < 0 or self.y + j >= self.n or \
                    i == j:
                    continue
                rslt.append(Pos((self.x + i, self.y + j), self.n))
        return rslt

class Board():
    """Representing Board of hex in n x n grid
    """
    def __init__(self, n : int, turn = WHITE) -> None:
        self.n = n
        self.action_space = n**2
        self._board = [0 for _ in range(self.n**2)]
        self.turn = turn
        self.has_won = 0

    def __repr__(self) -> str:
        chain = ""
        for i in range(self.n):
            chain += "|"
            for j in range(self.n):
                pos = Pos((i,j), self.n)
                car = self[pos]
                if car == WHITE:
                    chain += "O"
                elif car == BLACK:
                    chain += "X"
                else:
                    chain += " "
                chain += "|"
            chain += "\n" + " "*(i+1)
        return chain

            

    def __getitem__(self, pos : Pos):
        return self._board[pos.get()]

    def __setitem__(self, pos : Pos, new_value : int):
        self._board[pos.get()] = new_value


    def _touch_start(self, pos : Pos, player : int) -> bool:
        return player == BLACK and pos.x == 0 or player == WHITE and pos.y == 0

    def _touch_end(self, pos : Pos, player : int) -> bool:
        return player == BLACK and pos.x == self.n-1 or player == WHITE and pos.y == self.n-1

    def reset(self):
        self.has_won = 0
        self.turn = WHITE
        self._board = [0 for _ in range(self.n**2)]

    def light_copy(self):
        new_board = Board(self.n, self.turn)
        new_board._board = self._board.copy()
        return new_board



    def can_play(self, pos : Pos) -> bool:
        """If we can play at coord

        Args:
            coord (Tuple[int, int]): coord

        Returns:
            bool: if we can play
        """
        return self[pos] == 0

    def get_grid(self):
        """Get all possible coords"""
        for i in range(self.n**2):
            yield Pos(i, self.n)

    def actions(self) -> List[int]:
        """Create generator that yield all possible actions (i.e. empty spaces)

        Yields:
            Tuple[int, int]: coord
        """
        rslt = []
        for pos in self.get_grid():
            if self.can_play(pos):
                rslt.append(pos.get())
        return rslt

    def to_numpy(self, transform = None) -> np.ndarray:
        b = np.array(self._board).reshape(self.n,-1) * self.turn
        if transform:
            b = transform(b)
        return b

    def to_tensor(self, transform=None) -> torch.Tensor:
        b = np.array(self._board).reshape(self.n,-1) * self.turn
        if transform:
            b = transform(b)
        b = torch.from_numpy(b.copy()).unsqueeze(0).unsqueeze(0).float()
        if torch.cuda.is_available():
            b = b.cuda()
        return b

    def play_random(self) -> Tuple[Pos, int]:
        actions = self.actions()
        if not actions:
            raise ValueError("Board is full")
        pos = Pos(choice(actions).item(), self.n)
        return pos, self.play(pos)


    def play(self, pos : Pos, verbose = False) -> int:
        """Play a move a tell if it's a win or not.

        Args:
            pos (Pos): coord
            player (int): player who played

        Returns:
            int: 0 if nobody win else, -1 Black or 1 White
        """
        assert self.can_play(pos), f"Can't play at pos {pos}, {"white" if self[pos] == 1 else "black"} have a pawn here. {self._board}"

        self[pos] = self.turn

        visited = set()
        stack = [pos]
        touch_start_border = False
        touch_end_border = False
        while stack and not (touch_start_border and touch_end_border):
            current = stack.pop()
            visited.add(current)

            if self._touch_start(current, self.turn):
                touch_start_border = True
            if self._touch_end(current, self.turn):
                touch_end_border = True

            for neighbour in current.get_neighbours():
                if  neighbour in visited:
                    continue
                elif self[neighbour] == self[current]:
                    stack.append(neighbour)

        if touch_start_border and touch_end_border:
            if verbose:
                print(f"Gagné : {self.turn}")
            self.has_won = self.turn
            return self.turn
        # Change turn
        self.turn = self.turn * -1
        return 0