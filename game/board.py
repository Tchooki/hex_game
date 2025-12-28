"""Board of hex in n x n grid"""

from typing import Tuple, Literal, Union, List, Generator

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
        self._max_label = 1
        self._board = [0 for _ in range(self.n**2)]
        self.turn = turn
        self.has_won = 0

    def __getitem__(self, pos : Pos):
        return self._board[pos.get()]

    def __setitem__(self, pos : Pos, new_value : int):
        self._board[pos.get()] = new_value


    def _touch_start(self, pos : Pos, player : int) -> bool:
        return player == BLACK and pos.x == 0 or player == WHITE and pos.y == 0

    def _touch_end(self, pos : Pos, player : int) -> bool:
        return player == BLACK and pos.x == self.n-1 or player == WHITE and pos.y == self.n-1

    def play_copy(self, action_pos : Pos, player : Literal["white", "black"]):
        new_board = Board(self.n)
        new_board._board = self._board.copy()
        new_board[action_pos] = 1 if player == "white" else -1
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

    def play(self, pos : Pos) -> int:
        """Play a move a tell if it's a win or not.

        Args:
            pos (Pos): coord
            player (int): player who played

        Returns:
            int: 0 if nobody win else, -1 Black or 1 White
        """
        assert self.can_play(pos), f"Can't play at pos {pos}, {"white" if self[pos] == 1 else "black"} have a pawn here."
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
            print(f"Gagné : {self.turn}")
            self.has_won = self.turn
            return self.turn
        # Change turn
        self.turn = self.turn * -1
        return 0