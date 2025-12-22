"""Board of hex in n x n grid"""

from typing import Tuple, Literal
from collections import OrderedDict

WHITE = 1
BLACK = -1

class Board():
    """Representing Board of hex in n x n grid
    """
    def __init__(self, n : int) -> None:
        self.n = n
        self.index_conn_component = 1
        self._board = [[0] * self.n for _ in range(self.n)]
        self.has_start_bound = set()
        self.has_end_bound = set()


    def __getitem__(self, key : Tuple[int, int]):
        return self._board[key[0]][key[1]]

    def __setitem__(self, key : Tuple[int, int], new_value : int):
        self._board[key[0]][key[1]] = new_value


    def get_neighbours(self, key : Tuple[int, int]):
        """Get valid neighbours next to key

        Args:
            key (Tuple[int, int]): coord

        Yields:
            Tuple[int, int]: neighbor coord
        """
        for i in range(-1,2):
            for j in range(-1,2):
                if  key[0] + i < 0 or key[0] + i >= self.n or \
                    key[1] + j < 0 or key[1] + j >= self.n or \
                    i == j:
                    continue
                yield key[0] + i, key[1] + j

    def _propagate_component(self, key : Tuple[int, int], old_value : int, new_value : int):
        for i,j in self.get_neighbours(key):
            if self[i, j] == old_value:
                self[i, j] = new_value
                self._propagate_component((i, j), old_value, new_value)

    def touch_start(self, key : Tuple[int, int], player : Literal["white", "black"]) -> bool:
        return player == "black" and key[0] == 0 or player == "white" and key[1] == 0
        
    def touch_end(self, key : Tuple[int, int], player : Literal["white", "black"]) -> bool:
        return player == "black" and key[0] == self.n-1 or player == "white" and key[1] == self.n-1

    def set_bound(self, key : Tuple[int, int], player : Literal["white", "black"]):
        """Set id componant in sets if they touch border
        /!\ Required to be called after affecting board

        Args:
            key (Tuple[int, int]): coord
            player (Literal[&quot;white&quot;, &quot;black&quot;]): player
        """
        if self.touch_start(key, player):
            self.has_start_bound.add(self[key])
        if self.touch_end(key, player):
            self.has_end_bound.add(self[key])

    def transfert_bound(self, old, new):
        if old in self.has_start_bound:
            self.has_start_bound.remove(old)
            self.has_start_bound.add(new)
        if old in self.has_end_bound:
            self.has_end_bound.remove(old)
            self.has_end_bound.add(new)

    def play(self, key, player : Literal["white", "black"]) -> int:
        """Play a move a tell if it's a win or not.

        Args:
            key (_type_): coord
            player (Literal[&quot;white&quot;, &quot;black&quot;]): player who played

        Returns:
            int: 0 if nobody win else, -1 Black or 1 White
        """
        sign = 1 if player == "white" else -1
        dico = OrderedDict()
        for i, j in self.get_neighbours(key):
            if self[i, j]*sign >= 1:
                dico[i, j] = self[i, j]
        if dico:
            _, new_value = dico.popitem()
            self[key] = new_value
            self.set_bound(key, player)
            for neighbour in dico:
                if self[neighbour] != new_value:
                    old_value = self[neighbour]
                    self[neighbour] = new_value
                    # Transfert bounds
                    self.transfert_bound(old_value, new_value)
                    self._propagate_component(neighbour, old_value, new_value)
        else:
            self[key] = sign*self.index_conn_component
            self.index_conn_component += 1
            self.set_bound(key, player)
        if self[key] in self.has_start_bound and self[key] in self.has_end_bound:
            print(f"Gagné : {player}")
            return sign
        return 0
