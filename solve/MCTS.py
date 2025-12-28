from typing import List, Tuple, Optional, Union

from copy import deepcopy

import numpy as np

from game.board import Board, Pos, WHITE, BLACK

class RootNode():
    def __init__(self, state : Board) -> None:
        self.parent = None
        self.state = state
        self.children_P = np.zeros(self.state.action_space, dtype=float)
        self.children_sum_V = np.zeros(self.state.action_space, dtype=float)
        self.children_N = np.zeros(self.state.action_space, dtype=int)
        self.children : dict[int, Node] = dict()

        self.N = 0
        self.sum_V = 0

    @property
    def Q(self):
        return self.sum_V/self.N

    @property
    def children_Q(self):
        return self.children_sum_V / self.children_N

    @property
    def children_U(self, c = 1):
        return self.children_P / (1 + self.children_N)

    def expand(self, proba_model : List[float]):
        self.is_expanded = True
        actions = self.state.actions()
        
        for i in range(len(proba_model)): # Mask ilegal move
            if i not in actions:
                proba_model[i] = 0.0

        self.children_P = np.array(proba_model)

    def select(self):
        current = self
        while current.is_expanded:
            current = current.best_child()
        return current

    def backpup(self, value : float):
        current = self
        while isinstance(current, Node):
            current.N += 1
            if current.state.turn == WHITE:
                current.sum_V += value * current.state.turn
            current = current.parent

    def best_child(self) -> "Node":
        i_max = np.argmax(self.children_Q + self.children_U).item()

        if i_max in self.children:
            return self.children[i_max]
        else:
            new_state = deepcopy(self.state)
            new_state.play(Pos(i_max, self.state.n))
            child = Node(new_state, action = i_max, parent=self)
            self.children[i_max] = child
            return child


class Node(RootNode):
    def __init__(self, state : Board, parent : Union[RootNode, "Node"], action : int) -> None:
        super().__init__(state)
        self.action = action
        self.parent = parent

    @property
    def N(self):
        return self.parent.children_N[self.action]
    
    @N.setter
    def N(self, value):
        self.parent.children_N[self.action] = value

    @property
    def sum_V(self):
        return self.parent.children_sum_V[self.action]
    
    @sum_V.setter
    def sum_V(self, value):
        self.parent.children_sum_V[self.action] = value


    
def MCTS(state : Board, n_iter : int = 1000, model):
    root = RootNode(state)
    for _ in range(n_iter):
        # Find  best child Q + U
        leaf = root.select()
        # predict with model
        proba, value = model.predict(leaf.state)
        
        leaf.expand(proba)
        # Update V N
        leaf.backpup(value)
        