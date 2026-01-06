import threading
import queue
import time

from typing import List, Tuple, Optional, Union, TYPE_CHECKING
import networkx as nx
from IPython.display import clear_output
from copy import deepcopy


import torch
import numpy as np

from graphics.display import HexBoard
from game.board import Board, Pos, WHITE, BLACK

if TYPE_CHECKING:
    from solve.model import HexNet
    import torch.nn as nn

class RootNode():
    def __init__(self, state : Board) -> None:
        self.action = None
        self.parent = None
        self.is_expanded = False
        self.state = deepcopy(state)
        self.possible_actions = self.state.actions()
        self.children_P = np.zeros(self.state.action_space, dtype=float)
        self.children_sum_V = np.zeros(self.state.action_space, dtype=float)
        self.children_N = np.zeros(self.state.action_space, dtype=int)
        self.children : dict[int, Node] = dict()
        self.N_root = 0
        self.sum_V_root = 0

    def build_graph(self):
        G = nx.DiGraph()
        stack = [self]
        G.add_node("Root")
        while stack:
            current = stack.pop()
            node_id = str(Pos(current.action, self.state.n)) if isinstance(current, Node) else "Root"
            for child in current.children.values():
                child_id = str(Pos(child.action, self.state.n))
                G.add_edge(node_id, child_id)
                stack.append(child)

        return G

    @property
    def N(self):
        return self.N_root

    @N.setter
    def N(self, value):
        self.N_root = value

    @property
    def sum_V(self):
        return self.sum_V_root

    @sum_V.setter
    def sum_V(self, value):
        self.sum_V_root = value


    @property
    def Q(self):
        """Action value"""
        if self.N == 0:
            return 0
        return self.sum_V/self.N

    @property
    def children_Q(self):
        """Action values"""
        Q = np.zeros_like(self.children_sum_V, dtype=float)-100
        mask = self.children_N > 0
        Q[mask] = self.children_sum_V[mask] / self.children_N[mask]
        return Q

    @property
    def children_U(self, c = 1):
        """Explore factor"""
        return c * self.children_P * np.sqrt(self.children_N.sum()+1) / (1 + self.children_N)

    def get_policy(self):
        policy = np.zeros(self.state.action_space, dtype=float)
        mask = self.children_N > 0
        policy[mask] = self.children_N[mask] / self.N
        return policy

    def expand(self, proba_model : np.ndarray):
        """Expand node"""
        self.is_expanded = True
        actions = self.state.actions()
        
        for i in range(len(proba_model)): # Mask ilegal move
            if i not in actions:
                proba_model[i] = 0.0
                self.children_Q[i] = -float("inf")

        self.children_P = np.array(proba_model)

    def select(self):
        """Select leaf node according to Q + U"""
        current = self
        while current.is_expanded:
            current = current.best_child()
        return current

    def backpup(self, value : float):
        """Update V N recursively"""
        current = self
        while current:
            current.N += 1
            current.sum_V += value
            current = current.parent

    def best_child(self) -> "Node":
        """Return best child according to Q + U"""
        i_max = np.argmax((self.children_Q + self.children_U)).item()
        if i_max in self.children:
            return self.children[i_max]
        else:
            new_state = deepcopy(self.state)
            try :
                new_state.play(Pos(i_max, self.state.n))
            except AssertionError:
                print("Problem")
                print("Q:",self.children_Q)
                print("U:",self.children_U)
            child = Node(new_state, parent=self , action = i_max)
            self.children[i_max] = child
            return child


class Node(RootNode):
    def __init__(self, state : Board, parent : RootNode, action : int) -> None:
        super().__init__(state)
        self.action = action
        self.parent = parent

    @property
    def N(self):
        """Number of visit getter"""
        return self.parent.children_N[self.action]
    
    @N.setter
    def N(self, value):
        """Number of visit setter"""
        self.parent.children_N[self.action] = value

    @property
    def sum_V(self):
        """Sum of values getter"""
        return self.parent.children_sum_V[self.action]
    
    @sum_V.setter
    def sum_V(self, value):
        """Sum of values setter"""
        self.parent.children_sum_V[self.action] = value

def get_random_transformation():
    """Get random transformation (reflexion along diagonals)"""
    ref1, ref2 = np.random.random(2) > 0.5

    def transform(b : np.ndarray):
        if ref1:
            b = b.T
        if ref2:
            b = np.rot90(b, 2).T
        return b

    return transform

def perform_model(model : "HexNet", batch_leaves : List[RootNode]) -> Tuple[List[np.ndarray], np.ndarray]:
    """Forward model and add reflexions

    Args:
        model (HexNet): model
        leaf (RootNode): Node

    Returns:
        Tuple[np.ndarray, float]: Proba and value
    """
    transforms = [get_random_transformation() for _ in range(len(batch_leaves))]
    states = []
    with torch.no_grad():
        # Create batch
        for i, leaf in enumerate(batch_leaves):
            state = leaf.state.to_numpy(transforms[i])
            state = torch.from_numpy(state.copy()).unsqueeze(0).float()  # add extra dim for channel
            if torch.cuda.is_available():
                state = state.cuda()
            states.append(state)
        inputs = torch.stack(states)
        #Eval
        probas, values = model.forward(inputs)

        probas = torch.nn.functional.softmax(probas, dim=1)
        probas = [transforms[i](probas[i].cpu().numpy().reshape(model.n,-1)).reshape(-1) for i in range(len(batch_leaves))]
        values = values.cpu().squeeze(dim=1).numpy()
        return probas, values


def MCTS(root : RootNode, model : "HexNet", batch_size = 16, timeout : float = 0.010, n_iter : int = 100):
    """Perform Monte-Carlo Tree Search

    Args:
        root (RootNode): RootNode
        model (HexNet): DL model to predict value and proba
        n_iter (int, optional): Number of iterations of MCTS. Defaults to 100.

    Returns:
        RootNode: the tree created
    """
    # First Root eval
    probas, values = perform_model(model, [root])
    root.expand(probas[0])
    root.backpup(values[0].item())

    batch_leaves = []
    start_time = time.perf_counter()
    for _ in range(n_iter):
        # Find  best child Q + U
        leaf = root.select()


        if len(batch_leaves) < batch_size and time.perf_counter() - start_time < timeout:
            batch_leaves.append(leaf)
            leaf.sum_V -= 10 # Virtual loss
        else:
            # predict with model
            # if len(batch_leaves) < batch_size:
            #     print("Timout", len(batch_leaves))
            probas, values = perform_model(model, batch_leaves)
            # Expand & backup
            for i, (proba, value) in enumerate(zip(probas, values)):
                leaf.sum_V += 10 # Restore loss
                if leaf.state.has_won == 0:
                    leaf.expand(proba)
                # Update V N
                leaf.backpup(value)
            start_time = time.perf_counter()
            batch_leaves = []

    return root

def generate_data(model, n_games, n_random_plays = 1, n_iter = 10, show = False):
    """Generate data for training according to model

    Args:
        model (HexNet): model that predict
        n_games (int): number of games generated
        n_random_plays (int, optional): first plays that are chosen randomly. Defaults to 1.
        n_iter (int, optional): number of iteration in MCTS. Defaults to 10.
        show (bool, optional): Show with pygame the generation of data. Defaults to False.

    Returns:
        _type_: data
    """
    b = Board(11)
    boards = []
    policies = []
    values = []
    move_queue = None

    if show:
        move_queue = queue.Queue()
        threading.Thread(target = lambda : HexBoard(11).run(mode='training', move_queue=move_queue), daemon=True).start()

    for _ in range(n_games):
        b.reset()
        if show:
            move_queue.put_nowait("reset")
        for _ in range(n_random_plays):
            pos, _ = b.play_random()
            if show:
                move_queue.put_nowait(pos)
        root = RootNode(b)
        current = root
        while current.state.has_won == 0:
            MCTS(current, model, n_iter=n_iter)
            current : Node = current.best_child()
            if show:
                move_queue.put_nowait(Pos(current.action, current.state.n))
        won = current.state.has_won

        while current is not None:
            boards.append(current.state.to_tensor())
            policies.append(current.children_P)
            values.append(current.Q)
            current = current.parent

    return won
