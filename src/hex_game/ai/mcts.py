from __future__ import annotations

import pathlib
import queue
import threading
import time
from collections.abc import Callable
from copy import deepcopy
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import torch

from hex_game.game.board import Board
from hex_game.ui.display import HexBoard

if TYPE_CHECKING:
    # import torch.nn as nn

    from hex_game.ai.model import HexNet


# pylint: disable=invalid-name
RNG = np.random.default_rng()


class RootNode:
    """RootNode."""

    def __init__(self, state: Board) -> None:
        self.action: int | None = None
        self.parent: RootNode | None = None
        self.is_expanded: bool = False
        self.state: Board = deepcopy(state)
        self.possible_actions: list[int] = self.state.actions()
        self.children: dict[int, Node] = {}

        self.children_P = np.zeros(self.state.action_space, dtype=float)
        self.children_sum_V = np.zeros(self.state.action_space, dtype=float)
        self.children_N = np.zeros(self.state.action_space, dtype=int)
        self.sum_children_N = 0

        self.allow_update = True
        self._children_Q = np.zeros(self.state.action_space, dtype=float)
        self._children_U: np.ndarray
        self.need_update_U = True
        self.update_children_U()

        self.N_root: int = 0
        self.sum_V_root: float = 0

    def build_graph(self) -> nx.DiGraph:
        graph: nx.DiGraph = nx.DiGraph()
        stack = [self]
        graph.add_node("Root")
        while stack:
            current = stack.pop()
            node_id = (
                f"({current.action // self.state.n}, {current.action % self.state.n})"
                if isinstance(current, Node) and current.action is not None
                else "Root"
            )
            for action, child in current.children.items():
                child_id = f"({action // self.state.n}, {action % self.state.n})"
                graph.add_edge(node_id, child_id)
                stack.append(child)

        return graph

    @property
    def N(self) -> int:
        return self.N_root

    @N.setter
    def N(self, value: int) -> None:
        self.N_root = value

    @property
    def sum_V(self) -> float:
        return self.sum_V_root

    @sum_V.setter
    def sum_V(self, value: float) -> None:
        self.sum_V_root = value

    @property
    def Q(self):
        """Action value"""
        return self.sum_V / self.N if self.N != 0 else 0

    def update_children_Q(self, action):
        """Update children Q"""
        if self.allow_update and self.children_N[action] > 0:
            self._children_Q[action] = (
                self.children_sum_V[action] / self.children_N[action]
            )

    @property
    def children_Q(self):
        """Action values"""
        return self._children_Q

    def update_children_U(self, c: float = 1.0) -> None:
        """Update children U."""
        if self.allow_update:
            self._children_U = (
                c
                * self.children_P
                * np.sqrt(self.sum_children_N + 1)
                / (1 + self.children_N)
            )

    @property
    def children_U(self) -> np.ndarray:
        """Explore factor."""
        return self._children_U

    def get_policy(self, temperature: float = 1.0) -> np.ndarray:
        """
        Get policy from visit counts with temperature.

        Args:
            temperature (float): Temperature parameter.
                - temperature = 1: policy proportional to visit counts
                - temperature → 0: policy becomes argmax (best move only)
                - temperature > 1: more exploration

        Returns:
            np.ndarray: Policy distribution over actions

        """
        if self.N == 0:
            return np.zeros(self.state.action_space, dtype=float)

        if temperature == 0:
            # Deterministic: argmax
            policy = np.zeros(self.state.action_space, dtype=float)
            best_action = np.argmax(self.children_N)
            policy[best_action] = 1.0
            return policy

        # Apply temperature: π(a) ∝ N(a)^(1/τ)
        visits_temp = np.power(self.children_N, 1.0 / temperature)
        if np.sum(visits_temp) == 0:
            # Return uniform distribution over possible actions if no visits yet
            policy = np.zeros(self.state.action_space, dtype=float)
            actions = self.state.actions()
            if actions:
                policy[actions] = 1.0 / len(actions)
            return policy

        policy = visits_temp / np.sum(visits_temp)
        return policy

    def sample_action(self, temperature: float = 1.0) -> int:
        """
        Sample an action according to the policy with temperature.

        Args:
            temperature (float): Temperature parameter.

        Returns:
            int: Sampled action index

        """
        policy = self.get_policy(temperature=temperature)
        action = RNG.choice(self.state.action_space, p=policy)
        return action

    def expand(self, proba_model: np.ndarray):
        """Expand node"""
        self.is_expanded = True
        actions = self.state.actions()

        for i in range(len(proba_model)):  # Mask illegal move
            if i not in actions:
                proba_model[i] = 0.0  # P = 0 (ensures U = 0)
                self._children_Q[i] = -np.inf  # Q = -inf (double safety)

        self.children_P = np.asarray(proba_model)
        self.update_children_U()

    def select(self):
        """Select leaf node according to Q + U"""
        current = self
        while current.is_expanded:
            current = current.best_child()
        return current

    def backup(self, value: float):
        """Update V N recursively"""
        current = self
        while isinstance(current, Node):
            current.allow_update = False
            current.N += 1
            current.sum_V += value
            value = -value  # Negate value as we go up (players alternate)
            assert current.parent is not None
            current.parent.update_children_Q(current.action)
            current.parent.need_update_U = True
            current.allow_update = True
            current = current.parent
        current.N += 1
        current.sum_V += value

    def best_child(self) -> Node:
        """Return best child according to Q + U"""
        if self.need_update_U:
            self.update_children_U()
            self.need_update_U = False

        i_max = np.argmax(self.children_Q + self.children_U).item()
        if i_max in self.children:
            return self.children[i_max]
        new_state = self.state.light_copy()
        try:
            new_state.play(i_max)
        except AssertionError:
            print("Problem")
            print("Q:", self.children_Q)
            print("U:", self.children_U)
        child = Node(new_state, parent=self, action=i_max)
        self.children[i_max] = child
        return child

    def sample_child(self, temperature: float = 1.0) -> Node:
        """
        Sample a child according to policy with temperature.

        Args:
            temperature (float): Temperature parameter for sampling.
                - temperature = 1: proportional to visit counts
                - temperature → 0: best move (equivalent to best_child)
                - temperature > 1: more exploration

        Returns:
            Node: Sampled child node

        """
        action = self.sample_action(temperature=temperature)

        # Get or create the child for this action
        if action in self.children:
            return self.children[action]
        new_state = self.state.light_copy()
        new_state.play(action)
        child = Node(new_state, parent=self, action=action)
        self.children[action] = child
        return child


class Node(RootNode):
    def __init__(self, state: Board, parent: RootNode, action: int) -> None:
        super().__init__(state)
        self.action = action
        self.parent = parent

    @property
    def N(self):
        """Number of visit getter"""
        assert self.parent is not None
        return self.parent.children_N[self.action]

    @N.setter
    def N(self, value):
        """Number of visit setter"""
        assert self.parent is not None
        self.parent.sum_children_N += value - self.parent.children_N[self.action]
        self.parent.children_N[self.action] = value
        if self.allow_update:
            self.parent.update_children_Q(self.action)
            self.parent.need_update_U = True

    @property
    def sum_V(self):
        """Sum of values getter."""
        assert self.parent is not None
        return self.parent.children_sum_V[self.action]

    @sum_V.setter
    def sum_V(self, value):
        """Sum of values setter."""
        assert self.parent is not None
        self.parent.children_sum_V[self.action] = value
        if self.allow_update:
            self.parent.update_children_Q(self.action)


def get_random_transformation() -> Callable[[np.ndarray], np.ndarray]:
    """Get random transformation (reflexion along diagonals)."""
    ref1, ref2 = RNG.random(2) > 0.5

    def transform(b: np.ndarray) -> np.ndarray:
        if ref1:
            b = b.T
        if ref2:
            b = np.rot90(b, 2).T
        return b

    return transform


def perform_model(
    model: HexNet,
    batch_leaves: list[RootNode],
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Forward model and add reflexions.

    Args:
        model (HexNet): model
        batch_leaves (list[RootNode]): leaves nodes

    Returns:
        tuple[np.ndarray, float]: Proba and value

    """
    transforms = [get_random_transformation() for _ in range(len(batch_leaves))]
    # Create batch and apply transformations
    states = np.stack(
        [leaf.state.to_numpy(transforms[i]) for i, leaf in enumerate(batch_leaves)],
    )
    inputs = torch.from_numpy(states).unsqueeze(1).float()
    if torch.cuda.is_available():
        inputs = inputs.cuda()
    # Eval
    with torch.no_grad():
        probas, values = model.forward(inputs)
        probas = torch.nn.functional.softmax(probas, dim=1)
    # CPU
    probas_numpy = probas.cpu().numpy()
    values_numpy = values.cpu().squeeze(1).numpy()

    # undo transforms
    probas_transformed = [
        transforms[i](probas_numpy[i].reshape(model.n, -1)).reshape(-1)
        for i in range(len(batch_leaves))
    ]
    return probas_transformed, values_numpy


def MCTS(
    root: RootNode,
    model: HexNet,
    batch_size: int = 16,
    timeout: float = 0.010,
    n_iter: int = 100,
) -> RootNode:
    """
    Perform Monte-Carlo Tree Search.

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
    root.backup(values[0].item())

    batch_leaves: list[RootNode] = []
    start_time = time.perf_counter()
    for idx_iter in range(n_iter):
        # Find  best child Q + U
        leaf = root.select()
        batch_leaves.append(leaf)
        leaf.sum_V -= 10  # Virtual loss

        if (
            len(batch_leaves) == batch_size
            or (time.perf_counter() - start_time >= timeout and len(batch_leaves) > 0)
            or idx_iter == n_iter - 1
        ):
            # predict with model
            probas, values = perform_model(model, batch_leaves)
            # Expand & backup
            for j, (proba, value) in enumerate(zip(probas, values, strict=True)):
                batch_leaves[j].sum_V += 10  # Restore virtual loss
                if batch_leaves[j].state.has_won == 0:
                    batch_leaves[j].expand(proba)
                # Update V N
                batch_leaves[j].backup(value)
            start_time = time.perf_counter()
            batch_leaves = []

    return root


def generate_data(
    model: HexNet,
    n_games: int,
    n_random_plays: int = 1,
    n_iter: int = 10,
    show: bool = False,
    save_path: str | None = None,
    temperature: float = 1.0,
    temperature_threshold: int = 15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate data for training according to model.

    Args:
        model (HexNet): model that predict
        n_games (int): number of games generated
        n_random_plays (int, optional): first plays that are chosen randomly.
                                        Defaults to 1.
        n_iter (int, optional): number of iteration in MCTS. Defaults to 10.
        show (bool, optional): Show with pygame the generation of data.
                            Defaults to False.
        save_path (str, optional): Path to save data in compressed format.
                            If None, data is not saved. Defaults to None.
        temperature (float, optional): Temperature for move selection. Defaults to 1.0.
        temperature_threshold (int, optional): Number of moves after which
                            temperature → 0. Defaults to 15.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: boards, policies, values for training

    """
    boards: list[np.ndarray] = []
    policies: list[np.ndarray] = []
    values: list[float] = []
    move_queue: queue.Queue | None = None

    b = Board(model.n)

    if show:
        move_queue = queue.Queue()
        threading.Thread(
            target=lambda: HexBoard(model.n).run(
                mode="training",
                move_queue=move_queue,
            ),
            daemon=True,
        ).start()

    for _game_idx in range(n_games):
        b.reset()
        if show and move_queue is not None:
            move_queue.put_nowait("reset")
        for _ in range(n_random_plays):
            pos, _ = b.play_random()
            if show and move_queue is not None:
                move_queue.put_nowait(pos)
        root = RootNode(b)
        current = root
        game_history = []  # Store (node, state) for this game
        move_count = 0

        while current.state.has_won == 0:
            MCTS(current, model, n_iter=n_iter)
            game_history.append(current)

            # Use temperature-based selection early, then switch to best move
            tau = temperature if move_count < temperature_threshold else 0
            current = current.sample_child(temperature=tau)

            move_count += 1
            if show and move_queue is not None and current.action is not None:
                move_queue.put_nowait(current.action)

        # Game finished, get the winner
        won = current.state.has_won

        # Add the final position
        game_history.append(current)

        # Collect training data from this game (skip terminal state)
        for node in game_history[:-1]:
            boards.append(node.state.to_numpy())

            # Use improved policy from MCTS (based on visit counts)
            policies.append(node.get_policy())

            # Value target is the game outcome from current player's perspective
            value_target = won * node.state.turn
            values.append(value_target)

    # Convert lists to numpy arrays
    boards_array = np.array(boards, dtype=np.float32)
    policies_array = np.array(policies, dtype=np.float32)
    values_array = np.array(values, dtype=np.float32)

    # Save to disk if path is provided
    if save_path is not None:
        # Create data directory if it doesn't exist
        pathlib.Path(save_path).mkdir(exist_ok=True, parents=True)

        # Generate filename with timestamp
        timestamp = int(time.time())
        filename = pathlib.Path(save_path) / f"selfplay_data_{timestamp}.npz"

        # Save in compressed format
        np.savez_compressed(
            filename,
            boards=boards_array,
            policies=policies_array,
            values=values_array,
            n_games=n_games,
            n_iter=n_iter,
        )
        print(f"Data saved to {filename}")
        print(f"Total positions: {len(boards_array)}")

    return boards_array, policies_array, values_array
