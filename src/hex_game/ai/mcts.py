from __future__ import annotations

import pathlib
import queue
import threading
import time
from collections.abc import Callable

# from copy import deepcopy
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import torch

from hex_game.game.board import Board
from hex_game.ui.board_view import HexBoard
from hex_game.ui.players import QueuePlayer

if TYPE_CHECKING:
    from hex_game.ai.model import HexNet


# pylint: disable=invalid-name
RNG = np.random.default_rng()


class RootNode:
    """RootNode."""

    def __init__(self, state: Board) -> None:
        self.action: int | None = None
        self.parent: RootNode | None = None
        self.is_expanded: bool = False
        self.state: Board = state  # .light_copy()
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
    def Q(self) -> float:
        """Action value."""
        return self.sum_V / self.N if self.N != 0 else 0

    def update_children_Q(self, action: int) -> None:
        """Update children Q."""
        if self.allow_update and self.children_N[action] > 0:
            self._children_Q[action] = (
                self.children_sum_V[action] / self.children_N[action]
            )

    @property
    def children_Q(self) -> np.ndarray:
        """Action values."""
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

    def expand(self, proba_model: np.ndarray) -> None:
        """Expand node."""
        self.is_expanded = True
        actions = self.state.actions()

        for i in range(len(proba_model)):  # Mask illegal move
            if i not in actions:
                proba_model[i] = 0.0  # P = 0 (ensures U = 0)
                self._children_Q[i] = -np.inf  # Q = -inf (double safety)

        self.children_P = np.asarray(proba_model)
        self.update_children_U()

    def select(self) -> RootNode:
        """Select leaf node according to Q + U."""
        current = self
        while current.is_expanded:
            current = current.best_child()
        return current

    def backup(self, value: float) -> None:
        """Update V N recursively."""
        current = self
        while isinstance(current, Node):
            current.allow_update = False
            current.N += 1
            current.sum_V += value
            value = -value  # Negate value as we go up (players alternate)
            assert current.parent is not None
            assert current.action is not None
            current.parent.update_children_Q(current.action)
            current.parent.need_update_U = True
            current.allow_update = True
            current = current.parent
        current.N += 1
        current.sum_V += value

    def best_child(self) -> Node:
        """Return best child according to Q + U."""
        if self.need_update_U:  # Lazy update
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
    action: int
    parent: RootNode

    def __init__(self, state: Board, parent: RootNode, action: int) -> None:
        super().__init__(state)
        self.action = action
        self.parent = parent

    @property
    def N(self) -> int:
        """Number of visit getter."""
        assert self.parent is not None
        return self.parent.children_N[self.action]

    @N.setter
    def N(self, value: int) -> None:
        """Number of visit setter."""
        assert self.parent is not None
        self.parent.sum_children_N += value - self.parent.children_N[self.action]
        self.parent.children_N[self.action] = value
        if self.allow_update:
            self.parent.update_children_Q(self.action)
            self.parent.need_update_U = True

    @property
    def sum_V(self) -> float:
        """Sum of values getter."""
        assert self.parent is not None
        return self.parent.children_sum_V[self.action]

    @sum_V.setter
    def sum_V(self, value: float) -> None:
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
    n_iter: int | None = 100,
    time_limit: float | None = None,
) -> RootNode:
    """
    Perform Monte-Carlo Tree Search.

    Args:
        root (RootNode): RootNode
        model (HexNet): DL model to predict value and proba
        batch_size (int): Size of batch for model evaluation
        timeout (float): Max time to wait for a batch (seconds)
        n_iter (int, optional): Number of iterations of MCTS.
        time_limit (float, optional): Max time for total search (seconds).

    Returns:
        RootNode: the tree created

    """
    # First Root eval
    probas, values = perform_model(model, [root])
    root.expand(probas[0])
    root.backup(values[0].item())

    batch_leaves: list[RootNode] = []
    global_start_time = time.perf_counter()
    batch_start_time = global_start_time

    idx_iter = 0
    while True:
        # Check termination conditions
        if n_iter is not None and idx_iter >= n_iter:
            break
        if (
            time_limit is not None
            and (time.perf_counter() - global_start_time) >= time_limit
        ):
            break

        # Find best child Q + U
        leaf = root.select()
        batch_leaves.append(leaf)
        leaf.sum_V -= 10  # Virtual loss

        # Check if we should run the model
        reached_batch = len(batch_leaves) == batch_size
        batch_timeout = (time.perf_counter() - batch_start_time) >= timeout
        last_iter = n_iter is not None and idx_iter == n_iter - 1
        search_timeout = (
            time_limit is not None
            and (time.perf_counter() - global_start_time) >= time_limit
        )

        if len(batch_leaves) > 0 and (
            reached_batch or batch_timeout or last_iter or search_timeout
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
            batch_start_time = time.perf_counter()
            batch_leaves = []

        idx_iter += 1

    return root


def MCTS_multi(
    roots: list[RootNode],
    model: HexNet,
    n_iter: int = 100,
) -> None:
    """
    Perform MCTS on multiple roots in parallel (vectorized).

    Args:
        roots (list[RootNode]): List of RootNodes to expand
        model (HexNet): model
        n_iter (int): Number of iterations

    """
    # Filter roots that are already finished (safety)
    active_roots = [r for r in roots if r.state.has_won == 0]
    if not active_roots:
        return

    # First Root eval for those not expanded
    to_expand = [r for r in active_roots if not r.is_expanded]
    if to_expand:
        probas, values = perform_model(model, to_expand)
        for i, root in enumerate(to_expand):
            root.expand(probas[i])
            root.backup(values[i].item())

    for _ in range(n_iter):
        # 1. Select a leaf for each active root
        leaves = [root.select() for root in active_roots]

        # 2. Batch evaluate all leaves
        probas, values = perform_model(model, leaves)

        # 3. Expand and backup
        for i, leaf in enumerate(leaves):
            if leaf.state.has_won == 0:
                leaf.expand(probas[i])
            leaf.backup(values[i].item())


def generate_data(
    model: HexNet,
    n_games: int,
    n_random_plays: int = 1,
    n_iter: int = 10,
    show: bool = False,
    save_path: str | None = None,
    temperature: float = 1.0,
    temperature_threshold: int = 15,
    batch_size: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate data for training according to model using parallel games.

    Args:
        model (HexNet): model that predict
        n_games (int): number of games generated
        n_random_plays (int, optional): first plays that are chosen randomly.
        n_iter (int, optional): number of iteration in MCTS.
        show (bool, optional): Show with pygame (only for batch_size=1).
        save_path (str, optional): Path to save data.
        temperature (float, optional): Temperature for move selection.
        temperature_threshold (int, optional): Move count threshold for temp -> 0.
        batch_size (int, optional): Number of parallel games.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: boards, policies, values

    """
    boards: list[np.ndarray] = []
    policies: list[np.ndarray] = []
    values_list: list[float] = []

    # Current state of each parallel game
    class GameState:
        def __init__(self) -> None:
            self.board = Board(model.n)
            self.reset()

        def reset(self) -> None:
            self.board.reset()
            for _ in range(n_random_plays):
                self.board.play_random()
            self.root = RootNode(self.board)
            self.history: list = []
            self.move_count = 0

    if show and batch_size > 1:
        print("Warning: show=True is only supported for batch_size=1. Disabling show.")
        show = False

    move_queue: queue.Queue | None = None
    if show:
        move_queue = queue.Queue()
        threading.Thread(
            target=lambda: HexBoard(
                n=model.n,
                player_white=QueuePlayer(move_queue),
                player_black=QueuePlayer(move_queue),
            ).run(),
            daemon=True,
        ).start()

    active_games: list[GameState] = []
    finished_games = 0
    started_games = 0

    while finished_games < n_games:
        # 1. Fill active games up to batch_size
        while len(active_games) < batch_size and started_games < n_games:
            game = GameState()
            active_games.append(game)
            started_games += 1
            if show and move_queue:
                move_queue.put_nowait("reset")

        if not active_games:
            break

        # 2. Run MCTS on all active games in one batch
        MCTS_multi([g.root for g in active_games], model, n_iter=n_iter)

        # 3. Sample and play actions for each game
        to_remove = []
        for game in active_games:
            game.history.append(game.root)

            # Sample action
            tau = temperature if game.move_count < temperature_threshold else 0.1
            game.root = game.root.sample_child(temperature=tau)
            game.move_count += 1

            if show and move_queue and game.root.action is not None:
                move_queue.put_nowait(game.root.action)

            # 4. Check if game is finished
            if game.root.state.has_won != 0:
                won = game.root.state.has_won
                # Collect data from history
                for node in game.history:
                    boards.append(node.state.to_numpy())
                    policies.append(node.get_policy())
                    values_list.append(won * node.state.turn)

                finished_games += 1
                to_remove.append(game)
                if finished_games % 10 == 0:
                    print(f"Games finished: {finished_games}/{n_games}")

        # Remove finished games
        for game in to_remove:
            active_games.remove(game)

    # Convert to numpy
    boards_array = np.array(boards, dtype=np.float32)
    policies_array = np.array(policies, dtype=np.float32)
    values_array = np.array(values_list, dtype=np.float32)

    if save_path:
        pathlib.Path(save_path).mkdir(exist_ok=True, parents=True)
        timestamp = int(time.time())
        filename = pathlib.Path(save_path) / f"selfplay_data_{timestamp}.npz"
        np.savez_compressed(
            filename,
            boards=boards_array,
            policies=policies_array,
            values=values_array,
            n_games=n_games,
            n_iter=n_iter,
        )
        print(f"Data saved to {filename}")

    return boards_array, policies_array, values_array
