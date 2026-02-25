from __future__ import annotations

import queue
from typing import TYPE_CHECKING

from hex_game.game.board import Board

if TYPE_CHECKING:
    from hex_game.ai.model import HexNet


class BasePlayer:
    """Base class for all players."""

    def get_move(self, board: Board, **kwargs) -> int | str | None:
        """Return the next move or None if waiting for interaction."""
        raise NotImplementedError


class HumanPlayer(BasePlayer):
    """A human player that interacts via the GUI."""

    def get_move(self, board: Board, **kwargs) -> int | None:
        return None  # Move is handled by GUI events


class RandomPlayer(BasePlayer):
    """A player that makes random moves."""

    def get_move(self, board: Board, **kwargs) -> int:
        pos, _ = board.play_random()
        return pos


class AIPlayer(BasePlayer):
    """An AI player using MCTS and a neural network."""

    def __init__(
        self,
        model: HexNet,
        time_limit: float = 1.0,
        temperature: float = 1.0,
        temp_threshold: int = 10,
    ):
        self.model = model
        self.time_limit = time_limit
        self.temperature = temperature
        self.temp_threshold = temp_threshold

    def get_move(self, board: Board, **kwargs) -> int:
        from hex_game.ai.mcts import MCTS, RootNode

        move_count = kwargs.get("move_count", 0)

        # MCTS
        root = RootNode(board)
        MCTS(root, self.model, time_limit=self.time_limit, n_iter=None)

        temp = self.temperature if move_count < self.temp_threshold else 0.1
        action = root.sample_action(temp)

        return action


class QueuePlayer(BasePlayer):
    """A player that gets moves from a queue (used for training visualization)."""

    def __init__(self, move_queue: queue.Queue):
        self.move_queue = move_queue

    def get_move(self, board: Board, **kwargs) -> int | str | None:
        if not self.move_queue.empty():
            action = self.move_queue.get_nowait()
            self.move_queue.task_done()
            return action
        return None
