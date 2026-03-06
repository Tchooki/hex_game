import numpy as np
import torch

from hex_game.ai.mcts import MCTS, Node, RootNode
from hex_game.game.board import Board


def test_root_node_expansion():
    b = Board(3)
    root = RootNode(b)
    # Mock proba from model
    proba = np.zeros(9)
    proba[0] = 0.5
    proba[1] = 0.5

    root.expand(proba)
    assert root.is_expanded
    # Values should be normalized over legal moves
    assert np.isclose(root.children_P[0], 0.5)
    assert np.isclose(root.children_P[1], 0.5)
    assert (root.children_P[2:] == 0).all()


def test_backup_logic():
    b = Board(3)
    root = RootNode(b)
    child = Node(b.light_copy(), parent=root, action=0)
    root.children[0] = child

    # Backup value 1.0 from child
    # In MCTS.backup(value), it's called on the leaf.
    # If leaf is child, it updates child.N, child.sum_V, then parent.update_children_Q(action)
    child.backup(1.0)

    assert child.N == 1
    assert np.isclose(child.sum_V, 1.0)
    assert root.N == 1
    # value is negated when going up: root.sum_V += -1.0
    assert np.isclose(root.sum_V, -1.0)
    assert root.children_N[0] == 1
    assert np.isclose(root.children_sum_V[0], 1.0)


def test_selection_logic():
    b = Board(3)
    root = RootNode(b)
    # Set probability of action 0 to 1.0, others to 0.0
    # This makes U(0) much larger than others, and U(others) will be 0
    proba = np.zeros(9)
    proba[0] = 1.0
    root.expand(proba)

    # Manually set some values to force selection
    root.children_N[0] = 10
    root.children_sum_V[0] = 10.0  # Q = 1.0

    root.children_N[1] = 1
    root.children_sum_V[1] = 0.0  # Q = 0.0

    selected = root.select()
    assert isinstance(selected, Node)
    assert selected.action == 0


def test_mcts_minimal_solve():
    # On a 2x2 board, check if MCTS runs with a mock model
    from unittest.mock import MagicMock

    n = 2
    b = Board(n)
    root = RootNode(b)

    with MagicMock() as mock_model:
        mock_model.n = n
        # Mocking forward: (probas, values)
        # perform_model does: probas, values = model.forward(inputs)
        # where inputs is (batch, 1, n, n)
        mock_model.forward.return_value = (torch.zeros(1, 4), torch.zeros(1, 1))

        MCTS(root, mock_model, n_iter=10, batch_size=1)

    assert root.N > 0
    assert root.is_expanded
    assert len(root.children) > 0
