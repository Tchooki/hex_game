
from solve.model import HexNet
from game.board import Board
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def train(n_res_block = 10):
    net = HexNet(n_res_block=n_res_block)
    optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)

    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()
    batch_size = 32
    n_epochs = 100

    for epoch in range(n_epochs):
        b = Board(11)
        board = np.array(b._board).reshape(11,-1)
        board = torch.from_numpy(board).unsqueeze(0).unsqueeze(0).float()
        # board.unsqueeze(0).shape
        board.size()
        proba, value = net(board)
        loss = policy_loss_fn(proba, b.action_space) + value_loss_fn(value, torch.tensor([b.has_won], dtype=torch.float32))