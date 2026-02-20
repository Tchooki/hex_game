from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from hex_game.ai.mcts import generate_data
from hex_game.ai.model import HexNet


class HexDataset(Dataset):
    """Dataset for Hex game training data."""

    def __init__(self, boards: np.ndarray, policies: np.ndarray, values: np.ndarray):
        self.boards = torch.from_numpy(boards).unsqueeze(1).float()
        self.policies = torch.from_numpy(policies).float()
        self.values = torch.from_numpy(values).unsqueeze(1).float()

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx):
        return self.boards[idx], self.policies[idx], self.values[idx]


def load_all_data(data_dir: str = "data") -> HexDataset | None:
    """Load all .npz data files from a directory."""
    path = Path(data_dir)
    if not path.exists():
        return None

    files = list(path.glob("*.npz"))
    if not files:
        return None

    all_boards = []
    all_policies = []
    all_values = []

    for file in files:
        try:
            data = np.load(file)
            all_boards.append(data["boards"])
            all_policies.append(data["policies"])
            all_values.append(data["values"])
        except (KeyError, ValueError, OSError) as e:
            print(f"Error loading {file}: {e}")

    if not all_boards:
        return None

    return HexDataset(
        np.concatenate(all_boards, axis=0),
        np.concatenate(all_policies, axis=0),
        np.concatenate(all_values, axis=0),
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    policy_loss_fn: nn.Module,
    value_loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0

    for boards, policies, values in loader:
        boards, policies, values = (
            boards.to(device),
            policies.to(device),
            values.to(device),
        )

        optimizer.zero_grad()
        p_pred, v_pred = model(boards)

        # CrossEntropyLoss expects class indices or probabilities
        # Here policies are already probabilities from MCTS
        loss_p = policy_loss_fn(p_pred, policies)
        loss_v = value_loss_fn(v_pred, values)
        loss = loss_p + loss_v

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def train(
    n: int = 11,
    n_res_block: int = 10,
    n_selfplay_games: int = 10,
    n_mcts_iter: int = 100,
    n_epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    data_dir: str = "data",
    checkpoint_path: str = "model_checkpoint.pth",
):
    """Main training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    model = HexNet(n=n, n_res_block=n_res_block).to(device)

    # Load existing checkpoint if it exists
    if Path(checkpoint_path).exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # BinaryCrossEntropy with logits for policy if needed,
    # but CrossEntropyLoss with soft targets (probs) works in modern torch
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()

    for cycle in range(1, 100):  # Continuous training cycles
        print(f"\n--- Starting Cycle {cycle} ---")

        # Phase 1: Generate new data via self-play
        print(f"Generating {n_selfplay_games} self-play games...")
        generate_data(
            model,
            n_games=n_selfplay_games,
            n_iter=n_mcts_iter,
            save_path=data_dir,
        )

        # Phase 2: Load all data
        dataset = load_all_data(data_dir)
        if dataset is None:
            print("No data found, skipping training.")
            continue

        print(f"Loaded total of {len(dataset)} positions for training.")
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Phase 3: Train
        print(f"Training for {n_epochs} epochs...")
        for epoch in range(1, n_epochs + 1):
            avg_loss = train_epoch(
                model,
                loader,
                optimizer,
                policy_loss_fn,
                value_loss_fn,
                device,
            )
            print(f"Epoch {epoch}/{n_epochs} - Avg Loss: {avg_loss:.4f}")

        # Phase 4: Save checkpoint
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model saved to {checkpoint_path}")


if __name__ == "__main__":
    # Small parameters for testing the loop
    train(
        n=5,  # Board size 5 for speed
        n_res_block=5,
        n_selfplay_games=2,
        n_mcts_iter=20,
        n_epochs=5,
        batch_size=16,
    )
