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


def get_next_gen_id(run_name: str, data_dir: str = "data") -> int:
    """Find the next generation ID by scanning the data directory."""
    path = Path(data_dir) / run_name
    if not path.exists():
        return 1
    gens = []
    for d in path.iterdir():
        if d.is_dir() and d.name.startswith("gen_"):
            try:
                gens.append(int(d.name.split("_")[1]))
            except (IndexError, ValueError):
                continue
    return max(gens) + 1 if gens else 1


def load_window_data(
    run_name: str,
    window_size: int = 5,
    data_dir: str = "data",
) -> HexDataset | None:
    """Load data from the last N generations (sliding window)."""
    path = Path(data_dir) / run_name
    if not path.exists():
        return None

    # Get all gen folders and sort them descending
    gen_folders = sorted(
        [d for d in path.iterdir() if d.is_dir() and d.name.startswith("gen_")],
        key=lambda x: int(x.name.split("_")[1]),
        reverse=True,
    )

    # Take the last window_size
    target_folders = gen_folders[:window_size]
    if not target_folders:
        return None

    all_boards, all_policies, all_values = [], [], []
    for folder in target_folders:
        for file in folder.glob("*.npz"):
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
) -> tuple[float, float, float]:
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0
    total_loss_p = 0.0
    total_loss_v = 0.0

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
        total_loss_p += loss_p.item()
        total_loss_v += loss_v.item()

    avg_loss = total_loss / len(loader)
    avg_loss_p = total_loss_p / len(loader)
    avg_loss_v = total_loss_v / len(loader)

    return avg_loss, avg_loss_p, avg_loss_v


def train(
    run_name: str = "hex11x11",
    n: int = 11,
    n_res_block: int = 10,
    n_selfplay_games: int = 10,
    n_mcts_iter: int = 100,
    window_size: int = 5,
    n_epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    data_dir: str = "data",
    models_dir: str = "models",
):
    """Main training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for run: {run_name}")

    # Paths
    run_models_dir = Path(models_dir) / run_name
    run_models_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = run_models_dir / "best_model.pth"

    # Initialize model
    model = HexNet(n=n, n_res_block=n_res_block).to(device)

    # Load existing best model if it exists
    if best_model_path.exists():
        print(f"Loading best model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()

    while True:  # Continuous training cycles
        gen_id = get_next_gen_id(run_name, data_dir)
        print(f"\n--- Starting Generation {gen_id} ---")

        # Phase 1: Generate new data via self-play
        gen_data_path = Path(data_dir) / run_name / f"gen_{gen_id}"
        print(f"Generating {n_selfplay_games} games in {gen_data_path}...")
        generate_data(
            model,
            n_games=n_selfplay_games,
            n_iter=n_mcts_iter,
            save_path=str(gen_data_path),
        )

        # Phase 2: Load data from window
        dataset = load_window_data(run_name, window_size, data_dir)
        if dataset is None:
            print("No data found, skipping training.")
            continue

        print(f"Loaded total of {len(dataset)} positions (window={window_size}).")
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Phase 3: Train
        print(f"Training for {n_epochs} epochs...")
        total_steps_this_gen = 0
        for epoch in range(1, n_epochs + 1):
            avg_loss, avg_p, avg_v = train_epoch(
                model,
                loader,
                optimizer,
                policy_loss_fn,
                value_loss_fn,
                device,
            )
            total_steps_this_gen += len(loader)

            print(
                f"Epoch {epoch}/{n_epochs} - Loss: {avg_loss:.4f} "
                f"(P: {avg_p:.4f}, V: {avg_v:.4f})"
            )

        model_path = run_models_dir / f"model_{gen_id}.pth"
        torch.save(model.state_dict(), model_path)
        torch.save(model.state_dict(), best_model_path)
        print(f"Model saved as {model_path} and updated best_model.pth")


if __name__ == "__main__":
    # Small parameters for testing
    train(
        run_name="test_run",
        n=5,
        n_res_block=5,
        n_selfplay_games=2,
        n_mcts_iter=20,
        n_epochs=2,
        batch_size=16,
    )
