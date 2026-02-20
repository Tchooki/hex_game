import argparse
import cProfile
import sys
from pathlib import Path

import torch

from hex_game.ai.model import HexNet
from hex_game.ui.display import HexBoard


def play_ai(run_name: str, n: int = 11, time_limit: float = 1.0):
    """Play against the best model from a run."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path("models") / run_name / "best_model.pth"

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    print(f"Loading model from {model_path}")
    model = HexNet(n=n).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Computer plays as BLACK (second player) by default
    game = HexBoard(
        n=n,
        model=model,
        ai_color=-1,  # BLACK
        time_limit=time_limit,
        temperature=1.0,
    )
    game.run(mode="ai")


def main():
    parser = argparse.ArgumentParser(description="Hex Game with AI")
    parser.add_argument("--ai", action="store_true", help="Play against AI")
    parser.add_argument(
        "--run", type=str, default="hex11x11", help="Run name for the model"
    )
    parser.add_argument("--size", type=int, default=11, help="Board size")
    parser.add_argument("--time", type=float, default=3.0, help="AI thought time (s)")
    parser.add_argument("--profile", action="store_true", help="Run profiling")

    args = parser.parse_args()

    if args.profile:
        from hex_game.ai.mcts import generate_data  # noqa: F401

        cProfile.run(
            "generate_data(HexNet().cuda(), 5, show = False, n_iter=400)",
            filename="profile/profile_batch16_iter400_pos_refactored.stats",
        )
    elif args.ai:
        play_ai(args.run, n=args.size, time_limit=args.time)
    else:
        # Default: basic play
        HexBoard(n=args.size).run(mode="basic")


if __name__ == "__main__":
    main()
