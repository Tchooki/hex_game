#!/usr/bin/env -S uv run --script

import argparse
import sys
from pathlib import Path

import torch

from hex_game.ai.model import HexNet
from hex_game.ui.board_view import HexBoard
from hex_game.ui.players import AIPlayer, BasePlayer, HumanPlayer


def load_model(
    path: str | Path, n: int, n_res_block: int, device: torch.device
) -> HexNet:
    """Load a model from a specific path."""
    path = Path(path)
    if not path.exists():
        print(f"Error: Model not found at {path}")
        sys.exit(1)

    print(f"Loading model from {path}")
    model = HexNet(n=n, n_res_block=n_res_block).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


def play_ai(
    run_name: str | None = None,
    n: int = 11,
    n_res_block: int = 10,
    time_limit: float = 1.0,
    self_play: bool = False,
    m1_path: str | None = None,
    m2_path: str | None = None,
) -> None:
    """Play with specific models or the best model from a run."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    player_white: BasePlayer
    player_black: BasePlayer

    if m1_path:
        model1 = load_model(m1_path, n, n_res_block, device)
        if m2_path:
            model2 = load_model(m2_path, n, n_res_block, device)
            player_white = AIPlayer(
                model=model1, time_limit=time_limit, temperature=0.0
            )
            player_black = AIPlayer(
                model=model2, time_limit=time_limit, temperature=0.0
            )
        else:
            player_white = HumanPlayer()
            player_black = AIPlayer(
                model=model1, time_limit=time_limit, temperature=0.0
            )
    else:
        # Fallback to run_name logic
        run_name = run_name or "hex11x11"
        model_path = Path("models") / run_name / "best_model.pth"
        model = load_model(model_path, n, n_res_block, device)

        player_white = (
            AIPlayer(model=model, time_limit=time_limit, temperature=0.1)
            if self_play
            else HumanPlayer()
        )
        player_black = AIPlayer(
            model=model,
            time_limit=time_limit,
            temperature=0.0,
        )

    game = HexBoard(
        n=n,
        player_white=player_white,
        player_black=player_black,
    )
    game.run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Hex Game with AI")
    parser.add_argument("--ai", action="store_true", help="Play against AI")
    parser.add_argument(
        "--self", action="store_true", help="Watch AI play against itself"
    )
    parser.add_argument("--m1", type=str, help="Path to first model")
    parser.add_argument("--m2", type=str, help="Path to second model (optional)")
    parser.add_argument(
        "--run", type=str, default="hex11x11", help="Run name for the model"
    )
    parser.add_argument("--size", type=int, default=11, help="Board size")
    parser.add_argument(
        "--res-blocks", type=int, default=10, help="Number of residual blocks"
    )
    parser.add_argument("--time", type=float, default=1.0, help="AI thought time (s)")

    args = parser.parse_args()

    if args.ai or args.self or args.m1:
        play_ai(
            run_name=args.run,
            n=args.size,
            n_res_block=args.res_blocks,
            time_limit=args.time,
            self_play=args.self,
            m1_path=args.m1,
            m2_path=args.m2,
        )
    else:
        # Default: basic play (Human vs Human)
        HexBoard(
            n=args.size,
            player_white=HumanPlayer(),
            player_black=HumanPlayer(),
        ).run()


if __name__ == "__main__":
    main()
