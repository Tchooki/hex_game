#!/usr/bin/env -S uv run --script

import argparse
from typing import Any, cast

import torch
from tqdm import tqdm

from hex_game.ai.mcts import MCTS_multi, RootNode
from hex_game.ai.model import HexNet
from hex_game.game.board import BLACK, WHITE, Board
from hex_game.main import load_model


def evaluate_models(
    model_challenger: HexNet,
    model_champion: HexNet,
    n_games: int = 40,
    n_mcts_iter: int = 100,
    temperature: float = 0.1,
    batch_size: int = 40,
) -> tuple[int, int]:
    """
    Evaluate two models against each other.

    Model challenger plays WHITE for the first half of games, BLACK for the rest.

    Args:
        model_challenger: The newly trained model
        model_champion: The current best model
        n_games: Total number of games to play
        n_mcts_iter: Number of MCTS iterations per move
        temperature: Temperature for move selection (usually low for eval)
        batch_size: Number of parallel games

    Returns:
        tuple[int, int]: wins for challenger, wins for champion

    """
    assert n_games % 2 == 0, "n_games must be even for fair evaluation"
    half_games = n_games // 2

    # Track results
    challenger_wins = 0
    champion_wins = 0
    pbar = tqdm(total=n_games, desc="Eval [0-0]", unit="game")

    active_games: list[dict[str, Any]] = []
    started_games = 0
    finished_games = 0

    while finished_games < n_games:
        # 1. Fill active games up to batch_size
        while len(active_games) < batch_size and finished_games < n_games:
            board = Board(model_challenger.n)

            # First half: challenger is WHITE. Second half: challenger is BLACK
            challenger_color = WHITE if started_games < half_games else BLACK

            game = {
                "root": RootNode(board),
                "challenger_color": challenger_color,
            }
            active_games.append(game)
            started_games += 1

        if not active_games:
            break

        # 2. Separate roots based on whose turn it is
        challenger_roots: list[RootNode] = []
        champion_roots: list[RootNode] = []

        for game in active_games:
            current_player = cast("RootNode", game["root"]).state.turn
            if current_player == game["challenger_color"]:
                challenger_roots.append(cast("RootNode", game["root"]))
            else:
                champion_roots.append(cast("RootNode", game["root"]))

        # 3. Run MCTS for each model's turn
        if challenger_roots:
            MCTS_multi(challenger_roots, model_challenger, n_iter=n_mcts_iter)
        if champion_roots:
            MCTS_multi(champion_roots, model_champion, n_iter=n_mcts_iter)

        # 4. Sample and play actions for each game
        to_remove = []
        for game in active_games:
            game["root"] = cast("RootNode", game["root"]).sample_child(
                temperature=temperature
            )

            # Check if game is finished
            if cast("RootNode", game["root"]).state.has_won != 0:
                won = cast("RootNode", game["root"]).state.has_won

                if won == game["challenger_color"]:
                    challenger_wins += 1
                else:
                    champion_wins += 1

                finished_games += 1
                to_remove.append(game)
                pbar.set_description(f"Eval [{challenger_wins}-{champion_wins}]")
                pbar.update(1)

        # Remove finished games
        for game in to_remove:
            active_games.remove(game)

    pbar.close()
    num_played = challenger_wins + champion_wins
    if num_played > 0:
        win_rate = challenger_wins / num_played
        print(f"Eval complete! Challenger win rate: {win_rate:.1%}")

    return challenger_wins, champion_wins


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate models")
    parser.add_argument("--m1", type=str, required=True, help="Path to first model")
    parser.add_argument("--m2", type=str, required=True, help="Path to second model")
    parser.add_argument("--size", type=int, default=11, help="Board size")
    parser.add_argument(
        "--res-blocks", type=int, default=10, help="Number of residual blocks"
    )
    parser.add_argument("--n_games", type=int, default=40, help="Number of games")
    parser.add_argument(
        "--n_mcts_iter", type=int, default=100, help="Number of MCTS iterations"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1, help="Temperature for move selection"
    )
    parser.add_argument("--batch_size", type=int, default=40, help="Batch size")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Model 1:")
    model1 = load_model(args.m1, args.size, args.res_blocks, device)
    print("Model 2:")
    model2 = load_model(args.m2, args.size, args.res_blocks, device)

    wins1, wins2 = evaluate_models(
        model1,
        model2,
        n_games=args.n_games,
        n_mcts_iter=args.n_mcts_iter,
        temperature=args.temperature,
        batch_size=args.batch_size,
    )

    print(f"Model 1 wins: {wins1}")
    print(f"Model 2 wins: {wins2}")
