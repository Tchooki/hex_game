import cProfile

import torch

from hex_game.ai.mcts import MCTS_multi, RootNode
from hex_game.ai.model import HexNet
from hex_game.game.board import Board


def profile_mcts_multi() -> None:
    """
    Script pour profiler l'exécution de MCTS_multi (version vectorisée).

    On initialise plusieurs plateaux et on lance une recherche MCTS multi-racines.
    """
    n = 11
    batch_size = 16
    # On initialise plusieurs plateaux
    roots = [RootNode(Board(n)) for _ in range(batch_size)]
    # Modèle léger
    model = HexNet(n=n, n_res_block=2)
    if torch.cuda.is_available():
        model = model.cuda()

    print(
        f"Lancement du profiling MCTS_multi sur {batch_size} racines (plateau {n}x{n})..."
    )

    profiler = cProfile.Profile()
    profiler.enable()

    # On lance 100 itérations de MCTS sur chaque racine en parallèle
    MCTS_multi(roots, model, n_iter=100)

    profiler.disable()

    # Sauvegarde des résultats
    stats_file = "mcts_multi_profile.prof"
    profiler.dump_stats(stats_file)
    print(f"Profiling terminé. Résultats sauvegardés dans '{stats_file}'.")
    print("Pour visualiser, lancez : uv run snakeviz mcts_multi_profile.prof")


if __name__ == "__main__":
    profile_mcts_multi()
