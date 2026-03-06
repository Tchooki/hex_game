import cProfile

import torch

from hex_game.ai.mcts import generate_data
from hex_game.ai.model import HexNet


def profile_generate_data() -> None:
    """
    Script pour profiler le processus complet de génération de données (self-play).
    """
    n = 11
    # On génère peu de jeux pour le profilage (par ex. 2), mais avec assez d'itérations
    n_games = 2
    n_iter = 100
    batch_size = 16

    # Modèle léger
    model = HexNet(n=n, n_res_block=2)
    if torch.cuda.is_available():
        model = model.cuda()

    print(
        f"Lancement du profiling generate_data ({n_games} jeux, {n_iter} itérations, plateau {n}x{n})..."
    )

    profiler = cProfile.Profile()
    profiler.enable()

    # Appel de generate_data
    generate_data(
        model, n_games=n_games, n_iter=n_iter, batch_size=batch_size, show=False
    )

    profiler.disable()

    # Sauvegarde des résultats
    stats_file = "generate_data_profile.prof"
    profiler.dump_stats(stats_file)
    print(f"Profiling terminé. Résultats sauvegardés dans '{stats_file}'.")
    print("Pour visualiser, lancez : uv run snakeviz generate_data_profile.prof")


if __name__ == "__main__":
    profile_generate_data()
