"""Run game"""

import cProfile

# from hex_game.ai.mcts import generate_data
# from hex_game.ai.model import HexNet

# from hex_game.ui.display import HexBoard


def main():
    # HexBoard(4).run(debug=False)

    # model = HexNet().cuda()
    # generate_data(model, 2, show=True, n_iter=200)
    cProfile.run(
        "generate_data(HexNet().cuda(), 5, show = False, n_iter=400)",
        filename="profile/profile_batch16_iter400_pos_refactored.stats",
    )


if __name__ == "__main__":
    main()
