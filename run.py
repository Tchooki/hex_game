"""Run game"""

import cProfile

from solve.MCTS import generate_data
from solve.model import HexNet

if __name__ == "__main__":
    # HexBoard(4).run(debug=False)

    model = HexNet().cuda()
    # generate_data(model, 2, show=True, n_iter=200)
    cProfile.run(
        "generate_data(HexNet().cuda(), 5, show = False, n_iter=400)",
        filename="profile/profile_batch16_iter400.stats",
    )
