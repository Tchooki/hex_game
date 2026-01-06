"""Run game"""
from graphics.display import HexBoard
import cProfile

from solve.MCTS import generate_data
from solve.model import HexNet




if __name__ == '__main__':
    # HexBoard(4).run(debug=False)

    model = HexNet().cuda()
    cProfile.run("generate_data(HexNet().cuda(),5, show = False, n_iter=200)", filename="profile_Codu_minibatch16_200_nopygame.stats")
