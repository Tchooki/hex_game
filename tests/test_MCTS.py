import unittest
import numpy as np

from solve.MCTS import get_random_transformation


class TestMCTSMethods(unittest.TestCase):

    def test_reflexion(self):
        data = np.random.randint(-10,10,(10,10))
        for _ in range(20):
            trans = get_random_transformation()
            test = (data == trans(trans(data.copy()))).any()
            self.assertTrue(test)


    def test_reflexion_with_flatten(self):
        data = np.random.randint(-10,10,(10,10)).reshape(-1)
        for _ in range(20):
            trans = get_random_transformation()
            data_trans = data.copy().reshape(10,-1)
            data_trans = trans(trans(data_trans)).reshape(-1)
            test = (data == data_trans).any()
            self.assertTrue(test)

if __name__ == '__main__':
    unittest.main()