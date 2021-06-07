from unittest import TestCase

from envs.tictactoe import TicTacToeEnv


class TestTicTacToeEnv(TestCase):

    def test_encode_decode_1(self):
        x_pos_orig = [1, 3, 5]
        o_pos_orig = [2, 8, 4]
        encoded = TicTacToeEnv.encode(x_pos_orig, o_pos_orig)
        x_pos_restored, o_pos_restored = TicTacToeEnv.decode(encoded)

        self.assertEqual(x_pos_restored, sorted(x_pos_orig))
        self.assertEqual(o_pos_restored, sorted(o_pos_orig))

    def test_encode_decode_2(self):
        x_pos_orig = []
        o_pos_orig = []
        encoded = TicTacToeEnv.encode(x_pos_orig, o_pos_orig)
        x_pos_restored, o_pos_restored = TicTacToeEnv.decode(encoded)

        self.assertEqual(x_pos_restored, sorted(x_pos_orig))
        self.assertEqual(o_pos_restored, sorted(o_pos_orig))

    def test_encode_decode_3(self):
        x_pos_orig = [4]
        o_pos_orig = [8]
        encoded = TicTacToeEnv.encode(x_pos_orig, o_pos_orig)
        x_pos_restored, o_pos_restored = TicTacToeEnv.decode(encoded)

        self.assertEqual(x_pos_restored, sorted(x_pos_orig))
        self.assertEqual(o_pos_restored, sorted(o_pos_orig))
