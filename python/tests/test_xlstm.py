import unittest
import torch
from xlstm import PySLSTMLayer, PyMLSTMLayer, PyXLSTMBlock, PyXLSTMModel

class TestXLSTM(unittest.TestCase):
    def test_slstm_layer(self):
        input_size = 10
        hidden_size = 20
        batch_size = 5

        slstm_layer = PySLSTMLayer(input_size, hidden_size)
        input_data = torch.randn(batch_size, input_size)
        h_prev = torch.zeros(batch_size, hidden_size)
        c_prev = torch.zeros(batch_size, hidden_size)
        n_prev = torch.zeros(batch_size, hidden_size)

        h, c, n = slstm_layer(input_data, h_prev, c_prev, n_prev)

        self.assertEqual(h.shape, (batch_size, hidden_size))
        self.assertEqual(c.shape, (batch_size, hidden_size))
        self.assertEqual(n.shape, (batch_size, hidden_size))

    def test_mlstm_layer(self):
        input_size = 10
        hidden_size = 20
        batch_size = 5

        mlstm_layer = PyMLSTMLayer(input_size, hidden_size)
        input_data = torch.randn(batch_size, input_size)
        h_prev = torch.zeros(batch_size, hidden_size)
        C_prev = torch.zeros(batch_size, hidden_size, hidden_size)
        n_prev = torch.zeros(batch_size, hidden_size)

        h, C, n = mlstm_layer(input_data, h_prev, C_prev, n_prev)

        self.assertEqual(h.shape, (batch_size, hidden_size))
        self.assertEqual(C.shape, (batch_size, hidden_size, hidden_size))
        self.assertEqual(n.shape, (batch_size, hidden_size))

    def test_xlstm_block(self):
        input_size = 10
        hidden_size = 20
        proj_size = 15
        use_mlstm = True
        batch_size = 5

        xlstm_block = PyXLSTMBlock(input_size, hidden_size, proj_size, use_mlstm)
        input_data = torch.randn(batch_size, input_size)
        h_prev = torch.zeros(batch_size, hidden_size)
        c_prev = torch.zeros(batch_size, hidden_size)
        C_prev = torch.zeros(batch_size, hidden_size, hidden_size)
        n_prev = torch.zeros(batch_size, hidden_size)

        h, c, C, n = xlstm_block(input_data, h_prev, c_prev, C_prev, n_prev)

        self.assertEqual(h.shape, (batch_size, hidden_size))
        self.assertEqual(c.shape, (batch_size, hidden_size))
        self.assertEqual(C.shape, (batch_size, hidden_size, hidden_size))
        self.assertEqual(n.shape, (batch_size, hidden_size))

    def test_xlstm_model(self):
        input_size = 10
        hidden_size = 20
        proj_size = 15
        use_mlstm_vec = [True, False, True]
        num_layers = len(use_mlstm_vec)
        batch_size = 5
        seq_length = 3

        xlstm_model = PyXLSTMModel(input_size, hidden_size, proj_size, use_mlstm_vec, num_layers)
        input_data = torch.randn(batch_size, seq_length, input_size)
        output = xlstm_model(input_data)

        self.assertEqual(output.shape, (batch_size, hidden_size))

if __name__ == '__main__':
    unittest.main()