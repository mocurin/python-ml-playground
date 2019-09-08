import unittest
import neural_network as neural


def make_threshold(threshold):
    return lambda x: 1 if x > threshold else 0


class NeuralNetworkTest(unittest.TestCase):
    def test_and(self):
        rectifier = lambda x: max(0, x)
        nn = neural.make_feedforward_neural_network(input_count=2, output_count=1)
        nn.add_layer('linear', matrix=[[1, 1]], bias=[-1])
        nn.add_layer('function', func=rectifier)
        self.assertTrue(nn.check_dimensions())
        self.assertEqual(nn([0, 0]), [0])
        self.assertEqual(nn([0, 1]), [0])
        self.assertEqual(nn([1, 0]), [0])
        self.assertEqual(nn([1, 1]), [1])

    def test_or(self):
        zero_threshold = make_threshold(0)
        nn = neural.make_feedforward_neural_network(input_count=2, output_count=1)
        nn.add_layer('linear', matrix=[[1, 1]], bias=[0])
        nn.add_layer('function', func=zero_threshold)
        self.assertTrue(nn.check_dimensions())
        self.assertEqual(nn([0, 0]), [0])
        self.assertEqual(nn([0, 1]), [1])
        self.assertEqual(nn([1, 0]), [1])
        self.assertEqual(nn([1, 1]), [1])

    def test_xor(self):
        threshold = make_threshold(0.5)
        nn = neural.make_feedforward_neural_network(input_count=2, output_count=1)
        nn.add_layer('linear', matrix=[[1, -1],
                                       [-1, 1]], bias=[0, 0])
        nn.add_layer('function', func=threshold)
        nn.add_layer('linear', matrix=[[1, 1]], bias=[0])
        nn.add_layer('function', func=threshold)
        self.assertTrue(nn.check_dimensions())
        self.assertEqual(nn([0, 0]), [0])
        self.assertEqual(nn([0, 1]), [1])
        self.assertEqual(nn([1, 0]), [1])
        self.assertEqual(nn([1, 1]), [0])


if __name__ == '__main__':
    unittest.main()
